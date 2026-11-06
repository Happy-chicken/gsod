#!/usr/bin/env python
"""Train the GS slot model (PyTorch Lightning, Weights & Biases).

Examples:
    python train.py --dataset primordia
    python train.py --dataset clevr567          # data/clevr/clevr_567_train
    WANDB_MODE=offline python train.py ...      # log locally, sync later

One scene per step; schedules, validation and saving are in epochs; the
learning rate warms up linearly, then decays (cosine). Checkpoints use the
uORF layout for latest, best and epoch_<n>;
--resume continues from one.

    val      held out of the training folder: val/psnr picks the best
             checkpoint
    monitor  the first MONITOR_SCENES test scenes (Primordia: val), with GT
             masks: watched only

W&B: train/* (losses, gradient norm, background attention share), val/*
(val PSNR; monitor ARI, FG-ARI, NV-ARI), diag/* and slots/* (gsod/utils/visualize.py),
images.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader, Subset

from gsod.configs import DATASETS, BaseConfig, config_for
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model
from gsod.model.evaluate import SEGMENTATION_KEYS, evaluate
from gsod.model.gsod import GSODModel
from gsod.model.losses import GSODLoss
from gsod.utils.io import latest_run_dir, load_networks, save_networks
from gsod.utils.visualize import report, wandb_histograms, wandb_images, wandb_slot_table

WANDB_PROJECT = "gsod"
# Monitor scenes (fixed, comparable across epochs) and how many are drawn.
MONITOR_SCENES = 50
IMAGE_SCENES = 4
DIAG_CHARTS = (
    "obj_recall_view0",
    "bg_on_objects",
    "bg_attn_on_bg",
    "fg_on_bg",
    "occ_object",
    "occ_object_min",
    "occ_other",
    "n_object",
    "n_duplicate",
    "n_haze",
    "n_visible",
    "render_dominance",
    "slot_cos_mean",
    "extent_mean",
    "extent_at_min",
    "extent_at_max",
    "gt_outside_region",
)


class GSODTraining(L.LightningModule):
    """One scene per step: view 0 is a random view of the scene, the others
    are the novel views the encoder does not see."""

    def __init__(
        self,
        model: GSODModel,
        cfg: BaseConfig,
        val_set,
        monitor_set,
        steps_per_epoch: int,
        *,
        resume: str | None = None,
        start_epoch: int = 0,
        best_psnr: float = float("-inf"),
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = GSODLoss(cfg.loss)
        self.val_set = val_set
        self.monitor_set = monitor_set
        self.steps_per_epoch = steps_per_epoch
        self.out = Path(cfg.out_dir)
        self.resume = resume
        self.start_epoch = start_epoch
        self.best_psnr = best_psnr
        self.rng = random.Random(cfg.seed)

    @property
    def epoch(self) -> int:
        """Epochs done, counting those before a resume."""
        return self.start_epoch + self.current_epoch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        # LambdaLR's index i counts this run's steps; the schedule reads the
        # (fractional) epoch after step i + 1, offset by a resume.
        start = self.start_epoch * self.steps_per_epoch
        schedule = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda i: _lr_factor(self.cfg, (start + i + 1) / self.steps_per_epoch),
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }

    def on_fit_start(self) -> None:
        if self.resume is not None:
            load_networks(
                str(self.out), self.resume, {}, self.trainer.optimizers, device=self.device
            )

    def on_train_start(self) -> None:
        if self.resume is None:
            self._validate(0)  # the untrained model, as a baseline

    def training_step(self, item: dict, batch_idx: int) -> torch.Tensor:
        images, cams = item["images"], item["cam2world"]
        ref = self.rng.randrange(images.shape[0])
        order = [ref] + [i for i in range(images.shape[0]) if i != ref]
        images, cams = images[order], cams[order]
        epoch = self.start_epoch + self.global_step / self.steps_per_epoch
        self.model.draw_power = min(1.0, epoch / max(self.cfg.draw_ramp_epochs, 1e-8))
        outputs = self.model(images, cams)
        loss, terms = self.criterion(outputs, images)
        self.log("train/l1", terms["l1"], prog_bar=True, batch_size=1)
        self.log("train/view0", terms["view0"], batch_size=1)
        self.log("train/bg_attn", outputs["attn"][0].detach().mean(), batch_size=1)
        self.log("train/draw_power", self.model.draw_power, batch_size=1)
        self.log("train/attn_tau", self.model.slot_attention.temperature().detach(), batch_size=1)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Before clipping (far above grad_clip: clipping shrinks everything).
        norms = grad_norm(self.model, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"], batch_size=1)

    def on_train_epoch_end(self) -> None:
        epoch = self.epoch + 1  # this epoch is done
        if epoch % self.cfg.eval_every == 0:
            self._validate(epoch)
        if epoch % self.cfg.save_every == 0:
            self._save(f"epoch_{epoch}", epoch)

    def _validate(self, epoch: int) -> None:
        """val PSNR (keeps the best checkpoint); the monitor scenes' FG-ARI,
        statistics, slot table, histograms and images (one pass over them)."""
        if not len(self.val_set):
            return
        with torch.no_grad():
            metrics = evaluate(self.model, self.val_set, self.device)
            rep = report(self.model, self.monitor_set, self.device, IMAGE_SCENES)
        self.logger.log_metrics(
            {
                "val/psnr": metrics["psnr"],
                # Segmentation on the monitor scenes (GT masks): 2D ari /
                # fg_ari on the input view, 3D nv_ari on the novel views.
                **{f"val/{key}": rep.stats.get(key, float("nan")) for key in SEGMENTATION_KEYS},
                **{f"diag/{key}": rep.stats.get(key, float("nan")) for key in DIAG_CHARTS},
                "slots/table": wandb_slot_table(rep.rows),
                **wandb_histograms(rep.rows),
                **wandb_images(rep.images),
            },
            step=self.global_step,
        )
        if metrics["psnr"] > self.best_psnr:
            self.best_psnr = metrics["psnr"]
            self._save("best", epoch)
            summary = self.logger.experiment.summary
            summary["best_psnr"] = self.best_psnr
            summary["best_epoch"] = epoch

    def on_train_end(self) -> None:
        self._save("latest", self.start_epoch + self.trainer.current_epoch)
        print(f"done, wrote {self.out}/latest_*.pth")

    def _save(self, suffix: str, epoch: int) -> None:
        """epoch: the epochs done at this checkpoint (where --resume starts)."""
        save_networks(str(self.out), suffix, self.model.networks(), self.trainer.optimizers)
        (self.out / f"{suffix}_state.json").write_text(
            json.dumps({"epoch": epoch, "best_psnr": self.best_psnr})
        )


def _lr_factor(cfg: BaseConfig, epoch: float) -> float:
    """Linear warm-up to cfg.lr, then cosine decay to cfg.lr_final * cfg.lr
    at cfg.epochs."""
    warm = max(cfg.lr_warmup_epochs, 1e-8)
    if epoch < warm:
        return epoch / warm
    t = min(1.0, (epoch - warm) / max(cfg.epochs - warm, 1e-8))
    return cfg.lr_final + (1.0 - cfg.lr_final) * 0.5 * (1.0 + math.cos(math.pi * t))


def _has_masks(dataset) -> bool:
    return bool(len(dataset)) and dataset[0].get("masks") is not None


def train(model: GSODModel, cfg: BaseConfig, *, resume: str | None = None) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = dict(
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
        n_views=cfg.n_views,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    train_set = make_dataset(cfg.data_root, "train", **dataset_kwargs)
    # CLEVR-567 reads *_mask.png when present (and skips them otherwise);
    # Primordia masks live in the scene folders and need the wrapper.
    val_set = make_dataset(cfg.data_root, "val", load_masks=True, **dataset_kwargs)
    if cfg.dataset_mode == "primordia":
        val_set = WithGTMasks(val_set, cfg.data_root)
    monitor_set, monitor_root = val_set, f"{cfg.data_root} (val)"
    if cfg.test_root:
        test_set = make_dataset(cfg.test_root, "test", load_masks=True, **dataset_kwargs)
        monitor_set = Subset(test_set, range(min(MONITOR_SCENES, len(test_set))))
        monitor_root = f"{cfg.test_root} (test, first {len(monitor_set)})"
    print(f"train: {cfg.data_root} ({len(train_set)} scenes)")
    print(f"val: {cfg.data_root} ({len(val_set)} scenes, GT masks: {_has_masks(val_set)})")
    print(f"monitor: {monitor_root} (GT masks: {_has_masks(monitor_set)})")
    loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    start_epoch, best_psnr = 0, float("-inf")
    if resume is not None:
        # Networks now; the optimizer once Lightning has built it (on_fit_start).
        load_networks(str(out), resume, model.networks(), device="cpu")
        state = json.loads((out / f"{resume}_state.json").read_text())
        start_epoch, best_psnr = state["epoch"], state["best_psnr"]
        print(f"resuming from {out} suffix={resume} after epoch {start_epoch}")
    module = GSODTraining(
        model,
        cfg,
        val_set,
        monitor_set,
        len(train_set),
        resume=resume,
        start_epoch=start_epoch,
        best_psnr=best_psnr,
    )

    logger = WandbLogger(
        project=WANDB_PROJECT, name=out.name, save_dir=str(out), config=asdict(cfg)
    )
    # Lightning's counters (the x axis) get no charts of their own.
    for counter in ("trainer/global_step", "epoch"):
        logger.experiment.define_metric(counter, hidden=True)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.epochs - start_epoch,
        gradient_clip_val=cfg.grad_clip if cfg.grad_clip > 0 else None,
        logger=logger,
        log_every_n_steps=cfg.log_every,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=str(out),
    )
    print(f"epochs={cfg.epochs} steps/epoch={len(train_set)} out={out}")
    trainer.fit(module, loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default="primordia"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None, help="epochs")
    parser.add_argument("--save-every", type=int, default=None, help="epochs")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--attn-iter", type=int, default=None)
    parser.add_argument("--gaussians-per-slot", type=int, default=None)
    parser.add_argument("--decoder-hidden-dim", type=int, default=None)
    parser.add_argument("--resume", default=None, help="checkpoint suffix, e.g. latest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_for(args.dataset)
    if args.resume is not None and args.out is None:
        latest = latest_run_dir(args.dataset)
        if latest is None:
            raise FileNotFoundError(
                "--resume needs --out or an existing timestamped run"
            )
        cfg = replace(cfg, out_dir=latest)

    fields = [
        "epochs",
        "lr",
        "seed",
        "num_workers",
        "eval_every",
        "save_every",
        "input_size",
        "z_dim",
        "num_slots",
        "attn_iter",
        "gaussians_per_slot",
        "decoder_hidden_dim",
    ]
    values = {
        field: getattr(args, field)
        for field in fields
        if getattr(args, field) is not None
    }
    if args.data is not None:
        values["data_root"] = args.data
    if args.out is not None:
        values["out_dir"] = args.out
    cfg = replace(cfg, **values)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but torch.cuda.is_available() is False"
        )

    meta = dataset_meta(
        cfg.data_root,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
    )
    # Seed before building the model so cfg.seed also fixes its initialisation.
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, meta)
    train(model, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
