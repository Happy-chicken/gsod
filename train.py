#!/usr/bin/env python
"""Train the GS slot model (PyTorch Lightning, Weights & Biases).

Examples:
    python train.py --dataset primordia
    python train.py --dataset clevr567          # data/clevr/clevr_567_train
    WANDB_MODE=offline python train.py ...      # log locally, sync later

    # Quick run (~10k steps): 500 scenes, schedules in epochs scale along.
    python train.py --dataset room_diverse --train-scenes 500 --val-scenes 50 \
        --monitor-scenes 20 --epochs 20 --eval-every 2 --save-every 10
    # Decoder capacity check: overfit 8 scenes (view-0 render should be sharp).
    python train.py --dataset room_diverse --train-scenes 8 --val-scenes 8 \
        --monitor-scenes 8 --epochs 1000 --eval-every 100 --save-every 500

One scene per step; schedules, validation and saving are in epochs; the
learning rate warms up linearly, then decays (cosine). Checkpoints use the
uORF layout for latest, best and epoch_<n>;
--resume continues from one.

    val      held out of the training folder: val/psnr picks the best
             checkpoint
    monitor  the first cfg.monitor_scenes test scenes (Primordia: val),
             with GT masks: watched only

The run's config is written to <out>/config.json; test.py builds the model
from it.

W&B (metrics: gsod/model/evaluate.py):
    train/*  losses, gradient norm, schedules
    val/*    psnr, ssim (val)
    seg/*    ari, fg_ari (2D, input view), nv_ari (3D, novel views) (monitor)
    diag/*   where a low score comes from (monitor)
    slots/table, images (monitor)
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
from gsod.model.evaluate import (
    DIAGNOSTIC_KEYS,
    RECONSTRUCTION_KEYS,
    SEGMENTATION_KEYS,
    evaluate,
)
from gsod.model.gsod import GSODModel
from gsod.model.losses import GSODLoss
from gsod.utils.io import latest_run_dir, load_networks, save_networks
from gsod.utils.visualize import wandb_images, wandb_slot_table

WANDB_PROJECT = "gsod"
# Monitor scenes drawn as images at every validation.
IMAGE_SCENES = 4


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
        """val: psnr, ssim (keeps the best checkpoint); monitor (GT masks):
        segmentation, diagnostics, slot table and images."""
        if not len(self.val_set):
            return
        monitor = evaluate(self.model, self.monitor_set, self.device, IMAGE_SCENES)
        val = monitor if self.monitor_set is self.val_set else evaluate(
            self.model, self.val_set, self.device
        )
        self.logger.log_metrics(
            {
                **{f"val/{key}": val.metrics[key] for key in RECONSTRUCTION_KEYS},
                **{f"seg/{key}": monitor.metrics[key] for key in SEGMENTATION_KEYS},
                **{f"diag/{key}": monitor.metrics[key] for key in DIAGNOSTIC_KEYS},
                "slots/table": wandb_slot_table(monitor.slots),
                **wandb_images(monitor.images),
            },
            step=self.global_step,
        )
        psnr = val.metrics["psnr"]
        if psnr > self.best_psnr:
            self.best_psnr = psnr
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


def _first(dataset, n: int | None):
    """The first n scenes of dataset (all when n is None or larger)."""
    if n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, range(n))


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
    train_set = _first(make_dataset(cfg.data_root, "train", **dataset_kwargs), cfg.train_scenes)
    # CLEVR-567 reads *_mask.png when present (and skips them otherwise);
    # Primordia masks live in the scene folders and need the wrapper.
    val_set = make_dataset(cfg.data_root, "val", load_masks=True, **dataset_kwargs)
    if cfg.dataset_mode == "primordia":
        val_set = WithGTMasks(val_set, cfg.data_root)
    val_set = _first(val_set, cfg.val_scenes)
    monitor_set, monitor_root = val_set, f"{cfg.data_root} (val)"
    if cfg.test_root:
        test_set = make_dataset(cfg.test_root, "test", load_masks=True, **dataset_kwargs)
        monitor_set = _first(test_set, cfg.monitor_scenes)
        monitor_root = f"{cfg.test_root} (test, first {len(monitor_set)})"
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
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
    parser.add_argument(
        "--pixel-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="decoders read view 0's tokens through their slot's attention",
    )
    parser.add_argument("--train-scenes", type=int, default=None, help="first N training scenes")
    parser.add_argument("--val-scenes", type=int, default=None, help="first N val scenes")
    parser.add_argument("--monitor-scenes", type=int, default=None, help="first N test scenes")
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
        "pixel_features",
        "train_scenes",
        "val_scenes",
        "monitor_scenes",
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
