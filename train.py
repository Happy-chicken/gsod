#!/usr/bin/env python
"""Train the GS slot model (Lightning, W&B). batch_scenes scenes per step.

    python train.py --dataset clevr567
    python train.py --dataset room_diverse --train-scenes 500 --val-scenes 50 \
        --monitor-scenes 20 --epochs 20 --eval-every 2   # quick run

val/psnr picks "best"; monitor (test scenes with GT) gives seg/*, diag/*.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, fields, replace
from pathlib import Path

import lightning as L
import torch
import torch._dynamo.config
import torch._inductor.config
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from gsod.configs import DATASETS, BaseConfig, config_for
from gsod.data_loader import make_dataset
from gsod.eval.evaluate import (
    DIAGNOSTIC_KEYS,
    RECONSTRUCTION_KEYS,
    SEGMENTATION_KEYS,
    evaluate,
)
from gsod.eval.visualize import WANDB_PROJECT, wandb_images, wandb_slot_table
from gsod.model import build_model
from gsod.model.gsod import GSODModel
from gsod.model.losses import GSODLoss
from gsod.utils.io import latest_run_dir, load_networks, save_networks
from gsod.utils.schedule import cosine

IMAGE_SCENES = 4


class GSODTraining(L.LightningModule):
    """View 0 is a random view of the scene; the others are novel views."""

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
        images, cams = item["images"], item["cam2world"]  # (B, V, ...)
        n_views = images.shape[1]
        orders = []
        for _ in range(images.shape[0]):  # view 0: a random view of each scene
            ref = self.rng.randrange(n_views)
            orders.append([ref] + [i for i in range(n_views) if i != ref])
        orders = torch.tensor(orders, device=images.device)
        images = torch.stack([x[o] for x, o in zip(images, orders)])
        cams = torch.stack([x[o] for x, o in zip(cams, orders)])
        # Noise annealed by scenes seen, as with one scene per step.
        step = self.start_epoch * self.steps_per_epoch + self.global_step
        self.model.slot_attention.anneal(step * self.cfg.batch_scenes)
        outputs = self.model(images, cams)
        loss, terms = self.criterion(outputs, images)
        for name, value in terms.items():
            self.log(f"train/{name}", value, prog_bar=name == "recon", batch_size=1)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Clip and log the norm before clipping (one GPU op, no sync).
        max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else float("inf")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.log("train/grad_norm", norm, batch_size=1)

    def on_train_epoch_end(self) -> None:
        epoch = self.epoch + 1  # this epoch is done
        if epoch % self.cfg.eval_every == 0:
            self._validate(epoch)
        if epoch % self.cfg.save_every == 0:
            self._save(f"epoch_{epoch}", epoch)

    def _validate(self, epoch: int) -> None:
        """val: psnr, ssim (best checkpoint); monitor: seg, diag, images."""
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
                "epoch": epoch,
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
        """epoch: epochs done (where --resume starts)."""
        save_networks(str(self.out), suffix, self.model.networks(), self.trainer.optimizers)
        (self.out / f"{suffix}_state.json").write_text(
            json.dumps({"epoch": epoch, "best_psnr": self.best_psnr})
        )


def _lr_factor(cfg: BaseConfig, epoch: float) -> float:
    """Linear warm-up, then cosine decay to cfg.lr_final."""
    warm = max(cfg.lr_warmup * cfg.epochs, 1e-8)
    if epoch < warm:
        return epoch / warm
    return cosine((epoch - warm) / max(cfg.epochs - warm, 1e-8), 1.0, cfg.lr_final)


def _first(dataset, n: int | None):
    """The first n scenes (all if n is None)."""
    if n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, range(n))


def _has_masks(dataset) -> bool:
    return bool(len(dataset)) and dataset[0].get("masks") is not None


def train(model: GSODModel, cfg: BaseConfig, *, resume: str | None = None) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_set = _first(make_dataset(cfg, cfg.data_root, "train"), cfg.train_scenes)
    val_set = _first(make_dataset(cfg, cfg.data_root, "val", load_masks=True), cfg.val_scenes)
    monitor_set, monitor_root = val_set, f"{cfg.data_root} (val)"
    if cfg.test_root:
        monitor_set = _first(make_dataset(cfg, cfg.test_root, "test", load_masks=True), cfg.monitor_scenes)
        monitor_root = f"{cfg.test_root} (test, first {len(monitor_set)})"
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"train: {cfg.data_root} ({len(train_set)} scenes)")
    print(f"val: {cfg.data_root} ({len(val_set)} scenes, GT masks: {_has_masks(val_set)})")
    print(f"monitor: {monitor_root} (GT masks: {_has_masks(monitor_set)})")
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_scenes,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    start_epoch, best_psnr = 0, float("-inf")
    if resume is not None:
        # Optimizer state is loaded in on_fit_start.
        load_networks(str(out), resume, model.networks(), device="cpu")
        state = json.loads((out / f"{resume}_state.json").read_text())
        start_epoch, best_psnr = state["epoch"], state["best_psnr"]
        print(f"resuming from {out} suffix={resume} after epoch {start_epoch}")
    module = GSODTraining(
        model,
        cfg,
        val_set,
        monitor_set,
        len(loader),
        resume=resume,
        start_epoch=start_epoch,
        best_psnr=best_psnr,
    )

    logger = WandbLogger(
        project=WANDB_PROJECT, name=out.name, save_dir=str(out), config=asdict(cfg)
    )
    for counter in ("trainer/global_step", "epoch"):
        logger.experiment.define_metric(counter, hidden=True)
    # Evaluation curves against epochs (comparable across batch_scenes).
    for group in ("val", "seg", "diag"):
        logger.experiment.define_metric(f"{group}/*", step_metric="epoch")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.epochs - start_epoch,
        logger=logger,
        log_every_n_steps=cfg.log_every,
        enable_checkpointing=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=str(out),
    )
    print(f"epochs={cfg.epochs} steps/epoch={len(loader)} ({cfg.batch_scenes} scenes each) out={out}")
    trainer.fit(module, loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS), required=True
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-scenes", type=int, default=None, help="scenes per step")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None, help="epochs")
    parser.add_argument("--save-every", type=int, default=None, help="epochs")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--attn-iter", type=int, default=None)
    parser.add_argument("--gaussians-per-slot", type=int, default=None)
    parser.add_argument("--gaussians-per-query", type=int, default=None)
    parser.add_argument("--decoder-hidden-dim", type=int, default=None)
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

    known = {f.name for f in fields(cfg)}
    values = {k: v for k, v in vars(args).items() if v is not None and k in known}
    if args.data is not None:
        values["data_root"] = args.data
    if args.out is not None:
        values["out_dir"] = args.out
    cfg = replace(cfg, **values)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but torch.cuda.is_available() is False"
        )

    torch.manual_seed(cfg.seed)
    model = build_model(cfg)
    # Not the encoder (Inductor bug with replicate padding); random numbers as without compiling.
    torch._inductor.config.fallback_random = True
    # Room for every compiled variant (bg / object slots, train / eval, grad / no grad, batch sizes).
    torch._dynamo.config.cache_size_limit = 64
    for net in (model.slot_attention, model.obj_decoder, model.bg_decoder):
        net.compile()
    train(model, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
