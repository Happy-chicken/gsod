#!/usr/bin/env python
"""Train the GS slot model (PyTorch Lightning, logged to Weights & Biases).

Examples:
    python train.py --dataset primordia
    python train.py --dataset clevr567          # data/clevr/clevr_567_train
    WANDB_MODE=offline python train.py ...      # log locally, sync later

Training runs for cfg.epochs passes over the training scenes (one scene
per step); the learning-rate schedule, validation and saving are in epochs.

Checkpoints stay in the uORF layout (gsod/model/checkpoint.py):
``{suffix}_net_*.pth`` / ``{suffix}_optimizer_0.pth`` / ``{suffix}_state.json``
for latest, best and epoch_<n>, so test.py and the diagnostics load them
unchanged; ``--resume`` continues from one of them. Lightning writes no
checkpoints of its own. The W&B run lives in the checkpoint directory.

The training loop lives here because train.py is the only training entry
point. Model code stays training-loop-agnostic.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import DataLoader

from gsod.configs import BaseConfig, config_for
from gsod.utils.paths import latest_run_dir
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model
from gsod.model.checkpoint import load_networks, save_networks
from gsod.model.evaluate import evaluate
from gsod.model.gsod import GSODModel
from gsod.model.losses import GSODLoss

WANDB_PROJECT = "gsod"


class GSODTraining(L.LightningModule):
    """One scene per step: view 0 is a random view of the scene, the others
    are the novel views the encoder does not see."""

    def __init__(
        self,
        model: GSODModel,
        cfg: BaseConfig,
        val_set,
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

    def training_step(self, item: dict, batch_idx: int) -> torch.Tensor:
        images, cams = item["images"], item["cam2world"]
        ref = self.rng.randrange(images.shape[0])
        order = [ref] + [i for i in range(images.shape[0]) if i != ref]
        images, cams = images[order], cams[order]
        loss, terms = self.criterion(self.model(images, cams), images)
        self.log_dict(
            {f"train/{name}": value for name, value in terms.items() if name != "l1"},
            batch_size=1,
        )
        self.log("train/l1", terms["l1"], prog_bar=True, batch_size=1)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Before clipping: a norm far above grad_clip means the clipping
        # scales every other gradient down to almost nothing.
        norms = grad_norm(self.model, norm_type=2)
        self.log("train/grad_norm", norms["grad_2.0_norm_total"], batch_size=1)

    def on_train_epoch_end(self) -> None:
        epoch = self.epoch + 1  # this epoch is done
        if len(self.val_set) and epoch % self.cfg.eval_every == 0:
            with torch.no_grad():
                metrics = evaluate(self.model, self.val_set, self.device)
            self.log_dict({f"val/{k}": v for k, v in metrics.items()}, batch_size=1)
            print(
                f"epoch {epoch} val mse {metrics['mse']:.5f} psnr {metrics['psnr']:.2f} "
                f"fg_ari {metrics['fg_ari']:.3f} attn_fg_ari {metrics['attn_fg_ari']:.3f}"
            )
            if metrics["psnr"] > self.best_psnr:
                self.best_psnr = metrics["psnr"]
                self._save("best", epoch)
                print(f"epoch {epoch} new best psnr {self.best_psnr:.2f}")
        if epoch % self.cfg.save_every == 0:
            self._save(f"epoch_{epoch}", epoch)

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
    """Linear warm-up to cfg.lr, then halved every cfg.lr_half_life epochs."""
    warmup = min(1.0, epoch / max(cfg.lr_warmup_epochs, 1e-8))
    return warmup * 0.5 ** (epoch / cfg.lr_half_life)


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
    print(
        f"data={cfg.data_root} train_scenes={len(train_set)} val_scenes={len(val_set)}"
    )
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
        len(train_set),
        resume=resume,
        start_epoch=start_epoch,
        best_psnr=best_psnr,
    )

    logger = WandbLogger(
        project=WANDB_PROJECT, name=out.name, save_dir=str(out), config=asdict(cfg)
    )
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
        "--dataset", choices=["primordia", "clevr567"], default="primordia"
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
