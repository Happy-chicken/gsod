#!/usr/bin/env python
"""Train the GS slot model.

Examples:
    python train.py --dataset primordia
    python train.py --dataset clevr567          # data/clevr/clevr_567_train

The training loop lives here because train.py is the only training entry
point. Model code stays training-loop-agnostic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from itertools import count
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gsod.configs import BaseConfig, config_for
from gsod.utils.paths import latest_run_dir
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model
from gsod.model.checkpoint import load_networks, save_networks
from gsod.model.evaluate import evaluate
from gsod.model.gsod import GSODModel
from gsod.model.losses import GSODLoss


def train(
    model: GSODModel,
    device: torch.device,
    cfg: BaseConfig,
    *,
    resume: str | None = None,
) -> None:
    """Train one scene per step and render the views the encoder did not see."""
    out = Path(cfg.out_dir)
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

    rng = random.Random(cfg.seed)
    loader = DataLoader(
        train_set,
        batch_size=None,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    model = model.to(device)
    criterion = GSODLoss(cfg.loss).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    start_step, best_psnr = 0, float("-inf")
    if resume is not None:
        load_networks(str(out), resume, model.networks(), [opt], device=device)
        state = json.loads((out / f"{resume}_state.json").read_text())
        start_step, best_psnr = state["step"], state["best_psnr"]
        print(f"resumed from {out} suffix={resume} at step {start_step}")

    print(f"device={device} steps={cfg.steps} out={out}")
    batches = _forever(loader)
    started = time.time()
    running: dict[str, float] = {}
    last_step = start_step
    for step in range(start_step + 1, cfg.steps + 1):
        last_step = step
        item = next(batches)
        images, cams = item["images"], item["cam2world"]
        ref = rng.randrange(images.shape[0])
        order = [ref] + [i for i in range(images.shape[0]) if i != ref]
        images, cams = images[order].to(device), cams[order].to(device)

        model.opacity_floor = cfg.opacity_floor * max(
            0.0, 1.0 - step / cfg.opacity_floor_steps
        )
        out_dict = model(images, cams)
        loss, terms = criterion(out_dict, images, step)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for name, value in terms.items():
            running[name] = running.get(name, 0.0) + value

        if step % cfg.log_every == 0:
            parts = " ".join(f"{k} {v / cfg.log_every:.4f}" for k, v in running.items())
            print(f"step {step} {parts} elapsed {time.time() - started:.1f}s")
            running = {}

        if len(val_set) and step % cfg.eval_every == 0:
            metrics = evaluate(model, val_set, device)
            print(
                f"step {step} val mse {metrics['mse']:.5f} psnr {metrics['psnr']:.2f} "
                f"fg_ari {metrics['fg_ari']:.3f} attn_fg_ari {metrics['attn_fg_ari']:.3f}"
            )
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                _save(out, "best", model, opt, step, best_psnr)
                print(f"step {step} new best psnr {best_psnr:.2f}")

        if step % cfg.save_every == 0:
            _save(out, f"iter_{step}", model, opt, step, best_psnr)

    _save(out, "latest", model, opt, last_step, best_psnr)
    print(f"done, wrote {out}/latest_*.pth")


def _forever(loader: DataLoader):
    for _ in count():
        yield from loader


def _save(
    out: Path,
    suffix: str,
    model: GSODModel,
    opt: torch.optim.Optimizer,
    step: int,
    best_psnr: float,
) -> None:
    save_networks(str(out), suffix, model.networks(), [opt])
    (out / f"{suffix}_state.json").write_text(
        json.dumps({"step": step, "best_psnr": best_psnr})
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=["primordia", "clevr567"], default="primordia"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--attn-iter", type=int, default=None)
    parser.add_argument("--gaussians-per-slot", type=int, default=None)
    parser.add_argument("--bg-gaussians-per-slot", type=int, default=None)
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
        "steps",
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
        "bg_gaussians_per_slot",
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
    device = torch.device("cuda")

    meta = dataset_meta(
        cfg.data_root,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
    )
    # Seed before building the model so cfg.seed also fixes its initialisation.
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, meta)
    train(model, device, cfg, resume=args.resume)


if __name__ == "__main__":
    main()
