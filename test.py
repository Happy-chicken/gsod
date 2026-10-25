#!/usr/bin/env python
"""Evaluate and render a trained checkpoint.

Examples:
    python test.py --dataset primordia --split test
    python test.py --dataset clevr567 \
        --data clevr_567_test --checkpoint checkpoints/clevr567_<timestamp> --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gsod.configs import config_for
from gsod.utils.paths import latest_run_dir
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model, load_networks
from gsod.model.evaluate import evaluate, render_scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=["primordia", "clevr567"], default="primordia"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory")
    parser.add_argument("--suffix", default="latest", help="checkpoint suffix")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None, help="output directory for PNGs")
    parser.add_argument("--no-render", action="store_true", help="only print metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_for(args.dataset)
    data_root = args.data or cfg.data_root
    checkpoint = args.checkpoint or latest_run_dir(cfg.dataset_mode)
    if checkpoint is None:
        raise FileNotFoundError(
            "no timestamped checkpoint found; pass --checkpoint explicitly"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA testing requested but torch.cuda.is_available() is False"
        )
    device = torch.device("cuda")

    meta = dataset_meta(
        data_root,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
    )
    model = build_model(cfg, meta).to(device)
    load_networks(checkpoint, args.suffix, model.networks(), device=device)

    dataset = make_dataset(
        data_root,
        args.split,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
        n_views=cfg.n_views,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        load_masks=True,  # CLEVR-567 *_mask.png; ignored by Primordia
    )
    if cfg.dataset_mode == "primordia":
        dataset = WithGTMasks(dataset, data_root)
    metrics = evaluate(model, dataset, device)
    print(f"{args.split} split:", {k: round(v, 4) for k, v in metrics.items()})
    # print(f"{args.split} split: mse {metrics['mse']:.5f} psnr {metrics['psnr']:.2f}")

    if not args.no_render:
        out = Path(args.out or f"{checkpoint}/renders/{args.split}_{args.suffix}")
        render_scenes(model, dataset, device, out)
        print(f"wrote renders for {len(dataset)} scenes to {out}")


if __name__ == "__main__":
    main()
