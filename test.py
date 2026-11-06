#!/usr/bin/env python
"""Evaluate and render a checkpoint (model from the run's config.json).

    python test.py --dataset clevr567 --split test --suffix best

Writes metrics.json, scenes.csv, slots.csv, sc####/*.png to
<checkpoint>/renders/<split>_<suffix>.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import torch
import wandb

from gsod.configs import DATASETS, config_for
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model
from gsod.model.evaluate import (
    DIAGNOSTIC_KEYS,
    RECONSTRUCTION_KEYS,
    SEGMENTATION_KEYS,
    SLOT_COLUMNS,
    collect,
    evaluate,
    slot_table,
)
from gsod.utils.io import latest_run_dir, load_networks
from gsod.utils.visualize import panels, save_panels, scene_images, wandb_images, wandb_slot_table

WANDB_PROJECT = "gsod"
IMAGE_SCENES = 16
WORST_SCENES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS), default="primordia"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory")
    parser.add_argument("--suffix", default="latest", help="checkpoint suffix")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None, help="output directory for PNGs")
    parser.add_argument("--no-render", action="store_true", help="skip the PNG renders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_for(args.dataset)
    data_root = args.data or (
        cfg.test_root if args.split == "test" and cfg.test_root else cfg.data_root
    )
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

    cfg = run_config(cfg, Path(checkpoint))
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
    result = evaluate(model, dataset, device, IMAGE_SCENES)
    groups = {
        "seg": SEGMENTATION_KEYS,
        "recon": RECONSTRUCTION_KEYS,
        "diag": DIAGNOSTIC_KEYS,
    }
    summary = {
        f"{group}/{key}": result.metrics[key] for group, keys in groups.items() for key in keys
    }
    m = result.metrics
    print(f"{args.split} split ({len(dataset)} scenes)")
    print(f"  2D   ARI {m['ari']:.4f}   FG-ARI {m['fg_ari']:.4f}")
    print(f"  3D   NV-ARI {m['nv_ari']:.4f}")
    print(f"  novel views   PSNR {m['psnr']:.2f}   SSIM {m['ssim']:.4f}   LPIPS {m['lpips']:.4f}")
    print("  diag  " + "   ".join(f"{k} {m[k]:.3f}" for k in DIAGNOSTIC_KEYS))

    # Worst NV-ARI first (scenes without GT last).
    scenes = sorted(
        result.per_scene, key=lambda r: (math.isnan(r["nv_ari"]), r["nv_ari"])
    )
    worst = {}
    for row in scenes[:WORST_SCENES]:
        rec = collect(model, dataset[row["index"]], device)
        worst.update(scene_images(rec, slot_table(rec), prefix="worst_"))

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{Path(checkpoint).name}_{args.split}_{args.suffix}",
        job_type="test",
        dir=str(checkpoint),
        config={"checkpoint": str(checkpoint), "suffix": args.suffix, "split": args.split},
    )
    run.summary.update(summary)
    run.log(
        {
            "slots/table": wandb_slot_table(result.slots),
            **wandb_images(result.images),
            **wandb_images(worst),
        }
    )
    run.finish()

    out = Path(args.out or f"{checkpoint}/renders/{args.split}_{args.suffix}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    _write_csv(out / "scenes.csv", scenes)
    _write_csv(out / "slots.csv", result.slots, columns=list(SLOT_COLUMNS))
    print(f"wrote {out}/metrics.json, scenes.csv (worst first) and slots.csv")
    if not args.no_render:
        for i in range(len(dataset)):
            rec = collect(model, dataset[i], device)
            save_panels(panels(rec, slot_table(rec)), out / f"sc{rec['scene']:04d}")
        print(f"wrote the panels of {len(dataset)} scenes to {out}")


MODEL_FIELDS = (
    "input_size",
    "z_dim",
    "num_slots",
    "attn_iter",
    "gaussians_per_slot",
    "decoder_hidden_dim",
    "near",
)


def run_config(cfg, checkpoint: Path):
    """cfg with the model fields of the run's config.json, if any."""
    path = checkpoint / "config.json"
    if not path.is_file():
        print(f"no {path}: building the model from the {cfg.dataset_mode} profile")
        return cfg
    saved = json.loads(path.read_text())
    known = {f.name for f in fields(cfg)}
    return replace(cfg, **{k: saved[k] for k in MODEL_FIELDS if k in saved and k in known})


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
