#!/usr/bin/env python
"""Evaluate and render a trained checkpoint.

W&B (job_type "test"): test/* and diag/* (summary), slots/*, images, and
worst_* (the WORST_SCENES lowest NV-ARI scenes). Segmentation as uORF:
2D ARI / FG-ARI on the input view, 3D NV-ARI on the novel views.
Output directory (default <checkpoint>/renders/<split>_<suffix>):
scenes.csv (worst first), slots.csv, sc####/*.png (unless --no-render).

Examples:
    python test.py --dataset primordia --split test
    python test.py --dataset clevr567 --split test   # data/clevr/clevr_567_test
    python test.py --dataset primordia --suffix epoch_50
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import wandb

from gsod.configs import DATASETS, config_for
from gsod.data_loader import dataset_meta, make_dataset, WithGTMasks
from gsod.model import build_model
from gsod.model.evaluate import evaluate
from gsod.utils.io import latest_run_dir, load_networks
from gsod.utils.visualize import (
    SLOT_COLUMNS,
    collect,
    panels,
    report,
    save_panels,
    scene_images,
    slot_table,
    wandb_histograms,
    wandb_images,
    wandb_slot_table,
)

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
    per_scene: list[dict] = []
    with torch.no_grad():
        metrics = evaluate(model, dataset, device, per_scene=per_scene)
        rep = report(model, dataset, device, IMAGE_SCENES)
    print(f"{args.split} split:", {k: round(v, 4) for k, v in metrics.items()})
    print(
        f"  2D  ARI {metrics['ari']:.4f}  FG-ARI {metrics['fg_ari']:.4f}"
        f"   3D  NV-ARI {metrics['nv_ari']:.4f}"
    )

    # One row per scene, worst NV-ARI first (scenes without GT last).
    stats = {int(s["scene"]): s for s in rep.per_scene}
    scenes = []
    for row in per_scene:
        extra = stats.get(row["scene"], {})
        scenes.append({**{k: v for k, v in extra.items() if k != "scene"}, **row})
    key = "nv_ari"
    scenes.sort(key=lambda r: (math.isnan(r.get(key, math.nan)), r.get(key, math.nan)))
    worst = {}
    model.eval()
    with torch.no_grad():
        for row in scenes[:WORST_SCENES]:
            i = row["index"]
            rec = collect(model, dataset[i], device)
            worst.update(scene_images(rec, slot_table(rec), prefix="worst_"))

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{Path(checkpoint).name}_{args.split}_{args.suffix}",
        job_type="test",
        dir=str(checkpoint),
        config={"checkpoint": str(checkpoint), "suffix": args.suffix, "split": args.split},
    )
    run.summary.update({f"test/{k}": v for k, v in metrics.items()})
    run.summary.update({f"diag/{k}": v for k, v in rep.stats.items()})
    run.log(
        {
            "slots/table": wandb_slot_table(rep.rows),
            **wandb_histograms(rep.rows),
            **wandb_images(rep.images),
            **wandb_images(worst),
        }
    )
    run.finish()

    out = Path(args.out or f"{checkpoint}/renders/{args.split}_{args.suffix}")
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "scenes.csv", scenes)
    _write_csv(out / "slots.csv", rep.rows, columns=list(SLOT_COLUMNS))
    print(f"wrote {out}/scenes.csv (worst first) and {out}/slots.csv")
    if not args.no_render:
        with torch.no_grad():
            for i in range(len(dataset)):
                rec = collect(model, dataset[i], device)
                save_panels(panels(rec, slot_table(rec)), out / f"sc{rec['scene']:04d}")
        print(f"wrote the panels of {len(dataset)} scenes to {out}")


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or (list(rows[0]) if rows else [])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
