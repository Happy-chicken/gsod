#!/usr/bin/env python
"""Probe: does an object slot hold its object's shape?

Fits a small MLP from each object slot of a frozen checkpoint to its GT
object's view-0 silhouette (cropped to the GT box, resized to SxS) and
reports held-out IoU next to the mean silhouette's (a blob that knows no
shape). Slot -> object: the GT object holding most of the slot's attention
(skipped below half). Needs GT masks; no rendering, so CPU works.

    python scripts/probe_slots.py --dataset room_diverse --checkpoint <run dir> \\
        --data data/room_diverse/room_diverse_test --fit-split train --eval-split val

Writes <checkpoint>/probe/{metrics.json, examples.png} (examples: target,
probe, mean per row).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs import DATASETS, config_for, run_config
from gsod.data_loader import WithGTMasks, dataset_meta, make_dataset
from gsod.model import build_model
from gsod.utils.io import latest_run_dir, load_networks, save_png


def silhouette(obj: torch.Tensor, size: int) -> torch.Tensor:
    """Binary mask (H, W) -> its box crop resized to (size, size) in [0, 1]."""
    rows, cols = obj.any(1).nonzero()[:, 0], obj.any(0).nonzero()[:, 0]
    crop = obj[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].float()
    return F.interpolate(crop[None, None], size=(size, size), mode="area")[0, 0]


@torch.no_grad()
def collect(model, dataset, device, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Object slots (M, C) and their objects' silhouettes (M, size*size)."""
    side = model.input_size
    slots, targets = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        if item.get("masks") is None:
            raise ValueError("the probe needs GT masks")
        gt = item["masks"][0].to(device)
        scene = model(
            item["images"].to(device),
            item["cam2world"].to(device),
            render=False,
            masks=gt if model.oracle_masks else None,
        )
        tokens = F.interpolate(gt[None, None].float(), size=side, mode="nearest")[0, 0].long().flatten()
        labels = [l for l in tokens.unique() if l > 0]
        for k in range(1, scene["attn"].shape[0]):
            a = scene["attn"][k]
            if not labels:
                break
            mass = torch.stack([a[tokens == l].sum() for l in labels])
            j = int(mass.argmax())
            if mass[j] < 0.5 * a.sum():
                continue
            slots.append(scene["slots"][k - 1].float().cpu())
            targets.append(silhouette(gt == labels[j], size).flatten().cpu())
    return torch.stack(slots), torch.stack(targets)


def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean IoU of (M, P) masks thresholded at 0.5."""
    p, t = pred > 0.5, target > 0.5
    inter, union = (p & t).sum(1).float(), (p | t).sum(1).float()
    return float((inter / union.clamp_min(1)).mean())


def fit(x: torch.Tensor, y: torch.Tensor, epochs: int, seed: int):
    """Full-batch MLP fit with BCE -> predict(slots) in [0, 1]."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    mlp = nn.Sequential(
        nn.Linear(x.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, y.shape[1])
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(mlp((x - mean) / std), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda z: torch.sigmoid(mlp((z - mean) / std))


def examples(target, probe, mean, size: int, n: int = 8) -> np.ndarray:
    """Rows of (target | probe | mean) silhouettes, upscaled, as uint8."""
    rows = []
    for i in range(min(n, len(target))):
        tiles = [t[i].reshape(size, size) for t in (target, probe, mean)]
        rows.append(torch.cat([F.pad(t, (1, 1, 1, 1), value=0.5) for t in tiles], 1))
    grid = torch.cat(rows, 0).clamp(0, 1)
    grid = grid.repeat_interleave(8, 0).repeat_interleave(8, 1)
    return (grid.numpy() * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="room_diverse")
    parser.add_argument("--data", default=None, help="dataset root with GT masks")
    parser.add_argument("--checkpoint", default=None, help="run directory")
    parser.add_argument("--suffix", default="best")
    parser.add_argument("--fit-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--eval-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--size", type=int, default=16, help="silhouette side")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    cfg = config_for(args.dataset)
    checkpoint = args.checkpoint or latest_run_dir(cfg.dataset_mode)
    cfg = run_config(cfg, Path(checkpoint))
    root = args.data or cfg.test_root or cfg.data_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = dataset_meta(
        root, dataset_mode=cfg.dataset_mode, image_size=cfg.dataset_image_size, focal_ratio=cfg.dataset_focal_ratio
    )
    model = build_model(cfg, meta).to(device).eval()
    load_networks(checkpoint, args.suffix, model.networks(), device=device)

    def split(name):
        data = make_dataset(
            root,
            name,
            dataset_mode=cfg.dataset_mode,
            image_size=cfg.dataset_image_size,
            focal_ratio=cfg.dataset_focal_ratio,
            n_views=cfg.n_views,
            val_fraction=cfg.val_fraction,
            seed=cfg.seed,
            load_masks=True,
        )
        return WithGTMasks(data, root) if cfg.dataset_mode == "primordia" else data

    x_fit, y_fit = collect(model, split(args.fit_split), device, args.size)
    x_eval, y_eval = collect(model, split(args.eval_split), device, args.size)
    probe = fit(x_fit, y_fit, args.epochs, cfg.seed)
    mean = y_fit.mean(0, keepdim=True)
    with torch.no_grad():
        p_fit, p_eval = probe(x_fit), probe(x_eval)
    metrics = {
        "objects_fit": len(x_fit),
        "objects_eval": len(x_eval),
        "iou_mean_silhouette": iou(mean.expand_as(y_eval), y_eval),
        "iou_probe_eval": iou(p_eval, y_eval),
        "iou_probe_fit": iou(p_fit, y_fit),
    }
    out = Path(checkpoint) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_png(out / "examples.png", examples(y_eval, p_eval, mean.expand_as(y_eval), args.size))
    for k, v in metrics.items():
        print(f"{k:22s} {v:.4f}" if isinstance(v, float) else f"{k:22s} {v}")
    print(f"wrote {out}/metrics.json and examples.png (rows: target | probe | mean)")


if __name__ == "__main__":
    main()
