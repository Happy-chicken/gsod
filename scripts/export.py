#!/usr/bin/env python
"""Export scenes' Gaussians as 3DGS .ply (all, objects, background, per
slot); object opacity is scaled by the draw factor.

    python scripts/export.py --dataset room_chair --checkpoint <run dir> --scenes 0 1 2
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gsod.configs import DATASETS, config_for
from gsod.data_loader import dataset_meta, make_dataset
from gsod.model import build_model
from gsod.utils.geometry import from_world, rotate_gaussians
from gsod.utils.io import latest_run_dir, load_networks, write_ply

# z up -> y up: (x, y, z) -> (x, z, -y), a -90 deg turn about x.
_Y_UP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_Y_UP_QUAT = torch.tensor([math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="primordia")
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--checkpoint", default=None, help="checkpoint directory")
    parser.add_argument("--suffix", default="latest")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scenes", type=int, nargs="+", default=[0], help="indices in the split")
    parser.add_argument("--out", default=None)
    parser.add_argument("--frame", choices=["world", "scene"], default="world")
    parser.add_argument("--y-up", action="store_true")
    args = parser.parse_args()

    cfg = config_for(args.dataset)
    root = args.data or (cfg.test_root if args.split == "test" and cfg.test_root else cfg.data_root)
    checkpoint = args.checkpoint or latest_run_dir(cfg.dataset_mode)
    if checkpoint is None:
        raise FileNotFoundError("no checkpoint found; pass --checkpoint")
    kwargs = dict(dataset_mode=cfg.dataset_mode, image_size=cfg.dataset_image_size,
                  focal_ratio=cfg.dataset_focal_ratio)
    model = build_model(cfg, dataset_meta(root, **kwargs))
    load_networks(str(checkpoint), args.suffix, model.networks(), device="cpu")
    model.eval()
    data = make_dataset(root, args.split, n_views=cfg.n_views, val_fraction=cfg.val_fraction,
                        seed=cfg.seed, **kwargs)
    out = Path(args.out or Path(checkpoint) / "ply" / f"{args.split}_{args.suffix}")

    for index in args.scenes:
        item = data[index]
        with torch.no_grad():
            scene = model(item["images"], item["cam2world"], render=False)
        obj, draw, frame = scene["objects"]["gaussians"], scene["draw"], scene["frame"]
        c = frame.centre.tolist()
        slots = [{**{k: v[k_] for k, v in obj.items()}, "opacity": obj["opacity"][k_] * draw[k_]}
                 for k_ in range(draw.shape[0])]
        parts = {f"slot{k + 1}": g for k, g in enumerate(slots)}
        parts["objects"] = {k: torch.cat([s[k] for s in slots]) for k in slots[0]}
        parts["background"] = scene["background"]["gaussians"]
        parts[""] = {k: torch.cat([parts["objects"][k], parts["background"][k]]) for k in slots[0]}
        name = f"scene{int(item.get('scene', index)):04d}"
        for part, g in parts.items():
            if args.frame == "scene":
                g = from_world(g, frame)
            if args.y_up:
                g = rotate_gaussians(g, _Y_UP, _Y_UP_QUAT)
            path = write_ply(out / (f"{name}_{part}.ply" if part else f"{name}.ply"), g)
            if not part:
                fill = ", ".join(f"{float(c):.3f}" for c in scene["background"]["fill"])
                print(f"{path}: {g['mu'].shape[0]} Gaussians"
                      + f"; fill colour ({fill}) where the floor does not cover the view; "
                      + f"scene frame c ({c[0]:.2f}, {c[1]:.2f}), R {float(frame.scale):.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
