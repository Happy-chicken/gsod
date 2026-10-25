#!/usr/bin/env python
"""Generate the Primordia synthetic dataset.

Glues the three responsibilities together: sample (generator) -> render
(analytic ray-cast) -> serialize (io). No geometry or camera maths lives here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gsod.data_generator import io
from gsod.data_generator.generator import SceneGenConfig, config_dict, generate
from gsod.data_generator.scene import render_views


def render_scene(sample: dict, cfg: SceneGenConfig) -> dict:
    """Analytically render every view of one sampled scene."""
    rendered = render_views(sample["scene"], sample["cam2world"], cfg.image_size, cfg.focal_ratio)
    return {
        "images": rendered["images"],
        "cam2world": sample["cam2world"].astype(np.float32),
        "mask": rendered["mask"],
        "depth": rendered["depth"],
        "num_objects": sample["scene"].num_objects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/primordia"))
    parser.add_argument("--n-scenes", type=int, default=200)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128, help="square image side")
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shapes", nargs="+", default=["sphere", "cuboid"])
    parser.add_argument("--placement-radius", type=float, default=1.8,
                        help="max object-centre radius in the xy plane")
    parser.add_argument("--radius-min", type=float, default=0.35)
    parser.add_argument("--radius-max", type=float, default=0.55)
    parser.add_argument("--min-gap", type=float, default=0.15,
                        help="minimum gap between object surfaces")
    parser.add_argument("--distance-min", type=float, default=4.5,
                        help="minimum camera-orbit radius")
    parser.add_argument("--distance-max", type=float, default=6.5,
                        help="maximum camera-orbit radius")
    parser.add_argument("--elevation-min", type=float, default=12.0)
    parser.add_argument("--elevation-max", type=float, default=28.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SceneGenConfig(
        n_scenes=args.n_scenes,
        k_min=args.k_min,
        k_max=args.k_max,
        image_size=(args.image_size, args.image_size),
        n_views=args.views,
        seed=args.seed,
        shapes=tuple(args.shapes),
        placement_radius=args.placement_radius,
        radius_range=(args.radius_min, args.radius_max),
        min_gap=args.min_gap,
        distance_range=(args.distance_min, args.distance_max),
        elevation_range=(args.elevation_min, args.elevation_max),
    )
    meta = {
        "design": "primordia",
        "image_size": list(cfg.image_size),
        "focal_ratio": list(cfg.focal_ratio),
        "n_views": cfg.n_views,
        "split_ratios": list(cfg.split),
        "seed": cfg.seed,
        "config": config_dict(cfg),
    }

    started = time.time()
    records = (render_scene(sample, cfg) for sample in generate(cfg))
    root = io.write_dataset(args.out, records, meta)
    print(
        f"wrote {cfg.n_scenes} scenes x {cfg.n_views} views "
        f"to {root} in {time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
