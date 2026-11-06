#!/usr/bin/env python
"""Preview one Primordia scene: save each view's image and mask as its own file.

Writes ``img_v<i>.png`` and ``mask_v<i>.png`` into ``--out`` (no stitching).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: the gsod package lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np

from PIL import Image

from gsod.data_generator import io
from gsod.data_generator.constants import PALETTE
from gsod.utils.png import save_png

_MASK_BG = np.array([35, 35, 40], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map an object-index mask to RGB: 0 -> dark background, i -> palette i-1."""
    mask = np.asarray(mask)
    out = np.broadcast_to(_MASK_BG, (*mask.shape, 3)).copy()
    for index in np.unique(mask):
        if index == 0:
            continue
        color = np.array(PALETTE[(int(index) - 1) % len(PALETTE)]) * 255.0
        out[mask == index] = color.round().astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset root (uORF-style folder)")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("preview"))
    args = parser.parse_args()

    frames = io.find_scene_frames(args.root, args.scene)
    if not frames:
        raise SystemExit(f"no frames for scene {args.scene} in {args.root}")
    scene_dir = args.root / io.scene_dirname(args.scene)

    args.out.mkdir(parents=True, exist_ok=True)
    for view, stem in enumerate(frames):
        image = io.read_view(scene_dir, stem)["image"]
        mask = np.asarray(Image.open(scene_dir / f"{stem}_mask.png").convert("L"))
        save_png(args.out / f"img_v{view}.png", image)
        save_png(args.out / f"mask_v{view}.png", colorize_mask(mask))
    print(f"scene {args.scene}: {len(frames)} views -> {args.out}/img_v*.png, mask_v*.png")


if __name__ == "__main__":
    main()
