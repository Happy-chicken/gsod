#!/usr/bin/env python
"""Check a CLEVR-567 root (flat uORF layout) against the Clevr567Config.

    python scripts/inspect_clevr567.py data/clevr/clevr_567_train
    python scripts/inspect_clevr567.py data/clevr/clevr_567_test --masks 5

Reports the image size, the camera convention (OpenCV: the cam2world third
column points at the scene centre), the camera distance range next to the
config's scene_prior["distance_range"], and, for a few *_mask.png files, the
grey values and which one the loader treats as background (the most
frequent one).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: the gsod package lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
from PIL import Image

from gsod.configs import Clevr567Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--masks", type=int, default=3, help="mask files to show")
    args = parser.parse_args()

    poses = sorted(args.root.glob("*_RT.txt"))
    if not poses:
        raise SystemExit(f"no *_RT.txt files in {args.root}")
    images = [p.with_name(p.name[: -len("_RT.txt")] + ".png") for p in poses]
    sizes = {Image.open(p).size for p in images[:20] if p.is_file()}
    print(f"{len(poses)} views; image sizes (W, H) of the first 20: {sorted(sizes)}")

    cams = np.stack([np.loadtxt(p) for p in poses])  # (N, 4, 4) cam2world
    centre = cams[:, :3, 3]
    dist = np.linalg.norm(centre, axis=1)
    elev = np.degrees(np.arcsin(centre[:, 2] / dist))
    # OpenCV: +z (third column) is the viewing direction, towards the origin.
    facing = np.einsum("ni,ni->n", cams[:, :3, 2], -centre / dist[:, None])
    print(f"camera distance  min {dist.min():.3f}  max {dist.max():.3f}")
    print(f"camera elevation min {elev.min():.1f}  max {elev.max():.1f} deg")
    print(
        f"cos(view dir, dir to origin)  min {facing.min():.3f}  "
        "(~1: OpenCV, as the loader assumes; ~-1: OpenGL, flip y/z)"
    )
    prior = Clevr567Config().scene_prior["distance_range"]
    if dist.min() < prior[0] or dist.max() > prior[1]:
        print(f"WARNING: distances outside scene_prior distance_range {prior}")

    for path in sorted(args.root.glob("*_mask.png"))[: args.masks]:
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.int64)
        values, counts = np.unique(gray, return_counts=True)
        bg = values[counts.argmax()]
        share = ", ".join(f"{v}:{c / gray.size:.2f}" for v, c in zip(values, counts))
        print(f"{path.name}: grey value:share {share}  -> background {bg}")


if __name__ == "__main__":
    main()
