"""On-disk format for the Primordia dataset.

Responsibility: serialize already-rendered scenes to per-scene folders and read
them back. No geometry, no randomness, no rendering.

Layout (one folder per scene, one set of files per view)::

    <root>/
      meta.json                 # config, split, per-scene object counts
      sc0000/
        img0.png                # RGB image
        img0_mask.png           # per-pixel labels (0 = background, i = object i)
        img0_RT.txt             # cam2world, 4x4
        img0_azi_rot.txt        # azi_rot, 3x3
        img0_depth.npy          # depth, HxWx1 float32
        img1.png ...
      sc0001/
        ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from gsod.utils.png import save_png

__all__ = [
    "META_FILE",
    "scene_dirname",
    "frame_stem",
    "write_frame",
    "read_frame",
    "read_view",
    "write_dataset",
    "find_scene_frames",
    "write_meta",
    "load_meta",
    "make_split",
]

META_FILE = "meta.json"

# Matches an image file inside a scene folder, ignoring sidecars (``img0_mask.png``).
_IMAGE_RE = re.compile(r"^img(\d+)\.png$")


def scene_dirname(scene_index: int) -> str:
    """Folder name of a scene, e.g. ``sc0000``."""
    return f"sc{scene_index:04d}"


def frame_stem(view_index: int) -> str:
    """Filename stem shared by all files of one view, e.g. ``img2``."""
    return f"img{view_index}"


def write_frame(scene_dir, stem: str, image, mask, cam2world, azi_rot, depth) -> None:
    """Write the five files (image, mask, RT, azi_rot, depth) of one view."""
    scene_dir = Path(scene_dir)
    save_png(scene_dir / f"{stem}.png", np.asarray(image, dtype=np.uint8))
    save_png(scene_dir / f"{stem}_mask.png", np.asarray(mask, dtype=np.uint8))
    np.savetxt(
        scene_dir / f"{stem}_RT.txt",
        np.asarray(cam2world, dtype=np.float64),
        fmt="%.6f",
    )
    np.savetxt(
        scene_dir / f"{stem}_azi_rot.txt",
        np.asarray(azi_rot, dtype=np.float64),
        fmt="%.6f",
    )
    np.save(
        scene_dir / f"{stem}_depth.npy", np.asarray(depth, dtype=np.float32)[..., None]
    )


def read_frame(scene_dir, stem: str) -> dict:
    """Read one view's files into a dict of arrays (requires Pillow)."""
    from PIL import Image  # lazy: generation does not need Pillow

    scene_dir = Path(scene_dir)
    image = np.asarray(
        Image.open(scene_dir / f"{stem}.png").convert("RGB"), dtype=np.uint8
    )
    mask = np.asarray(
        Image.open(scene_dir / f"{stem}_mask.png").convert("L"), dtype=np.int16
    )
    return {
        "image": image,
        "mask": mask,
        "cam2world": np.loadtxt(scene_dir / f"{stem}_RT.txt", dtype=np.float64),
        "azi_rot": np.loadtxt(scene_dir / f"{stem}_azi_rot.txt", dtype=np.float64),
        "depth": np.load(scene_dir / f"{stem}_depth.npy"),
    }


def read_view(scene_dir, stem: str) -> dict:
    """Image and camera of one view only (requires Pillow)."""
    from PIL import Image  # lazy: generation does not need Pillow

    scene_dir = Path(scene_dir)
    image = np.asarray(
        Image.open(scene_dir / f"{stem}.png").convert("RGB"), dtype=np.uint8
    )
    return {
        "image": image,
        "cam2world": np.loadtxt(scene_dir / f"{stem}_RT.txt", dtype=np.float64),
    }


def find_scene_frames(root, scene_index: int) -> list[str]:
    """Sorted stems of every view of ``scene_index`` (image files only)."""
    scene_dir = Path(root) / scene_dirname(scene_index)
    stems = [
        path.name[:-4]  # drop ".png"
        for path in scene_dir.glob("img*.png")
        if _IMAGE_RE.match(path.name)
    ]
    return sorted(stems, key=lambda stem: int(_IMAGE_RE.match(stem + ".png").group(1)))  # type: ignore


def write_dataset(root, records, meta: dict) -> Path:
    """Write every scene record plus ``meta.json``; returns the dataset root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    n_scenes = 0
    n_frames = 0
    object_counts: list[int] = []
    for scene_index, record in enumerate(records):
        images = np.asarray(record["images"])
        scene_dir = root / scene_dirname(scene_index)
        scene_dir.mkdir(parents=True, exist_ok=True)
        for view_index in range(images.shape[0]):
            write_frame(
                scene_dir,
                frame_stem(view_index),
                images[view_index],
                record["mask"][view_index],
                record["cam2world"][view_index],
                record["azi_rot"][view_index],
                record["depth"][view_index],
            )
            n_frames += 1
        if "num_objects" in record:
            object_counts.append(int(record["num_objects"]))
        n_scenes = scene_index + 1

    meta = dict(meta)
    meta["n_scenes"] = n_scenes
    meta["n_frames"] = n_frames
    if object_counts:
        meta["objects_per_scene"] = object_counts
    if "split" not in meta:
        meta["split"] = make_split(
            n_scenes, meta.get("split_ratios", (0.8, 0.1, 0.1)), meta.get("seed", 0)
        )
    write_meta(root, meta)
    return root


def make_split(
    n_scenes: int, ratios=(0.8, 0.1, 0.1), seed: int = 0
) -> dict[str, list[int]]:
    """Deterministically split scene indices into train/val/test."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    indices = np.random.default_rng(seed).permutation(n_scenes)
    n_train = int(round(ratios[0] * n_scenes))
    n_val = int(round(ratios[1] * n_scenes))
    return {
        "train": sorted(int(i) for i in indices[:n_train]),
        "val": sorted(int(i) for i in indices[n_train : n_train + n_val]),
        "test": sorted(int(i) for i in indices[n_train + n_val :]),
    }


def write_meta(root, meta: dict) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / META_FILE
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def load_meta(root) -> dict:
    return json.loads((Path(root) / META_FILE).read_text())
