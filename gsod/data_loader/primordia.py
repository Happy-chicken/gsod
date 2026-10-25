"""Primordia dataset: one item per scene, all views stacked as tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gsod.data_generator.io import (
    find_scene_frames,
    load_meta,
    read_view,
    scene_dirname,
)

__all__ = ["PrimordiaScenes", "hit_distance_to_depth"]


def hit_distance_to_depth(depth: np.ndarray, focal_ratio) -> np.ndarray:
    """Convert stored hit distances (along unit rays) to camera-space z.

    The renderer produces z (distance along the optical axis). For a pixel with
    normalised image coords (x, y), the unit ray has z = 1 / sqrt(x^2 + y^2 + 1).
    """
    h, w = depth.shape[-2:]
    fx, fy = focal_ratio[0] * w, focal_ratio[1] * h
    us = np.arange(w, dtype=np.float64)
    vs = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - (w - 1) / 2.0) / fx
    y = (vv - (h - 1) / 2.0) / fy
    return (depth / np.sqrt(x * x + y * y + 1.0)).astype(np.float32)


class PrimordiaScenes(Dataset):
    """Scenes of one split. Item i is a dict with the scene index and

    images:    (V, 3, H, W) float32 in [0, 1]
    cam2world: (V, 4, 4) float32
    depths:    (V, H, W) float32 camera-space z, 0 where there is no surface;
               None unless load_depth (training does not use depth)
    """

    def __init__(self, root, split: str = "train", *, load_depth: bool = False):
        self.root = Path(root)
        self.load_depth = load_depth
        meta = load_meta(self.root)
        self.meta = meta
        self.image_size = tuple(meta["image_size"])
        self.focal_ratio = tuple(meta["focal_ratio"])
        self.scene_indices = list(meta["split"][split])
        self.frames = {s: find_scene_frames(self.root, s) for s in self.scene_indices}

    def __len__(self) -> int:
        return len(self.scene_indices)

    def __getitem__(self, i: int) -> dict:
        scene = self.scene_indices[i]
        scene_dir = self.root / scene_dirname(scene)
        views = [read_view(scene_dir, stem) for stem in self.frames[scene]]
        images = np.stack([v["image"] for v in views])
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        cams = torch.tensor(
            np.stack([v["cam2world"] for v in views]), dtype=torch.float32
        )
        depths = None
        if self.load_depth:
            depths = torch.from_numpy(
                np.stack(
                    [
                        hit_distance_to_depth(
                            np.load(scene_dir / f"{stem}_depth.npy")[..., 0],
                            self.focal_ratio,
                        )
                        for stem in self.frames[scene]
                    ]
                )
            )
        return {
            "scene": scene,
            "images": images,
            "cam2world": cams,
            "depths": depths,
        }
