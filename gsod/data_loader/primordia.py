"""Primordia dataset: one item per scene, all views stacked as tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gsod.data_generator.io import find_scene_frames, load_meta, read_view, scene_dirname

__all__ = ["PrimordiaScenes"]


class PrimordiaScenes(Dataset):
    """Item: scene, images (V, 3, H, W), cam2world (V, 4, 4)."""

    def __init__(self, root, split: str = "train"):
        self.root = Path(root)
        meta = load_meta(self.root)
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
        return {"scene": scene, "images": images, "cam2world": cams}
