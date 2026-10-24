"""Attach ground-truth instance masks to Primordia dataset items.

Wraps any dataset whose items contain "scene" (scene index) and "cam2world"
((N, 4, 4), world frame, as stored in img*_RT.txt). For every view in the item
the matching frame on disk is found by comparing camera matrices, so the
wrapper works whatever views/order the loader picked. Its mask is read from
img*_mask.png and resized (nearest) to the item's image resolution.

    from gsod.data_loader.gt_masks import WithGTMasks
    val_set = WithGTMasks(PrimordiaScenes(...), cfg.data_root)
    item = val_set[0]
    item["masks"]   # (N, H, W) int64, 0 = background, i = object i

Only needed for evaluation / diagnostics; training does not use masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gsod.data_generator.io import find_scene_frames, scene_dirname

__all__ = ["WithGTMasks"]


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resize_nearest(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    if (h, w) == tuple(size):
        return mask
    rows = (np.arange(size[0]) * h) // size[0]
    cols = (np.arange(size[1]) * w) // size[1]
    return mask[rows[:, None], cols[None, :]]


class WithGTMasks:
    def __init__(self, dataset, root, cam_tol: float = 1e-4):
        self.dataset = dataset
        self.root = Path(root)
        self.cam_tol = float(cam_tol)
        self._cache: dict[int, list[tuple[np.ndarray, str]]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name):
        # Forward everything else (split lists, meta, ...) to the dataset.
        return getattr(self.dataset, name)

    def _frames(self, scene: int) -> list[tuple[np.ndarray, str]]:
        if scene not in self._cache:
            scene_dir = self.root / scene_dirname(scene)
            self._cache[scene] = [
                (np.loadtxt(scene_dir / f"{stem}_RT.txt"), stem)
                for stem in find_scene_frames(self.root, scene)
            ]
        return self._cache[scene]

    def _load_mask(self, scene: int, stem: str) -> np.ndarray:
        from PIL import Image

        path = self.root / scene_dirname(scene) / f"{stem}_mask.png"
        return np.asarray(Image.open(path).convert("L"), dtype=np.int64)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.dataset[index])
        scene = int(item["scene"])
        cams = _to_numpy(item["cam2world"])
        size = tuple(_to_numpy(item["images"]).shape[-2:])
        frames = self._frames(scene)

        masks = []
        for cam in cams:
            errors = [np.abs(cam - rt).max() for rt, _ in frames]
            best = int(np.argmin(errors))
            if errors[best] > self.cam_tol:
                raise ValueError(
                    f"scene {scene}: no frame on disk matches a loader camera "
                    f"(closest error {errors[best]:.2e}). Does the loader "
                    "transform cam2world? Masks need the original world cameras."
                )
            masks.append(_resize_nearest(self._load_mask(scene, frames[best][1]), size))
        masks = np.stack(masks)

        if hasattr(item["images"], "detach"):
            import torch

            item["masks"] = torch.from_numpy(masks)
        else:
            item["masks"] = masks
        return item
