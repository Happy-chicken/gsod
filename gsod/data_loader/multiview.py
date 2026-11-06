"""Multi-view scenes in a flat directory (uORF format): ``*sc0000*.png``
with ``_RT.txt`` (cam2world), optional ``_mask.png`` and ``_depth.npy``."""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["MultiViewScenes", "DEFAULT_FOCAL_RATIO", "scene_files"]

# fx / W, fy / H of the CLEVR-567 / Room-* renders.
DEFAULT_FOCAL_RATIO = (350.0 / 320.0, 350.0 / 240.0)

_SCENE_RE = re.compile(r"sc(\d+)")
_NUM_RE = re.compile(r"(\d+)")

# Auxiliary PNGs (not views).
_AUX_SUFFIXES = (
    "_mask.png",
    "_mask_for_moving.png",
    "_moved.png",
    "_changed.png",
    "_mask_for_bg.png",
    "_mask_for_providing_bg.png",
    "_providing_bg.png",
)


def _natural_key(path: Path) -> list[tuple[int, str | int]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in _NUM_RE.split(path.name)
    ]


def scene_files(root: str | Path) -> dict[int, list[Path]]:
    """scene id -> its view PNGs, in natural order."""
    scenes: dict[int, list[Path]] = {}
    for path in sorted(Path(root).glob("*.png"), key=_natural_key):
        if path.name.endswith(_AUX_SUFFIXES):
            continue
        match = _SCENE_RE.search(path.name)
        if match is not None:
            scenes.setdefault(int(match.group(1)), []).append(path)
    return scenes


def _load_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_mask(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Mask PNG -> (H, W) labels; the most frequent grey is background (0)."""
    mask = Image.open(path).convert("L").resize((size[1], size[0]), Image.NEAREST)
    gray = np.asarray(mask, dtype=np.int64)
    values, labels, counts = np.unique(gray, return_inverse=True, return_counts=True)
    labels = labels.reshape(gray.shape) + 1
    labels[gray == values[counts.argmax()]] = 0
    return torch.from_numpy(labels)


class MultiViewScenes(Dataset):
    """Item: scene, images (V, 3, H, W), cam2world (V, 4, 4), depths or None,
    masks (V, H, W) when load_masks and every view has one."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        image_size: int | tuple[int, int] = 128,
        focal_ratio: tuple[float, float] = DEFAULT_FOCAL_RATIO,
        n_views: int = 4,
        val_fraction: float = 0.1,
        seed: int = 0,
        load_masks: bool = False,
        load_depth: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.focal_ratio = (float(focal_ratio[0]), float(focal_ratio[1]))
        self.n_views = int(n_views)
        self.load_masks = load_masks
        self.load_depth = load_depth

        scenes = scene_files(self.root)
        if not scenes:
            raise RuntimeError(
                f"no scenes found in {self.root}. Expected files such as "
                "'00000_sc0000_img0.png' and '00000_sc0000_img0_RT.txt'."
            )

        all_ids = sorted(scenes)
        train_ids, val_ids = self._split_ids(all_ids, val_fraction, seed)
        if split == "train":
            chosen = train_ids
        elif split == "val":
            chosen = val_ids
        else:
            chosen = all_ids

        self.scenes = scenes
        self.scene_ids = chosen
        self.meta = {
            "image_size": [self.image_size[0], self.image_size[1]],
            "focal_ratio": [self.focal_ratio[0], self.focal_ratio[1]],
            "split": {
                "train": train_ids,
                "val": val_ids,
                "test": all_ids,
            },
            "n_scenes": len(all_ids),
            "n_views": self.n_views,
        }

    @staticmethod
    def _split_ids(
        ids: list[int], val_fraction: float, seed: int
    ) -> tuple[list[int], list[int]]:
        if len(ids) < 2 or val_fraction <= 0:
            return list(ids), []
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
        n_val = min(n_val, len(shuffled) - 1)
        val_ids = sorted(shuffled[:n_val])
        train_ids = sorted(shuffled[n_val:])
        return train_ids, val_ids

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict:
        scene = self.scene_ids[index]
        paths = self.scenes[scene]
        if self.n_views > 0:
            paths = paths[: self.n_views]

        images, cams, depths, masks = [], [], [], []
        has_depth = self.load_depth
        has_mask = self.load_masks
        for path in paths:
            images.append(_load_rgb(path, self.image_size))

            pose_path = path.with_name(path.stem + "_RT.txt")
            if not pose_path.is_file():
                raise FileNotFoundError(f"missing camera pose: {pose_path}")
            cams.append(torch.from_numpy(np.loadtxt(pose_path, dtype=np.float32)))

            if has_depth:
                depth_path = path.with_name(path.stem + "_depth.npy")
                if depth_path.is_file():
                    depth = np.load(depth_path)
                    depth = np.squeeze(depth)
                    depth = np.asarray(
                        Image.fromarray(depth.astype(np.float32), mode="F").resize(
                            (self.image_size[1], self.image_size[0]), Image.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                    depths.append(torch.from_numpy(depth.copy()))
                else:
                    has_depth = False

            if has_mask:
                mask_path = path.with_name(path.stem + "_mask.png")
                if mask_path.is_file():
                    masks.append(_load_mask(mask_path, self.image_size))
                else:
                    has_mask = False

        item = {
            "scene": scene,
            "images": torch.stack(images),
            "cam2world": torch.stack(cams),
            "depths": torch.stack(depths) if has_depth and depths else None,
        }
        if has_mask and masks:
            item["masks"] = torch.stack(masks)
        return item
