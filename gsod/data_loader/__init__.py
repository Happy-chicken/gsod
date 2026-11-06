"""Datasets, one item per scene: Primordia (meta.json) or flat multi-view
directories (CLEVR-567, Room-*)."""

from __future__ import annotations

from pathlib import Path

from gsod.data_generator.io import load_meta
from gsod.data_loader.gt_masks import WithGTMasks
from gsod.data_loader.multiview import DEFAULT_FOCAL_RATIO, MultiViewScenes, scene_files
from gsod.data_loader.primordia import PrimordiaScenes, hit_distance_to_depth

__all__ = [
    "DEFAULT_FOCAL_RATIO",
    "MultiViewScenes",
    "PrimordiaScenes",
    "WithGTMasks",
    "dataset_meta",
    "hit_distance_to_depth",
    "make_dataset",
    "scene_files",
]


def make_dataset(
    root: str | Path,
    split: str,
    *,
    dataset_mode: str = "primordia",
    image_size: int | tuple[int, int] = 128,
    focal_ratio: tuple[float, float] = DEFAULT_FOCAL_RATIO,
    n_views: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
    load_masks: bool = False,
    load_depth: bool = False,
):
    """The dataset of ``split``."""
    if dataset_mode == "primordia":
        return PrimordiaScenes(root, split, load_depth=load_depth)
    return MultiViewScenes(
        root,
        split,
        image_size=image_size,
        focal_ratio=focal_ratio,
        n_views=n_views,
        val_fraction=val_fraction,
        seed=seed,
        load_masks=load_masks,
        load_depth=load_depth,
    )


def dataset_meta(
    root: str | Path,
    *,
    dataset_mode: str = "primordia",
    image_size: int | tuple[int, int] = 128,
    focal_ratio: tuple[float, float] = DEFAULT_FOCAL_RATIO,
) -> dict:
    """The metadata ``build_model`` needs, without loading images."""
    if dataset_mode == "primordia":
        return load_meta(root)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    return {
        "dataset": dataset_mode,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "focal_ratio": [float(focal_ratio[0]), float(focal_ratio[1])],
    }
