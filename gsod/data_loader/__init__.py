"""Datasets: multi-view scenes in flat directories (uORF format: CLEVR-567,
Room-*), one item per scene."""

from __future__ import annotations

from pathlib import Path

from gsod.data_loader.multiview import MultiViewScenes

__all__ = ["MultiViewScenes", "make_dataset"]


def make_dataset(cfg, root: str | Path, split: str, load_masks: bool = False) -> MultiViewScenes:
    """The split of the dataset at root, as the config profile describes it;
    load_masks: GT masks, for evaluation."""
    return MultiViewScenes(
        root,
        split,
        image_size=cfg.dataset_image_size,
        n_views=cfg.n_views,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        load_masks=load_masks,
    )
