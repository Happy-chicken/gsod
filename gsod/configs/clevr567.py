"""CLEVR-567 profile in the flat uORF file layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.utils.paths import default_out_dir


def _clevr_scene_prior() -> dict:
    """CLEVR generator geometry (check with scripts/inspect_clevr567.py).

    Objects: centres x, y ~ U(-3, 3) (so up to 3 * sqrt(2) from the origin),
    radius 0.35 (small) or 0.7 (large), resting on the floor z = 0. Cameras
    look at the origin; distance_range should cover the measured camera
    distances.
    """
    return {
        "radius_range": [0.35, 0.7],
        "placement_radius": 4.25,
        "floor_z": 0.0,
        "distance_range": [10.0, 14.0],
        "target_height": 0.0,
    }


@dataclass
class Clevr567Config(BaseConfig):
    """CLEVR-567 data profile; the CLEVR-567 data fields keep their base defaults."""

    # Use the flat CLEVR-567 loader.
    dataset_mode: str = "clevr567"
    # uORF's CLEVR-567 train / test roots (flat uORF layout).
    data_root: str = "data/clevr/clevr_567_train"
    test_root: str | None = "data/clevr/clevr_567_test"
    # 5-7 objects per scene + spare slots + the empty background.
    num_slots: int = 10
    scene_prior: dict | None = field(default_factory=_clevr_scene_prior)
    # CLEVR-567 checkpoint directory.
    out_dir: str = field(default_factory=lambda: default_out_dir("clevr567"))
