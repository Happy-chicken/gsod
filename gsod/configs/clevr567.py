"""CLEVR-567 profile in the flat uORF file layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.configs.loss import LossConfig
from gsod.utils.paths import default_out_dir


@dataclass
class Clevr567Config(BaseConfig):
    """CLEVR-567 data profile.

    The model has no spatial anchors or scale priors; only the dataset paths and
    camera settings differ from the Primordia profile.
    """

    # Use the flat CLEVR-567 loader.
    dataset_mode: str = "clevr567"
    # CLEVR-567 training root (flat uORF layout).
    data_root: str = "clevr_567_train"
    # CLEVR-567 checkpoint directory.
    out_dir: str = field(default_factory=lambda: default_out_dir("clevr567"))

    # Resize images to 128x128 for rendering and evaluation.
    dataset_image_size: int = 128
    # uORF's default focal ratio for its 320x240 renders.
    dataset_focal_ratio: tuple[float, float] = (350.0 / 320.0, 350.0 / 240.0)
    # uORF's CLEVR-567 training uses four views per scene.
    n_views: int = 4
    # Hold out 10% of training scenes for validation.
    val_fraction: float = 0.1

    # Image-only loss: L1 + SSIM.
    loss: LossConfig = field(default_factory=LossConfig)
