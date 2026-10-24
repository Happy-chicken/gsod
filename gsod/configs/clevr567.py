"""CLEVR-567 profile in the flat uORF file layout."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.utils.paths import default_out_dir


@dataclass
class Clevr567Config(BaseConfig):
    """CLEVR-567 data profile; the CLEVR-567 data fields keep their base defaults."""

    # Use the flat CLEVR-567 loader.
    dataset_mode: str = "clevr567"
    # CLEVR-567 training root (flat uORF layout).
    data_root: str = "clevr_567_train"
    # CLEVR-567 checkpoint directory.
    out_dir: str = field(default_factory=lambda: default_out_dir("clevr567"))
