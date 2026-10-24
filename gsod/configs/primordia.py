"""Primordia training profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.utils.paths import default_out_dir


@dataclass
class PrimordiaConfig(BaseConfig):
    """Full-size profile for the generated Primordia dataset."""

    # Use the folder-based Primordia loader.
    dataset_mode: str = "primordia"
    # Primordia root containing meta.json and scNNNN/ folders.
    data_root: str = "data/primordia"
    # Primordia checkpoint directory.
    out_dir: str = field(default_factory=lambda: default_out_dir("primordia"))
