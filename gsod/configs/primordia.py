"""Primordia training profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.utils.paths import default_out_dir


@dataclass
class PrimordiaConfig(BaseConfig):
    """Profile for the generated Primordia dataset (the base data defaults)."""

    # Primordia checkpoint directory.
    out_dir: str = field(default_factory=lambda: default_out_dir("primordia"))
