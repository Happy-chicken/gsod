"""Configuration profiles and loss settings.

Use ``BaseConfig`` for new profiles. ``PrimordiaConfig`` and ``Clevr567Config``
are the two dataset profiles. The project trains on CUDA.
"""

from gsod.configs.base import BaseConfig
from gsod.configs.clevr567 import Clevr567Config
from gsod.configs.primordia import PrimordiaConfig
from gsod.configs.loss import LossConfig
from gsod.utils.paths import default_out_dir, latest_run_dir

__all__ = [
    "BaseConfig",
    "PrimordiaConfig",
    "Clevr567Config",
    "LossConfig",
    "default_out_dir",
    "latest_run_dir",
    "config_for",
]


def config_for(dataset_mode: str) -> BaseConfig:
    """Return the training profile for a dataset format."""
    if dataset_mode == "primordia":
        return PrimordiaConfig()
    if dataset_mode == "clevr567":
        return Clevr567Config()
    raise ValueError(
        f"dataset_mode must be 'primordia' or 'clevr567', got {dataset_mode!r}"
    )
