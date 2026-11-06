"""Configuration: BaseConfig and LossConfig (base.py), the dataset profiles
(datasets.py). The project trains on CUDA."""

from gsod.configs.base import BaseConfig, LossConfig
from gsod.configs.datasets import (
    DATASETS,
    Clevr567Config,
    PrimordiaConfig,
    RoomChairConfig,
    RoomDiverseConfig,
    config_for,
)
from gsod.utils.io import default_out_dir, latest_run_dir

__all__ = [
    "BaseConfig",
    "LossConfig",
    "DATASETS",
    "PrimordiaConfig",
    "Clevr567Config",
    "RoomChairConfig",
    "RoomDiverseConfig",
    "config_for",
    "default_out_dir",
    "latest_run_dir",
]
