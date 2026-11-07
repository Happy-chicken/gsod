"""Configuration: base config and dataset profiles."""

from gsod.configs.base import BaseConfig, LossConfig
from gsod.configs.datasets import (
    DATASETS,
    Clevr567Config,
    PrimordiaConfig,
    RoomChairConfig,
    RoomDiverseConfig,
    config_for,
    run_config,
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
    "run_config",
    "default_out_dir",
    "latest_run_dir",
]
