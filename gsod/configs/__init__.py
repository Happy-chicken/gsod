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
]
