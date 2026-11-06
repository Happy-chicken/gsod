"""Dataset profiles: the dataset-specific fields of BaseConfig.

A profile sets where the data is and how many slots a scene needs; the
scene's scale comes from view 0's camera (gsod/utils/geometry.py). Adding a
dataset means adding a profile here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.configs.base import BaseConfig
from gsod.utils.io import default_out_dir


@dataclass
class PrimordiaConfig(BaseConfig):
    """Primordia (generated here; meta.json carries its geometry)."""

    out_dir: str = field(default_factory=lambda: default_out_dir("primordia"))


@dataclass
class Clevr567Config(BaseConfig):
    """CLEVR-567: 5-7 objects + the background."""

    dataset_mode: str = "clevr567"
    data_root: str = "data/clevr/clevr_567_train"
    test_root: str | None = "data/clevr/clevr_567_test"
    num_slots: int = 8
    out_dir: str = field(default_factory=lambda: default_out_dir("clevr567"))


@dataclass
class RoomChairConfig(BaseConfig):
    """Room-Chair: 4 chairs + the background."""

    dataset_mode: str = "room_chair"
    data_root: str = "data/room_chair/room_chair_train"
    test_root: str | None = "data/room_chair/room_chair_test"
    num_slots: int = 5
    out_dir: str = field(default_factory=lambda: default_out_dir("room_chair"))


@dataclass
class RoomDiverseConfig(BaseConfig):
    """Room-Diverse: 4 chairs + the background; 5x Room-Chair's scenes."""

    dataset_mode: str = "room_diverse"
    data_root: str = "data/room_diverse/room_diverse_train"
    test_root: str | None = "data/room_diverse/room_diverse_test"
    num_slots: int = 5
    epochs: int = 40
    lr_warmup_epochs: float = 2
    draw_ramp_epochs: float = 4
    eval_every: int = 2
    save_every: int = 5
    out_dir: str = field(default_factory=lambda: default_out_dir("room_diverse"))


PROFILES: dict[str, type[BaseConfig]] = {
    "primordia": PrimordiaConfig,
    "clevr567": Clevr567Config,
    "room_chair": RoomChairConfig,
    "room_diverse": RoomDiverseConfig,
}
DATASETS = tuple(PROFILES)


def config_for(dataset_mode: str) -> BaseConfig:
    """The profile of a dataset."""
    if dataset_mode not in PROFILES:
        raise ValueError(f"dataset_mode must be one of {DATASETS}, got {dataset_mode!r}")
    return PROFILES[dataset_mode]()
