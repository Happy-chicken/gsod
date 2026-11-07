"""Dataset profiles: data paths, slot count and schedule overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from gsod.configs.base import BaseConfig
from gsod.utils.io import default_out_dir


@dataclass
class PrimordiaConfig(BaseConfig):
    """Primordia (generated here)."""

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


MODEL_FIELDS = (
    "input_size",
    "z_dim",
    "num_slots",
    "attn_iter",
    "gaussians_per_slot",
    "decoder_hidden_dim",
    "near",
    "oracle_masks",
)


def run_config(cfg, checkpoint: str | Path):
    """cfg with the model fields of the run's config.json, if any."""
    path = Path(checkpoint) / "config.json"
    if not path.is_file():
        print(f"no {path}: building the model from the {cfg.dataset_mode} profile")
        return cfg
    saved = json.loads(path.read_text())
    known = {f.name for f in fields(cfg)}
    return replace(cfg, **{k: saved[k] for k in MODEL_FIELDS if k in saved and k in known})
