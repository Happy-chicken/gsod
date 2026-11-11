"""GS slot model"""

from __future__ import annotations

from gsod.configs import BaseConfig
from gsod.model.encoder import Encoder
from gsod.model.gsod import GSODModel
from gsod.model.slot_attention import SlotAttention


def build_model(cfg: BaseConfig, meta: dict) -> GSODModel:
    """Build a GSODModel from a config profile and dataset metadata."""
    height, width = meta["image_size"]
    return GSODModel(
        image_size=(height, width),
        input_size=min(cfg.input_size, height, width),
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        cell_side=cfg.cell_side,
        gaussians_per_cell=cfg.gaussians_per_cell,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        focal_ratio=tuple(meta["focal_ratio"]),
        near=cfg.near,
    )


__all__ = [
    "Encoder",
    "SlotAttention",
    "GSODModel",
    "build_model",
]
