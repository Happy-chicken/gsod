"""GS slot model"""

from __future__ import annotations

from gsod.configs import BaseConfig
from gsod.model.encoder import Encoder
from gsod.model.gsod import GSODModel
from gsod.model.slot_attention import SlotAttention
from gsod.utils.io import load_networks


def build_model(cfg: BaseConfig) -> GSODModel:
    """Build a GSODModel from a config profile."""
    size = cfg.dataset_image_size
    return GSODModel(
        image_size=(size, size),
        input_size=min(cfg.input_size, size),
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        gaussians_per_slot=cfg.gaussians_per_slot,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        refine_steps=cfg.refine_steps,
        focal_ratio=tuple(cfg.dataset_focal_ratio),
        near=cfg.near,
    )


def load_model(cfg: BaseConfig, checkpoint, suffix: str, device) -> GSODModel:
    """The model of a run with its saved weights."""
    model = build_model(cfg).to(device)
    load_networks(checkpoint, suffix, model.networks(), device=device)
    return model


__all__ = [
    "Encoder",
    "SlotAttention",
    "GSODModel",
    "build_model",
    "load_model",
]
