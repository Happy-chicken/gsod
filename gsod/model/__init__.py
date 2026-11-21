"""GS slot model"""

from __future__ import annotations

from gsod.configs import BaseConfig
from gsod.data_loader import dataset_meta
from gsod.model.encoder import Encoder
from gsod.model.gsod import GSODModel
from gsod.model.slot_attention import SlotAttention
from gsod.utils.io import load_networks


def build_model(cfg: BaseConfig, meta: dict) -> GSODModel:
    """Build a GSODModel from a config profile and dataset metadata."""
    height, width = meta["image_size"]
    return GSODModel(
        image_size=(height, width),
        input_size=min(cfg.input_size, height, width),
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        gaussians_per_slot=cfg.gaussians_per_slot,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        focal_ratio=tuple(meta["focal_ratio"]),
        near=cfg.near,
    )


def load_model(cfg: BaseConfig, root, checkpoint, suffix: str, device) -> GSODModel:
    """The model of a run (built for the dataset at root) with its saved weights."""
    meta = dataset_meta(
        root, dataset_mode=cfg.dataset_mode, image_size=cfg.dataset_image_size, focal_ratio=cfg.dataset_focal_ratio
    )
    model = build_model(cfg, meta).to(device)
    load_networks(checkpoint, suffix, model.networks(), device=device)
    return model


__all__ = [
    "Encoder",
    "SlotAttention",
    "GSODModel",
    "build_model",
    "load_model",
]
