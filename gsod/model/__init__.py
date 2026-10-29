"""Shared GS slot model: encoder, slot attention, Gaussian decoder.

The gsplat splatter lives in ``gsod/utils/splatter.py``. ``build_model`` is
kept here as the package-level factory, so callers can use::

    from gsod.model import GSODModel, build_model, load_networks
"""

from __future__ import annotations

from gsod.configs import BaseConfig
from gsod.model.decoders import (
    AttentionDecoder,
    BackgroundHead,
    ObjectHead,
    SceneBounds,
)
from gsod.model.checkpoint import load_networks, save_networks
from gsod.model.encoder import Encoder
from gsod.model.gsod import GSODModel
from gsod.model.slot_attention import SlotAttention
from gsod.utils.splatter import GsplatSplatter


def build_model(cfg: BaseConfig, meta: dict) -> GSODModel:
    """Build a GSODModel from a config profile and dataset metadata."""
    height, width = meta["image_size"]
    return GSODModel(
        splat=GsplatSplatter(),
        image_size=(height, width),
        input_size=min(cfg.input_size, height, width),
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        gaussians_per_slot=cfg.gaussians_per_slot,
        bg_gaussians_per_slot=cfg.bg_gaussians_per_slot,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        focal_ratio=tuple(meta["focal_ratio"]),
        near=cfg.near,
        # Primordia meta.json carries the SceneGenConfig; CLEVR-567 has none
        # and uses the profile's scene_prior instead.
        scene_config=meta.get("config", cfg.scene_prior),
    )


__all__ = [
    "Encoder",
    "SlotAttention",
    "AttentionDecoder",
    "BackgroundHead",
    "ObjectHead",
    "SceneBounds",
    "GSODModel",
    "build_model",
    "save_networks",
    "load_networks",
]
