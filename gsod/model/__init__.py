"""Shared GS slot model: encoder, slot attention, selectable Gaussian decoder.

The gsplat splatter lives in ``gsod/utils/splatter.py``. ``build_model`` is
kept here as the package-level factory, so callers can use::

    from gsod.model import GSODModel, build_model, load_networks
"""

from gsod.configs import BaseConfig
from gsod.model.decoders import (
    AttentionDecoder,
    SharedMLPDecoder,
    GaussianParameterHead,
    PerSlotMLPDecoder,
    SlotFrame,
)
from gsod.model.checkpoint import load_networks, save_networks
from gsod.model.encoder import Encoder
from gsod.model.gsod import GSODModel
from gsod.model.slot_attention import SlotAttention
from gsod.utils.splatter import GsplatSplatter

# Manual decoder selection. Options:
#   "slot"         one MLP per slot
#   "fg_bg"        background MLP + shared foreground MLP
#   "attention"    Gaussian queries cross-attend to slot tokens
DECODER_TYPE = "attention"


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
    )


__all__ = [
    "Encoder",
    "SlotAttention",
    "PerSlotMLPDecoder",
    "SharedMLPDecoder",
    "AttentionDecoder",
    "GaussianParameterHead",
    "SlotFrame",
    "GSODModel",
    "build_model",
    "save_networks",
    "load_networks",
]
