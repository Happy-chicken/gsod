"""Shared GS slot model: encoder, slot attention, selectable Gaussian decoder.

The gsplat splatter lives in ``gsod/utils/splatter.py``. ``build_model`` is
kept here as the package-level factory, so callers can use::

    from gsod.model import GSODModel, build_model, load_networks

The decoder is chosen by passing its class, e.g.
``build_model(cfg, meta, decoder_cls=SharedMLPDecoder)``.
"""

from __future__ import annotations

from torch import nn

from gsod.configs import BaseConfig
from gsod.model.aux_decoder import SpatialBroadcastDecoder
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


def build_model(
    cfg: BaseConfig,
    meta: dict,
    decoder_cls: type[nn.Module] = AttentionDecoder,
) -> GSODModel:
    """Build a GSODModel from a config profile and dataset metadata.

    decoder_cls: ``AttentionDecoder``, ``SharedMLPDecoder`` or
    ``PerSlotMLPDecoder`` (any class with the common decoder signature).
    """
    height, width = meta["image_size"]
    return GSODModel(
        splat=GsplatSplatter(),
        image_size=(height, width),
        input_size=min(cfg.input_size, height, width),
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        gaussians_per_slot=cfg.gaussians_per_slot,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        aux2d_size=cfg.aux2d_size,
        aux2d_views=cfg.aux2d_views,
        focal_ratio=tuple(meta["focal_ratio"]),
        near=cfg.near,
        # Primordia meta.json carries the SceneGenConfig; CLEVR-567 has none
        # and uses the profile's scene_prior instead.
        scene_config=meta.get("config", cfg.scene_prior),
        decoder_cls=decoder_cls,
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
    "SpatialBroadcastDecoder",
    "build_model",
    "save_networks",
    "load_networks",
]
