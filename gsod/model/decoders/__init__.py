"""Gaussian decoders for slots.

* ``PixelGeometry`` + ``SlotMixer`` (pixel_gaussians.py): Gaussians along
  the pixel rays of view 0, whose appearance the slots compete for.
* ``AttentionDecoder``: the background slot's Gaussians outside view 0.
"""

from __future__ import annotations

from gsod.model.decoders.attention_decoder import AttentionDecoder
from gsod.model.decoders.parameter_head import (
    PARAMS_PER_GAUSSIAN,
    GaussianParameterHead,
    SceneBounds,
    SlotFrame,
)
from gsod.model.decoders.pixel_gaussians import PixelGeometry, SlotMixer

__all__ = [
    "PARAMS_PER_GAUSSIAN",
    "GaussianParameterHead",
    "SceneBounds",
    "SlotFrame",
    "AttentionDecoder",
    "PixelGeometry",
    "SlotMixer",
]
