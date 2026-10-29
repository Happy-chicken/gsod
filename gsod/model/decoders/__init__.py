"""Gaussian decoder for slots: ``AttentionDecoder``, whose
``GaussianParameterHead`` places each slot's Gaussians in its ``SlotFrame``.
"""

from __future__ import annotations

from gsod.model.decoders.attention_decoder import AttentionDecoder
from gsod.model.decoders.parameter_head import (
    PARAMS_PER_GAUSSIAN,
    GaussianParameterHead,
    SlotFrame,
)

__all__ = [
    "PARAMS_PER_GAUSSIAN",
    "GaussianParameterHead",
    "SlotFrame",
    "AttentionDecoder",
]
