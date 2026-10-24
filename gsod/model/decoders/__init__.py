"""Gaussian decoder registry.

Two families:

* MLP: ``PerSlotMLPDecoder`` or ``FgBgMLPDecoder``
* Attention: ``AttentionDecoder``

All decoders return the same ``(gaussians, slot_probs)`` interface.
"""

from __future__ import annotations

from torch import nn

from gsod.model.decoders.attention_decoder import AttentionDecoder
from gsod.model.decoders.mlp_decoder import SharedMLPDecoder, PerSlotMLPDecoder
from gsod.model.decoders.parameter_head import (
    PARAMS_PER_GAUSSIAN,
    GaussianParameterHead,
    SlotFrame,
)

__all__ = [
    "PARAMS_PER_GAUSSIAN",
    "GaussianParameterHead",
    "SlotFrame",
    "PerSlotMLPDecoder",
    "SharedMLPDecoder",
    "AttentionDecoder",
]
