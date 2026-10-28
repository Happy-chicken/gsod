"""Gaussian decoders for slots.

Two families:

* MLP: ``PerSlotMLPDecoder`` or ``SharedMLPDecoder``
* Attention: ``AttentionDecoder``

All decoders share the constructor signature
``(n_slots, z_dim, gaussians_per_slot, hidden_dim, frame)`` and return the same
``(gaussians, slot_probs)`` interface; pass the class itself to ``GSODModel``.
"""

from __future__ import annotations

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
