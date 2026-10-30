"""Gaussian decoder for the object slots: ``AttentionDecoder`` with an
``ObjectHead`` (parameter_head.py).
"""

from __future__ import annotations

from gsod.model.decoders.attention_decoder import AttentionDecoder
from gsod.model.decoders.parameter_head import ObjectHead, SceneBounds

__all__ = ["AttentionDecoder", "ObjectHead", "SceneBounds"]
