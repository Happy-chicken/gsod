"""Gaussian decoder for slots: ``AttentionDecoder`` with a parameter head,
``ObjectHead`` for the object slots or ``BackgroundHead`` for the background
slot (parameter_head.py).
"""

from __future__ import annotations

from gsod.model.decoders.attention_decoder import AttentionDecoder
from gsod.model.decoders.parameter_head import BackgroundHead, ObjectHead, SceneBounds

__all__ = ["AttentionDecoder", "BackgroundHead", "ObjectHead", "SceneBounds"]
