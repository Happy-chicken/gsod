"""Gaussian decoders: ``ObjectDecoder`` for the object slots,
``BackgroundDecoder`` for the background slot; each builds its own head."""

from __future__ import annotations

from gsod.model.decoders.background import BackgroundDecoder
from gsod.model.decoders.objects import ObjectDecoder

__all__ = ["ObjectDecoder", "BackgroundDecoder"]
