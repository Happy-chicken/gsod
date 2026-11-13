"""Decoders: ``ObjectDecoder`` / ``BackgroundDecoder`` (Gaussians) and
``AttnDecoder`` (2D attention reconstruction)."""

from __future__ import annotations

from gsod.model.decoders.attention import AttnDecoder
from gsod.model.decoders.background import BackgroundDecoder
from gsod.model.decoders.objects import ObjectDecoder

__all__ = ["AttnDecoder", "ObjectDecoder", "BackgroundDecoder"]
