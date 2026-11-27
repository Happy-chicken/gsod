"""Decoders: ``ObjectDecoder`` (a local Splatter Image per object slot) / ``BackgroundDecoder`` (the floor)."""

from __future__ import annotations

from gsod.model.decoders.background import BackgroundDecoder
from gsod.model.decoders.objects import ObjectDecoder

__all__ = ["BackgroundDecoder", "ObjectDecoder"]
