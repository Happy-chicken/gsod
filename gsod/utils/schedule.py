"""Schedules."""

from __future__ import annotations

import math


def cosine(t: float, start: float, end: float) -> float:
    """Cosine from start (t = 0) to end (t >= 1)."""
    t = min(max(t, 0.0), 1.0)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))
