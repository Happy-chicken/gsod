"""Loss configuration shared by all training profiles (see model/losses.py).

The same weights are used for Primordia and CLEVR-567: no term assumes a
uniform background, a colour threshold or a dataset-specific schedule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    # Reconstruction on the novel views 1..V-1.
    l1: float = 1.0
    ssim: float = 0.2
    # L1 on view 0, the encoded view.
    view0: float = 1.0
    # consistency: slot attention and the rendered slot partition of view 0
    # agree (symmetric KL, background slot included).
    consistency: float = 0.1
    # opacity: object Gaussians are either opaque or transparent, o * (1 - o).
    opacity: float = 0.01
    # Steps over which consistency and opacity ramp linearly from 0 to their
    # weight, so reconstruction first makes the objects appear.
    warmup: int = 2_000
