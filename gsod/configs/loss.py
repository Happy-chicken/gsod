"""Loss configuration shared by all training profiles (see model/losses.py)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    # Reconstruction on the novel views 1..V-1.
    l1: float = 1.0
    ssim: float = 0.2
    # L1 on view 0, the encoded view.
    view0: float = 1.0
    # aux2d: 2D reconstruction of view 0 from its slots (L1).
    aux2d: float = 1.0
    # consistency: the view-0 rendered slot split follows the 2D slot masks
    # (scaled by the schedule in train.py).
    consistency: float = 0.05
