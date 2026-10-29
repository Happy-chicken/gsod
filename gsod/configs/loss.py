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
    # appearance: per-Gaussian colour residuals of the object slots.
    appearance: float = 0.02
    # compactness: object slots' spread beyond an object (scaled by the
    # schedule in train.py).
    compactness: float = 0.1
