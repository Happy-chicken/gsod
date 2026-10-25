"""Loss configuration shared by all training profiles (see model/losses.py).

The same weights are used for Primordia and CLEVR-567: no term assumes a
uniform background or a dataset-specific colour threshold.
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
    # attn_from_render: slot attention learns the rendered partition of view
    # 0 (background included), only where the render reconstructs the image.
    attn_from_render: float = 0.1
    # render_from_attn: on tokens the attention gives to objects, the render
    # splits them among object slots like the attention does.
    render_from_attn: float = 0.05
    # Per-pixel reconstruction error (max over RGB, in [0, 1]) at which the
    # render is no longer trusted as a target for the attention.
    trust_tau: float = 0.1
    # opacity: object Gaussians either opaque or transparent, o * (1 - o).
    # Off by default: early on it pushes object opacity towards 0.
    opacity: float = 0.0
    # The structural terms above are off until `start`, then ramp linearly to
    # their weight over `warmup` steps; reconstruction alone makes objects
    # appear first.
    start: int = 2_000
    warmup: int = 2_000
