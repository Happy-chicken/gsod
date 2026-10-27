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
    # fg_cov: object slots must draw the object pixels (target F), all views.
    fg_cov: float = 0.1
    # bg_attn: slot attention hands the background pixels to the background.
    bg_attn: float = 0.1
    # fg_render: each object is drawn by the slot that attends to it.
    fg_render: float = 0.05
    # Step from which F is used: fg_cov, bg_attn and the object-pixel
    # weighting of the L1 terms. 0 = from the start, which works when the
    # background shell starts close to the data background (Primordia).
    residual_start: int = 0
    # Colour difference (max over RGB, in [0, 1]) that counts as foreground in F.
    residual_tau: float = 0.1
