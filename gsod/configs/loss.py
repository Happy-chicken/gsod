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
    # attn_render: the slot attention follows the rendered slot partition
    # (after discovery).
    attn_render: float = 0.1
    # fg_render: each object is drawn by the slot that attends to it (scaled
    # by the anchor weight).
    fg_render: float = 0.05
    # spread: a slot's Gaussians do not shrink to a point.
    spread: float = 0.1
