"""Small building blocks shared by the model."""

from __future__ import annotations

import math

import torch
from torch import nn


def mlp(d_in: int, hidden: int, d_out: int, *, norm: bool = False, act=nn.ReLU) -> nn.Sequential:
    """Two linear layers with one activation: [LayerNorm,] Linear, act, Linear."""
    layers = [nn.LayerNorm(d_in)] if norm else []
    return nn.Sequential(*layers, nn.Linear(d_in, hidden), act(), nn.Linear(hidden, d_out))


def conv_block(c_in: int, c_out: int, stride: int = 1, up: bool = False) -> nn.Sequential:
    """3x3 conv (replicate padding), ReLU [, 2x bilinear upsampling]."""
    layers = [nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, padding_mode="replicate"), nn.ReLU(True)]
    if up:
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
    return nn.Sequential(*layers)


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def shift_colour(base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """A base colour in [0, 1] shifted by a residual in logit space."""
    return torch.sigmoid(torch.logit(base.clamp(0.02, 0.98)) + residual)
