"""Discriminator for the adversarial loss (as in uORF): StyleGAN2-like
residual blocks with equalised learning rate; images in [-1, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

D_LR = 1e-3
D_BETAS = (0.0, 0.9)
R1_WEIGHT = 10.0
R1_EVERY = 32  # steps (lazy R1)


def _act(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, 0.2) * math.sqrt(2.0)


class EqualConv2d(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c_out, c_in, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out)) if bias else None
        self.scale = (c_in * k * k) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.weight.shape[-1] // 2)


class EqualLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.scale = d_in**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


class ResBlock(nn.Module):
    """Two 3x3 convs, then half resolution; 1x1 skip."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = EqualConv2d(c_in, c_in, 3)
        self.conv2 = EqualConv2d(c_in, c_out, 3)
        self.skip = EqualConv2d(c_in, c_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.avg_pool2d(_act(self.conv2(_act(self.conv1(x)))), 2)
        return (y + self.skip(F.avg_pool2d(x, 2))) / math.sqrt(2.0)


class Discriminator(nn.Module):
    """(B, 3, H, W) -> realism logit (B,)."""

    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 256, 256)):
        super().__init__()
        self.from_rgb = EqualConv2d(3, channels[0], 1)
        self.blocks = nn.Sequential(*(ResBlock(a, b) for a, b in zip(channels[:-1], channels[1:])))
        self.final = EqualConv2d(channels[-1], channels[-1], 3)
        self.fc = EqualLinear(channels[-1], channels[-1])
        self.out = EqualLinear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(_act(self.from_rgb(x)))
        x = _act(self.final(x)).mean((2, 3))
        return self.out(_act(self.fc(x)))[:, 0]


def d_loss(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logistic loss: (real term, fake term)."""
    return F.softplus(-real_logit).mean(), F.softplus(fake_logit).mean()


def g_loss(fake_logit: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss."""
    return F.softplus(-fake_logit).mean()


def r1_penalty(real: torch.Tensor, real_logit: torch.Tensor) -> torch.Tensor:
    """E ||grad_x D(x)||^2 on real images (real must require grad)."""
    (grad,) = torch.autograd.grad(real_logit.sum(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()
