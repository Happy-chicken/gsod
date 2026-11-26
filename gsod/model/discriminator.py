"""GAN as in uORF: a StyleGAN2-style discriminator on 64 x 64 crops of every view, logistic
(non-saturating) losses, lazy R1. Training only."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

CROP = 64  # side of the random crop (same place for a view's render and GT)
R1_EVERY = 32  # lazy R1: one extra discriminator step every this many
LR, BETAS = 1e-3, (0.0, 0.9)  # discriminator optimiser (no schedule)


class EqualConv2d(nn.Module):
    """Conv with weights scaled at run time by 1 / sqrt(fan in) (equalised learning rate)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(c_out, c_in, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.scale, self.padding = 1 / math.sqrt(c_in * kernel**2), padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class EqualLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.scale = 1 / math.sqrt(d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias)


def act(x: torch.Tensor) -> torch.Tensor:
    """StyleGAN2's leaky ReLU, scaled to keep the variance."""
    return F.leaky_relu(x, 0.2) * math.sqrt(2)


class ResBlock(nn.Module):
    """Two 3 x 3 convs and a 1 x 1 skip, both halving the resolution."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = EqualConv2d(c_in, c_in, 3, padding=1)
        self.conv2 = EqualConv2d(c_in, c_out, 3, padding=1)
        self.skip = EqualConv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.avg_pool2d(act(self.conv2(act(self.conv1(x)))), 2)
        return (out + self.skip(F.avg_pool2d(x, 2))) / math.sqrt(2)


class Discriminator(nn.Module):
    """Images (N, 3, 64, 64) in [0, 1] -> one logit each (N, 1). Channels as uORF with ndf = 64."""

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.stem = EqualConv2d(3, ndf // 2, 1)
        chans = [ndf // 2, ndf, ndf, 2 * ndf, 2 * ndf]  # 64 -> 32 -> 16 -> 8 -> 4 px
        self.blocks = nn.Sequential(*(ResBlock(a, b) for a, b in zip(chans[:-1], chans[1:])))
        self.final = EqualConv2d(2 * ndf + 1, 2 * ndf, 3, padding=1)
        self.head1 = EqualLinear(2 * ndf * 4 * 4, 2 * ndf)
        self.head2 = EqualLinear(2 * ndf, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.blocks(act(self.stem(2.0 * images - 1.0)))
        x = act(self.final(torch.cat([x, minibatch_stddev(x)], 1)))
        return self.head2(act(self.head1(x.flatten(1))))


def minibatch_stddev(x: torch.Tensor, group: int = 4) -> torch.Tensor:
    """One channel: the std over groups of samples, averaged (StyleGAN2)."""
    n, c, h, w = x.shape
    g = math.gcd(min(n, group), n)
    std = x.view(g, -1, c, h, w).var(0, unbiased=False).add(1e-8).sqrt().mean(dim=(1, 2, 3))  # (n / g,)
    return std.view(-1, 1, 1, 1).repeat(g, 1, h, w)


def crops(renders: torch.Tensor, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every view of the renders and GT (B, V, 3, H, W) -> one random CROP x CROP crop per view,
    the same place in both: fake, real (B * V, 3, CROP, CROP)."""
    renders, images = renders.flatten(0, 1), images.flatten(0, 1)
    n, _, h, w = images.shape
    top = torch.randint(0, h - CROP + 1, (n,), device=images.device)
    left = torch.randint(0, w - CROP + 1, (n,), device=images.device)
    rows = (top[:, None] + torch.arange(CROP, device=images.device))[:, :, None]
    cols = (left[:, None] + torch.arange(CROP, device=images.device))[:, None, :]
    idx = torch.arange(n, device=images.device)[:, None, None]
    fake, real = (x[idx, :, rows, cols].permute(0, 3, 1, 2) for x in (renders, images))
    return fake, real


def d_loss(disc: Discriminator, real: torch.Tensor, fake: torch.Tensor) -> dict[str, torch.Tensor]:
    """Logistic loss: softplus(-D(real)) + softplus(D(fake)), fake detached."""
    real_logits, fake_logits = disc(real), disc(fake.detach())
    loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    return {
        "loss": loss,
        "gan_d": loss.detach(),
        "gan_real": real_logits.mean().detach(),
        "gan_fake": fake_logits.mean().detach(),
    }


def r1_penalty(disc: Discriminator, real: torch.Tensor) -> torch.Tensor:
    """|grad D(real)|^2 per image, averaged."""
    real = real.detach().requires_grad_()
    (grad,) = torch.autograd.grad(disc(real).sum(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


def g_loss(disc: Discriminator, fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: softplus(-D(fake))."""
    return F.softplus(-disc(fake)).mean()
