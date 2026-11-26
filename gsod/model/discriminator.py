"""Conditional PatchGAN on novel views: is view i real, given view 0? Training only."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# DiffAugment: largest shift (fraction of the side), brightness / contrast jitter.
SHIFT = 0.125
JITTER = 0.2


class PatchDiscriminator(nn.Module):
    """[view 0, view i] (N, 6, H, W) in [0, 1] -> patch logits (N, 1, h, w)."""

    def __init__(self, width: int = 64):
        super().__init__()
        layers, c = [], 6
        for out in (width, 2 * width, 4 * width):
            layers += [nn.Conv2d(c, out, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
            c = out
        layers.append(nn.Conv2d(c, 1, 4, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        return self.net(2.0 * pairs - 1.0)


def pairs(images: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    """View 0 (B, 3, H, W), views (B, V', 3, H, W) -> augmented pairs (B * V', 6, H, W)."""
    cond = images[:, None].expand_as(views)
    return diff_augment(torch.cat([cond, views], 2).flatten(0, 1))


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Same random shift and brightness / contrast change for every channel of a sample."""
    n, _, h, w = x.shape
    gain = 1 + JITTER * (2 * torch.rand(n, 1, 1, 1, device=x.device) - 1)
    bias = JITTER * (2 * torch.rand(n, 1, 1, 1, device=x.device) - 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * gain + mean + bias
    # Shift: pad, then crop at a random offset per sample (as a translation grid).
    shift = SHIFT * (2 * torch.rand(n, 1, 1, 2, device=x.device) - 1) * 2  # normalised coordinates
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device), indexing="ij"
    )
    grid = torch.stack([xs, ys], -1)[None] + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=True)


def d_loss(disc: PatchDiscriminator, real: torch.Tensor, fake: torch.Tensor, r1: float) -> dict[str, torch.Tensor]:
    """Hinge loss on real / fake pairs (fake detached) + r1 / 2 * |grad of each sample's mean logit|^2."""
    real = real.detach().requires_grad_(r1 > 0)
    real_logits, fake_logits = disc(real), disc(fake.detach())
    loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    terms = {"gan_d": loss.detach(), "gan_real": real_logits.mean().detach(), "gan_fake": fake_logits.mean().detach()}
    if r1 > 0:  # one score per sample (the patch mean), not the patch sum
        (grad,) = torch.autograd.grad(real_logits.mean(dim=(1, 2, 3)).sum(), real, create_graph=True)
        penalty = grad.pow(2).flatten(1).sum(1).mean()
        loss = loss + 0.5 * r1 * penalty
        terms["gan_r1"] = penalty.detach()
    terms["loss"] = loss
    return terms


def g_loss(disc: PatchDiscriminator, fake: torch.Tensor) -> torch.Tensor:
    """Generator hinge loss: -D(fake)."""
    return -disc(fake).mean()
