"""Reconstruction loss (no masks):

    total = l1 * L1 + ssim * (1 - SSIM)   novel views
          + view0 * L1                    view 0
          + percept * MSE(VGG16 relu4_3)  every view, once on (as uORF)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs.base import LossConfig


def _gaussian_window(
    size: int, sigma: float, channels: int, device, dtype
) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size)


def ssim(
    x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Mean SSIM over channels and batches; x, y in [0, 1], shape (B, C, H, W)."""
    window = _gaussian_window(size, sigma, x.shape[1], x.device, x.dtype)
    pad = size // 2
    groups = x.shape[1]
    mu_x = F.conv2d(x, window, padding=pad, groups=groups)
    mu_y = F.conv2d(y, window, padding=pad, groups=groups)
    mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    sig_xx = F.conv2d(x * x, window, padding=pad, groups=groups) - mu_xx
    sig_yy = F.conv2d(y * y, window, padding=pad, groups=groups) - mu_yy
    sig_xy = F.conv2d(x * y, window, padding=pad, groups=groups) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
    den = (mu_xx + mu_yy + c1) * (sig_xx + sig_yy + c2)
    return (num / den).mean()


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Perceptual(nn.Module):
    """Frozen ImageNet VGG16 up to relu4_3 (uORF's perceptual network)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        self.net.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)  # always eval

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y (B, 3, H, W) in [0, 1] -> feature MSE."""
        return F.mse_loss(self.net((x - self.mean) / self.std), self.net((y - self.mean) / self.std))


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.perceptual = Perceptual() if cfg.percept > 0 else None
        # train.py turns the perceptual term on after cfg.percept_from.
        self.percept_on = False

    def forward(
        self, outputs: dict[str, torch.Tensor], images: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """-> (total, per-term values for logging)."""
        cfg = self.cfg
        rendered, target = outputs["rgb"][1:], images[1:]
        terms: dict[str, torch.Tensor] = {
            "l1": F.l1_loss(rendered, target),
            "ssim": 1.0 - ssim(rendered, target),
        }
        weights = {"l1": cfg.l1, "ssim": cfg.ssim}
        if cfg.view0 > 0:
            terms["view0"] = F.l1_loss(outputs["rgb"][:1], images[:1])
            weights["view0"] = cfg.view0
        if self.perceptual is not None and self.percept_on:
            terms["percept"] = self.perceptual(outputs["rgb"], images)
            weights["percept"] = cfg.percept

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
