"""Reconstruction loss (no masks), uORF's terms on the novel views only:

    total = recon * MSE (images in [-1, 1]) + percept * MSE(VGG16 relu4_3)

View 0 is left out: both decoders copy it, so there any copy is right;
on the other views only correct 3D is. The perceptual term is on from
percept_start (uORF: epoch 20).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs.base import LossConfig

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Perceptual(nn.Module):
    """Frozen VGG16 up to relu4_3 (ImageNet weights, as uORF)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        self.net.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN)[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD)[:, None, None], persistent=False)

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y (B, 3, H, W) in [0, 1] -> MSE of their features."""
        f = self.net((torch.cat([x, y]) - self.mean) / self.std)
        return F.mse_loss(*f.chunk(2))


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


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg
        self.percept_on = True  # set by the trainer
        self.perceptual = Perceptual() if cfg.percept > 0 else None

    def forward(
        self, outputs: dict[str, torch.Tensor], images: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """-> (total, per-term values for logging)."""
        cfg = self.cfg
        rendered, images = outputs["rgb"][1:], images[1:]
        terms = {"recon": F.mse_loss(2.0 * rendered - 1.0, 2.0 * images - 1.0)}
        weights = {"recon": cfg.recon}
        if self.perceptual is not None and self.percept_on:
            terms["percept"] = self.perceptual(rendered, images)
            weights["percept"] = cfg.percept

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
