"""Reconstruction loss (no masks), every view:

total = recon * MSE (images in [-1, 1]) + ssim * (1 - SSIM)
        + lpips * LPIPS (VGG; from lpips_start of training on)
        + refine_aux * (recon * MSE + ssim * (1 - SSIM)) before refinement
        + box * mean object box half size
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs.base import LossConfig


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size)


def ssim(x: torch.Tensor, y: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
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
        self.lpips = None
        if cfg.lpips > 0:
            import lpips  # pip install lpips; VGG weights download on first use

            self.lpips = lpips.LPIPS(net="vgg", use_dropout=False, verbose=False).requires_grad_(False)

    def forward(
        self, outputs: dict[str, torch.Tensor], images: torch.Tensor, progress: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """progress: fraction of training done -> (total, per-term values for
        logging; tensors, so no GPU sync)."""
        cfg = self.cfg
        rendered, images = outputs["rgb"].flatten(0, -4), images.flatten(0, -4)  # every view of every scene
        terms = {
            "recon": F.mse_loss(2.0 * rendered - 1.0, 2.0 * images - 1.0),
            "ssim": 1.0 - ssim(rendered, images),
        }
        weights = {"recon": cfg.recon, "ssim": cfg.ssim}
        if "rgb_pre" in outputs and cfg.refine_aux > 0:
            pre = outputs["rgb_pre"].flatten(0, -4)
            terms["refine_aux"] = cfg.recon * F.mse_loss(2.0 * pre - 1.0, 2.0 * images - 1.0) + cfg.ssim * (
                1.0 - ssim(pre, images)
            )
            weights["refine_aux"] = cfg.refine_aux
        if cfg.box > 0:  # object box half sizes, scene units
            terms["box"] = outputs["extent"].mean()
            weights["box"] = cfg.box
        if self.lpips is not None and progress >= cfg.lpips_start:
            terms["lpips"] = self.lpips(rendered, images, normalize=True).mean()  # inputs in [0, 1]
            weights["lpips"] = cfg.lpips
        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: value.detach() for name, value in terms.items()}
        logged["total"] = total.detach()
        return total, logged
