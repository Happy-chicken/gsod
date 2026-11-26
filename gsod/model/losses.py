"""Reconstruction loss (no masks), every view; view 0 (the input) weighs view0 in the main terms
(MSE, SSIM, LPIPS), so copying the input is no shortcut:

total = recon * MSE (images in [-1, 1]) + ssim * (1 - SSIM)
        + refine_aux * (recon * MSE + ssim * (1 - SSIM)) before refinement
        + lpips * LPIPS (VGG) + box * mean object box half size, both from late_start of training on
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
        n_novel = images.shape[1] - 1

        def views(fn, rendered: torch.Tensor) -> torch.Tensor:  # (B, V, 3, H, W): view 0 weighs cfg.view0
            first = fn(rendered[:, 0], images[:, 0])
            novel = fn(rendered[:, 1:].flatten(0, 1), images[:, 1:].flatten(0, 1))
            return (cfg.view0 * first + n_novel * novel) / (cfg.view0 + n_novel)

        def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return F.mse_loss(2.0 * x - 1.0, 2.0 * y - 1.0)

        rendered = outputs["rgb"]
        terms = {"recon": views(mse, rendered), "ssim": 1.0 - views(ssim, rendered)}
        weights = {"recon": cfg.recon, "ssim": cfg.ssim}
        if "rgb_pre" in outputs and cfg.refine_aux > 0:
            pre, every = outputs["rgb_pre"].flatten(0, -4), images.flatten(0, -4)  # every view alike
            terms["refine_aux"] = cfg.recon * mse(pre, every) + cfg.ssim * (1.0 - ssim(pre, every))
            weights["refine_aux"] = cfg.refine_aux
        if progress >= cfg.late_start:  # late terms, once objects exist
            if self.lpips is not None:
                lpips = self.lpips

                def perceptual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return lpips(x, y, normalize=True).mean()  # inputs in [0, 1]

                terms["lpips"] = views(perceptual, rendered)
                weights["lpips"] = cfg.lpips
            if cfg.box > 0:  # object box half sizes, scene units
                terms["box"] = outputs["extent"].mean()
                weights["box"] = cfg.box
        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: value.detach() for name, value in terms.items()}
        logged["total"] = total.detach()
        return total, logged
