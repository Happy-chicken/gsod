"""Training loss: reconstruction, plus two capacity terms on the object
slots; no masks or targets.

    total = l1          * L1(render, image)            novel views 1..V-1
          + ssim        * (1 - SSIM(render, image))    novel views 1..V-1
          + view0       * L1(render, image)            view 0 (the encoded view)
          + appearance  * |colour residual|            object Gaussians
          + compactness * c * spread beyond an object  object Gaussians,
                                                       c from train.py

Every term reaches every module that produced it (end to end).

Reconstruction
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The image is the slots' soft z-buffer
    composite (gsod.py), so these terms are also the slots' competition for
    the pixels. The novel views carry the 3D signal.

appearance -- "one slot, one colour"
    Mean |per-Gaussian colour residual| of the object Gaussians (logit units,
    on top of the slot's base colour), weighted by opacity. Makes a slot a
    poor fit for two objects of different colours.

compactness -- "one slot, one object"
    Opacity-weighted mean of (distance to the slot's centre beyond an
    object's radius)^2. Off at first, so the slots start spread over the
    object box and compete for every object; ramped in by train.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs.loss import LossConfig


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

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        images: torch.Tensor,
        compactness_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """outputs: model output; images: (V,3,H,W); compactness_scale: the
        compactness schedule of this step (train.py). Returns (total,
        per-term values for logging)."""
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
        terms["appearance"] = outputs["appearance"]
        weights["appearance"] = cfg.appearance
        terms["compactness"] = outputs["compactness"]
        weights["compactness"] = cfg.compactness * compactness_scale

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
