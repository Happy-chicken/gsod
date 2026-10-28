"""Training loss: one reconstruction objective, no masks or targets.

    total = l1    * L1(render, image)               novel views 1..V-1
          + ssim  * (1 - SSIM(render, image))       novel views 1..V-1
          + view0 * L1(render, image)               view 0 (the encoded view)
          + aux2d * L1(2D recon, image)             every encoded view

Every term reaches every module that produced it (end to end); the 3D and
the 2D reconstruction share the encoder and the slot attention.

3D reconstruction (L1, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The novel views carry the 3D signal: a
    decomposition that is consistent across views.

aux2d -- "the slots reconstruct their image in 2D"
    The spatial broadcast decoder's reconstruction of each encoded view from
    that view's K slots, against the image area-downsampled to its size. K
    slots of equal capacity: drawing one part each costs less than one slot
    drawing everything, which is what makes slot attention decompose a scene
    (Slot Attention). L1 like the 3D term, so both send gradients of the
    same size into the shared modules and neither drowns the other.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.configs.loss import LossConfig

_EPS = 1e-6


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


def _to_tokens(maps: torch.Tensor, side: int) -> torch.Tensor:
    """(C, H, W) maps -> (C, side, side) by area averaging."""
    if maps.shape[-2:] == (side, side):
        return maps
    return F.interpolate(maps[None], size=(side, side), mode="area")[0]


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self, outputs: dict[str, torch.Tensor], images: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """outputs: model output; images: (V,3,H,W). Returns (total, per-term
        values for logging)."""
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
        if cfg.aux2d > 0:
            aux = outputs["aux_rgb"]  # (E, 3, a, a), views 0..E-1
            side = aux.shape[-1]
            target2d = F.interpolate(images[: aux.shape[0]], size=side, mode="area")
            terms["aux2d"] = F.l1_loss(aux, target2d)
            weights["aux2d"] = cfg.aux2d

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
