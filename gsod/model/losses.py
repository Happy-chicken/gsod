"""Training loss: one reconstruction objective, no masks or targets.

    total = l1    * L1(render, image)               novel views 1..V-1
          + ssim  * (1 - SSIM(render, image))       novel views 1..V-1
          + view0 * L1(render, image)               view 0 (the encoded view)
          + aux2d * L1(2D recon, image)             view 0
          + consistency * c * CE(render | 2D masks) view 0, c from train.py

Every term reaches every module that produced it (end to end), except the
2D masks in the consistency term, which are its fixed target; the 3D and
the 2D reconstruction share the encoder and the slot attention.

3D reconstruction (L1, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The novel views carry the 3D signal: a
    decomposition that is consistent across views.

aux2d -- "the slots reconstruct their image in 2D"
    The spatial broadcast decoder's reconstruction of view 0 from its K
    slots, against the image area-downsampled to its size. K
    slots of equal capacity: drawing one part each costs less than one slot
    drawing everything, which is what makes slot attention decompose a scene
    (Slot Attention). L1 like the 3D term, so both send gradients of the
    same size into the shared modules and neither drowns the other.

consistency -- "the 3D render follows the coarse 2D decomposition"
    Per pixel of view 0 (at the 2D size), the rendered slot split should
    follow the 2D decoder's slot masks, detached: only the 3D model learns,
    the 2D masks stay a product of 2D reconstruction alone. Pulls the
    background slot off the pixels 2D gives to objects and lines the 3D
    slots up with the 2D ones. Scaled by c (train.py): off until the 2D
    decomposition has formed, faded out late so that the 3D reconstruction
    can refine what the coarse 2D masks cannot resolve. A small epsilon in
    the normalisation keeps uncovered pixels at a uniform split with a
    usable gradient.
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


def consistency_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """consistency: view-0 rendered slot split follows the 2D slot masks."""
    target = outputs["aux_alpha"][0].detach()  # (K, a, a), sums to 1 per pixel
    render = _to_tokens(outputs["slot_alpha"][0], target.shape[-1]) + _EPS
    render = render / render.sum(0, keepdim=True)
    return -(target * torch.log(render)).sum(0).mean()


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        images: torch.Tensor,
        consistency_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """outputs: model output; images: (V,3,H,W); consistency_scale: the
        consistency schedule of this step (train.py). Returns (total,
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
        if cfg.aux2d > 0:
            aux = outputs["aux_rgb"]  # (1, 3, a, a), view 0
            target2d = F.interpolate(images[:1], size=aux.shape[-1], mode="area")
            terms["aux2d"] = F.l1_loss(aux, target2d)
            weights["aux2d"] = cfg.aux2d
        if cfg.consistency > 0 and consistency_scale > 0:
            terms["consistency"] = consistency_loss(outputs)
            weights["consistency"] = cfg.consistency * consistency_scale

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
