"""Training losses.

    total = l1        * L1(render, image)            novel views 1..V-1
          + ssim      * (1 - SSIM(render, image))    novel views 1..V-1
          + view0     * L1(render, image)            view 0 (the encoded view)
          + fg_render * CE(render_fg | attn_fg)      view 0

No foreground target: which pixels are background is learned, as in uORF.
The background slot colours the far-away background shell, which cannot
reproduce a near object in every view (background.py), so the novel-view
reconstruction is what keeps objects out of the background slot. The model's
opacity floor keeps object Gaussians alive until they reach the objects.

Reconstruction (L1, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The novel views carry the 3D signal; view 0
    is the encoded view, kept so its render matches the input.

fg_render -- "each object is drawn by the slot that attends to it"
    On pixels the attention gives to objects, the rendered split among object
    slots should follow the attention split (attention detached, so only the
    decoder learns). Removes renders where several slots redraw one object.
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


def attn_grid(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """(K, s, s) view-0 slot attention, normalised over slots (row 0 = background)."""
    attn = outputs["attn"]  # (K, s*s)
    n_slots, n_tokens = attn.shape
    side = int(round(n_tokens**0.5))
    attn = attn.view(n_slots, side, side)
    return attn / attn.sum(0, keepdim=True).clamp_min(_EPS)


def render_grid(outputs: dict[str, torch.Tensor], side: int) -> torch.Tensor:
    """(K, s, s) view-0 rendered slot weights incl. the uncovered share
    (slot_alpha_full), normalised over slots."""
    alpha = _to_tokens(outputs["slot_alpha_full"][0], side)
    return alpha / alpha.sum(0, keepdim=True).clamp_min(_EPS)


def fg_render_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """fg_render: rendered split among object slots follows the attention split."""
    attn = attn_grid(outputs)
    render = render_grid(outputs, attn.shape[-1])
    a_fg = attn[1:].detach()
    fg_mass = a_fg.sum(0)
    pix = fg_mass > 0.5  # tokens the attention gives to objects
    if not pix.any():
        return render.new_zeros(())
    a_fg = a_fg[:, pix] / fg_mass[pix]
    r_fg = render[1:, pix]
    r_fg = r_fg / r_fg.sum(0, keepdim=True).clamp_min(_EPS)
    return -(a_fg * torch.log(r_fg.clamp_min(_EPS))).sum(0).mean()


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        images: torch.Tensor,
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
        if cfg.fg_render > 0:
            terms["fg_render"] = fg_render_loss(outputs)
            weights["fg_render"] = cfg.fg_render

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
