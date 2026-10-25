"""Training losses, shared by every dataset.

    total = l1          * L1(render, image)            novel views 1..V-1
          + ssim        * (1 - SSIM(render, image))    novel views 1..V-1
          + view0       * L1(render, image)            view 0 (the encoded view)
          + r(t) * consistency * KL_sym(attn, render)  view 0, all K slots
          + r(t) * opacity     * mean o (1 - o)        object Gaussians

r(t) = min(1, step / warmup) ramps the two structural terms in.

No term assumes anything about the background (uniform colour, colour
threshold, converged background render): the same loss runs on Primordia and
on CLEVR-567, whose floor, shadows, lighting jitter and grey objects break
such assumptions. What keeps objects out of the background is the model: the
background is a scene-independent module, so it cannot draw objects that
move and change from scene to scene; reconstruction alone forces object
slots to draw them.

Reconstruction (L1, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The novel views constrain depth and the
    occluded parts; view 0 gets its own term so the encoded view is fitted
    as tightly as all novel views together.

consistency -- "slot attention and the render describe the same partition"
    On view 0, attention a (softmax over the background query and the object
    slots) and the rendered slot weights r (slot_alpha_full, uncovered share
    counted as background), both normalised over the K slots per token:

        KL(sg(r) || a) + KL(sg(a) || r),   mean over tokens

    The first direction teaches the attention which slot draws a token
    (including background, replacing a background-pixel target); the second
    makes each object be drawn by the slot that attends to it, so several
    slots do not redraw one object. Each direction stops the gradient of its
    target. Ramped in because the initial render is almost all background.

opacity -- "a Gaussian belongs to an object or is not there"
    o (1 - o) is 0 for o in {0, 1}; it sharpens masks and stops half-opaque
    Gaussians from being shared by overlapping slots. Ramped in because at
    initialisation (o ~ 0.1) it pushes opacity down while objects form.
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


def view0_partitions(
    outputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot distributions of view 0 at token resolution, each (K, s, s).

    attn:   slot attention (softmax over slots; row 0 = background query).
    render: rendered slot weights incl. the uncovered share (slot_alpha_full),
            normalised over slots per pixel.
    """
    attn = outputs["attn"]  # (K, s*s)
    n_slots, n_tokens = attn.shape
    side = int(round(n_tokens**0.5))
    attn = attn.view(n_slots, side, side)
    attn = attn / attn.sum(0, keepdim=True).clamp_min(_EPS)

    alpha = outputs["slot_alpha_full"][0:1]  # (1, K, H, W)
    if alpha.shape[-2:] != (side, side):
        alpha = F.interpolate(alpha, size=(side, side), mode="area")
    render = alpha[0] / alpha[0].sum(0, keepdim=True).clamp_min(_EPS)
    return attn, render


def _kl(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """KL(target || pred) over dim 0 (slots), mean over tokens; target detached."""
    target = target.detach().clamp_min(_EPS)
    return (target * (target.log() - pred.clamp_min(_EPS).log())).sum(0).mean()


def consistency_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """consistency: symmetric KL between attention and render on view 0."""
    attn, render = view0_partitions(outputs)
    return _kl(render, attn) + _kl(attn, render)


def opacity_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """opacity: mean o (1 - o) over object Gaussians (slot id >= 1)."""
    opacity = outputs["gaussians"]["opacity"][outputs["slot_ids"] > 0]
    if opacity.numel() == 0:
        return outputs["rgb"].new_zeros(())
    return (opacity * (1.0 - opacity)).mean()


def l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return (pred - target).abs().mean()


class GSODLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def ramp(self, step: int | None) -> float:
        """Weight factor of the structural terms; 1 when step is None."""
        if step is None or self.cfg.warmup <= 0:
            return 1.0
        return min(1.0, step / self.cfg.warmup)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        images: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """outputs: model output; images: (V,3,H,W); step: training step
        (None = all terms at full weight). Returns (total, per-term values)."""
        cfg = self.cfg
        rendered, target = outputs["rgb"][1:], images[1:]
        terms: dict[str, torch.Tensor] = {
            "l1": l1(rendered, target),
            "ssim": 1.0 - ssim(rendered, target),
        }
        weights = {"l1": cfg.l1, "ssim": cfg.ssim}
        if cfg.view0 > 0:
            terms["view0"] = l1(outputs["rgb"][:1], images[:1])
            weights["view0"] = cfg.view0

        ramp = self.ramp(step)
        if cfg.consistency > 0 and ramp > 0:
            terms["consistency"] = consistency_loss(outputs)
            weights["consistency"] = ramp * cfg.consistency
        if cfg.opacity > 0 and ramp > 0:
            terms["opacity"] = opacity_loss(outputs)
            weights["opacity"] = ramp * cfg.opacity

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
