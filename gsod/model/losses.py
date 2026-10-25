"""Training losses, shared by every dataset.

    total = l1        * L1(render, image)                    novel views 1..V-1
          + ssim      * (1 - SSIM(render, image))            novel views 1..V-1
          + view0     * L1(render, image)                    view 0 (encoded view)
          + r(t) * attn_from_render * KL_trust(sg(render) || attn)       view 0
          + r(t) * render_from_attn * CE_fg(sg(attn) -> render)          view 0
          + r(t) * opacity          * mean o (1 - o)          object Gaussians

r(t) is 0 before `start`, then ramps linearly to 1 over `warmup` steps.

No term assumes anything about the background (uniform colour, colour
threshold): the same loss runs on Primordia and on CLEVR-567.

Who decides what
----------------
* Whether a pixel shows an object is decided by reconstruction only. The
  background is a scene-independent module, so it cannot draw objects that
  change from scene to scene; object slots must.
* Which object slot owns a pixel is decided jointly by attention and render.

The direction of every structural term follows from that. The render
teaches the attention the full partition (background included), but nothing
ever pulls the render towards the background: an earlier symmetric version
(KL both ways, background included) collapsed -- the initial render is all
background, attention followed it, all object rows went to ~eps, slot
updates became identical uniform averages, every frame sat at the image
centre, and the objects never appeared.

attn_from_render -- "attention learns the rendered partition"
    On view 0, attention a (softmax over the background query and the object
    slots) and the rendered slot weights r (slot_alpha_full, uncovered share
    counted as background), both normalised over the K slots per token:

        sum_t w_t KL(sg(r_t) || a_t) / sum_t w_t,
        w_t = sg(clamp(1 - e_t / trust_tau, 0, 1))

    e_t is the render's reconstruction error at token t (max over RGB). Where
    the render misses an object (or draws one that is not there) e_t is large
    and the render is not used as a target, so attention is never taught
    "background" on an undrawn object. Replaces a background-pixel target
    computed from a background-only render.

render_from_attn -- "each object is drawn by the slot that attends to it"
    On tokens the attention gives to objects (object mass > 0.5), the render's
    split among object slots follows the attention's split (cross-entropy,
    attention detached). Only the split among objects: it never decides
    whether a pixel is background. Removes renders where several slots redraw
    one object.

opacity -- "a Gaussian belongs to an object or is not there" (off by default)
    o (1 - o) is 0 for o in {0, 1}. At initialisation (o ~ 0.1) it pushes
    opacity down, so enable it only once objects are drawn.

Reconstruction (L1, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. The novel views constrain depth and the
    occluded parts; view 0 gets its own term so the encoded view is fitted
    as tightly as all novel views together.
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


def _tokens(x: torch.Tensor, side: int) -> torch.Tensor:
    """(H, W) map -> (side, side) by area averaging."""
    x = x[None, None]
    if x.shape[-2:] != (side, side):
        x = F.interpolate(x, size=(side, side), mode="area")
    return x[0, 0]


def attn_from_render_loss(
    outputs: dict[str, torch.Tensor], images: torch.Tensor, trust_tau: float
) -> torch.Tensor:
    """attn_from_render: KL(sg(render) || attn) on view 0, weighted by how well
    the render reconstructs each token."""
    attn, render = view0_partitions(outputs)
    err = (outputs["rgb"][0] - images[0]).abs().amax(0).detach()
    trust = (1.0 - _tokens(err, attn.shape[-1]) / trust_tau).clamp(0.0, 1.0)
    target = render.detach().clamp_min(_EPS)
    kl = (target * (target.log() - attn.clamp_min(_EPS).log())).sum(0)
    return (trust * kl).sum() / trust.sum().clamp_min(_EPS)


def render_from_attn_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """render_from_attn: rendered split among object slots follows the
    attention split, on tokens the attention gives to objects."""
    attn, render = view0_partitions(outputs)
    a_fg = attn[1:].detach()
    fg_mass = a_fg.sum(0)
    pix = fg_mass > 0.5
    if not pix.any():
        return render.new_zeros(())
    a_fg = a_fg[:, pix] / fg_mass[pix]
    r_fg = render[1:, pix]
    r_fg = r_fg / r_fg.sum(0, keepdim=True).clamp_min(_EPS)
    return -(a_fg * torch.log(r_fg.clamp_min(_EPS))).sum(0).mean()


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
        cfg = self.cfg
        if step is None:
            return 1.0
        if cfg.warmup <= 0:
            return float(step >= cfg.start)
        return min(1.0, max(0.0, (step - cfg.start) / cfg.warmup))

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
        if cfg.attn_from_render > 0 and ramp > 0:
            terms["attn_from_render"] = attn_from_render_loss(
                outputs, images, cfg.trust_tau
            )
            weights["attn_from_render"] = ramp * cfg.attn_from_render
        if cfg.render_from_attn > 0 and ramp > 0:
            terms["render_from_attn"] = render_from_attn_loss(outputs)
            weights["render_from_attn"] = ramp * cfg.render_from_attn
        if cfg.opacity > 0 and ramp > 0:
            terms["opacity"] = opacity_loss(outputs)
            weights["opacity"] = ramp * cfg.opacity

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
