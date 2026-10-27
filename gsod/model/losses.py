"""Training losses.

    total = l1        * L1_w(render, image)          novel views 1..V-1
          + ssim      * (1 - SSIM(render, image))    novel views 1..V-1
          + view0     * L1_w(render, image)          view 0 (the encoded view)
          + fg_cov    * BCE_bal(fg_share, F)         all views, from residual_start
          + bg_attn   * BCE(attn_bg, 1 - F)          view 0, from residual_start
          + fg_render * CE(render_fg | attn_fg)      view 0
          + aux2d     * MSE(aux_rgb, image)          view 0, fg_source="attention"

F, the foreground target (background_residual_target)
    Per pixel of every view: does the image differ from what the background
    Gaussians alone render there? F ~ 1 on objects, ~ 0 elsewhere, no
    gradient. It does not depend on how well the objects are rendered yet, so
    it cannot reinforce a wrong render. It needs the background to be close
    to the data background; residual_start can delay it until then.

    F is what keeps the objects alive: with reconstruction alone, object
    Gaussians placed before slot attention has found the objects only add
    error over a background the shell already fits, so their opacity is
    driven to 0, the attention hands every token to the background and the
    slots collapse to one point (observed when F was removed).

Attention foreground (fg_source = "attention")
    Replaces F by the model's attn_fg: per view, 1 - the background row of
    that view's slot attention, from attn_fg_start on (no gradient). The slot
    attention is then trained by aux2d instead of bg_attn (which would
    supervise the attention with itself and is skipped), and before
    attn_fg_start the model's opacity floor keeps the object Gaussians from
    dying.

aux2d -- "the slots reconstruct view 0 in 2D"
    The 2D auxiliary decoder's reconstruction of view 0 from [background
    latent; object slots] against the image area-downsampled to its size;
    MSE, as in Slot Attention. Needs no foreground target.

Reconstruction (L1_w, SSIM)
    L1 rather than MSE (splatting speckle is over-penalised by L2); D-SSIM as
    a structure term, as in 3DGS. Objects cover ~12% of the pixels, so a plain
    mean lets the easy background dominate and object edges stay soft. With F
    the L1 terms weight each pixel by 1 + FG_RECON_WEIGHT * F (object pixels
    FG_RECON_WEIGHT + 1 times a background pixel); before that they are
    plain means.

fg_cov -- "objects must be drawn by object slots", in every view
    Per pixel, the share of the rendered alpha that belongs to object slots
    should equal F. Applied to all views: in the novel views it constrains each
    object's silhouette, so Gaussians smeared along the view-0 ray land on
    background pixels and are penalised. Balanced: the BCE is averaged over
    object pixels and over background pixels separately, so errors on the
    few object pixels (semi-transparent objects) are not diluted. Prevents the
    "everything is background" solution where object opacity dies.

bg_attn -- "slot attention must hand background pixels to the background"
    Per token of view 0, the background row of the slot attention should
    equal 1 - F. The background shell needs nothing from attention, so
    without this nothing stops one object slot from attending to the whole
    image. Only the bg/fg split is supervised; object slots still compete
    freely for the object pixels.

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
# Extra L1 weight on object pixels (weight = 1 + FG_RECON_WEIGHT * F).
FG_RECON_WEIGHT = 3.0


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


def background_residual_target(
    outputs: dict[str, torch.Tensor],
    images: torch.Tensor,
    tau: float = 0.1,
    softness: float = 0.02,
) -> torch.Tensor:
    """F: (V, H, W) soft foreground target for every view, in [0, 1], no gradient.

    Max-channel |image - background-only render| through a soft step at tau.
    On the generated data every object pixel differs from the background by
    >= 0.18, so tau = 0.1 separates them cleanly.
    """
    diff = (images - outputs["bg_rgb"]).abs().amax(dim=1)
    return torch.sigmoid((diff - tau) / softness).detach()


def weighted_l1(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None
) -> torch.Tensor:
    """Mean absolute error over (V, 3, H, W); per-pixel weights (V, H, W) or None."""
    err = (pred - target).abs().mean(dim=1)  # (V, H, W)
    if weight is None:
        return err.mean()
    return (weight * err).sum() / weight.sum().clamp_min(_EPS)


def fg_coverage_loss(
    outputs: dict[str, torch.Tensor], fg_target: torch.Tensor
) -> torch.Tensor:
    """fg_cov: rendered object-slot share matches F in every view (balanced BCE).

    fg_target: (V, H, W). The per-pixel BCE is averaged over object pixels and
    over background pixels separately (soft weights F and 1 - F), then the two
    means are averaged.
    """
    alpha = outputs["slot_alpha_full"]  # (V, K, H, W)
    fg_share = alpha[:, 1:].sum(1) / alpha.sum(1).clamp_min(_EPS)
    bce = F.binary_cross_entropy(
        fg_share.clamp(_EPS, 1.0 - _EPS), fg_target, reduction="none"
    )
    on_obj = (bce * fg_target).sum() / fg_target.sum().clamp_min(_EPS)
    on_bg = (bce * (1.0 - fg_target)).sum() / (1.0 - fg_target).sum().clamp_min(_EPS)
    return 0.5 * (on_obj + on_bg)


def bg_attention_loss(
    outputs: dict[str, torch.Tensor], fg_target: torch.Tensor
) -> torch.Tensor:
    """bg_attn: background row of the slot attention matches 1 - F (per token BCE).

    fg_target: (V, H, W); only view 0 is used (attention exists for view 0 only).
    """
    attn = attn_grid(outputs)
    target = _to_tokens(fg_target[:1], attn.shape[-1])[0]
    pred = attn[0].clamp(_EPS, 1.0 - _EPS)
    return F.binary_cross_entropy(pred, 1.0 - target)


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
        step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """outputs: model output; images: (V,3,H,W); step: training step
        (None = all terms active). Returns (total, per-term values for logging)."""
        cfg = self.cfg
        from_attention = "aux_rgb" in outputs
        fg_target = None
        if from_attention:
            if step is None or step >= cfg.attn_fg_start:
                fg_target = outputs["attn_fg"]
        elif step is None or step >= cfg.residual_start:
            fg_target = background_residual_target(outputs, images, cfg.residual_tau)
        weight = None if fg_target is None else 1.0 + FG_RECON_WEIGHT * fg_target

        rendered, target = outputs["rgb"][1:], images[1:]
        terms: dict[str, torch.Tensor] = {
            "l1": weighted_l1(rendered, target, None if weight is None else weight[1:]),
            "ssim": 1.0 - ssim(rendered, target),
        }
        weights = {"l1": cfg.l1, "ssim": cfg.ssim}
        if cfg.view0 > 0:
            terms["view0"] = weighted_l1(
                outputs["rgb"][:1], images[:1], None if weight is None else weight[:1]
            )
            weights["view0"] = cfg.view0

        if fg_target is not None:
            if cfg.fg_cov > 0:
                terms["fg_cov"] = fg_coverage_loss(outputs, fg_target)
                weights["fg_cov"] = cfg.fg_cov
            if cfg.bg_attn > 0 and not from_attention:
                terms["bg_attn"] = bg_attention_loss(outputs, fg_target)
                weights["bg_attn"] = cfg.bg_attn
        if cfg.fg_render > 0:
            terms["fg_render"] = fg_render_loss(outputs)
            weights["fg_render"] = cfg.fg_render
        if from_attention and cfg.aux2d > 0:
            aux = outputs["aux_rgb"]
            terms["aux2d"] = F.mse_loss(aux, _to_tokens(images[0], aux.shape[-1]))
            weights["aux2d"] = cfg.aux2d

        total = sum(weights[name] * value for name, value in terms.items())
        logged = {name: float(value.detach()) for name, value in terms.items()}
        logged["total"] = float(total.detach())
        return total, logged
