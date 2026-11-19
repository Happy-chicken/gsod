"""Background decoder: background slot -> flat Gaussians on the floor.

A 10 x 10 grid of anchors over [-_HALF, _HALF]^2, each with GAUSSIANS_PER_ANCHOR
flat Gaussians (z = 0). An anchor's base colour is view 0's colour around it,
weighted by the shape of background attention (a / mean a) over a window as wide
as the largest object (an object covered by the background is smeared, not
copied). This is the only scene-specific colour: MLP(PE(anchor), view-0 azimuth)
adds per-Gaussian offsets, sizes, opacity and colour residuals. World-aligned.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.objects import OBJECT_RADIUS, positional_encoding

ANCHOR_SIDE = 10
GAUSSIANS_PER_ANCHOR = 64
_HALF = 3.0  # past view 0's camera
_SPACING = 2.0 * _HALF / ANCHOR_SIDE
_PE_BANDS = 8
BACKGROUND_THICKNESS = 0.005
FLOOR_OPACITY_INIT = 0.95
# Per Gaussian: offset (2), log size (2), opacity, rgb residual (3), sh1 (9).
_N_OUT = 17


def _anchors() -> torch.Tensor:
    """(A, 3) anchor centres on the floor."""
    lin = torch.linspace(-_HALF + 0.5 * _SPACING, _HALF - 0.5 * _SPACING, ANCHOR_SIDE)
    y, x = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([x.flatten(), y.flatten(), torch.zeros(x.numel())], -1)


def _cell_pattern() -> torch.Tensor:
    """(G, 2) initial offsets: a sunflower pattern filling one cell."""
    i = torch.arange(GAUSSIANS_PER_ANCHOR, dtype=torch.float32) + 0.5
    r = 0.5 * _SPACING * (i / GAUSSIANS_PER_ANCHOR).sqrt()
    angle = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * angle.cos(), r * angle.sin()], -1)


class BackgroundDecoder(nn.Module):
    """Background slot -> flat floor Gaussians."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        d_in = 3 + 6 * _PE_BANDS + 2  # PE(anchor), azimuth
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)
        )
        self.out = nn.Linear(hidden_dim, GAUSSIANS_PER_ANCHOR * _N_OUT)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            bias = self.out.bias.view(GAUSSIANS_PER_ANCHOR, _N_OUT)
            bias.zero_()
            bias[:, 4] = math.log(FLOOR_OPACITY_INIT / (1.0 - FLOOR_OPACITY_INIT))
            self.out.weight.view(GAUSSIANS_PER_ANCHOR, _N_OUT, -1)[:, 8:].zero_()  # sh1: view-independent at first
        self.register_buffer("anchors", _anchors(), persistent=False)
        self.register_buffer("pattern", _cell_pattern(), persistent=False)

    def base_colour(
        self, attn: torch.Tensor, colours: torch.Tensor, rays: torch.Tensor, origin: torch.Tensor
    ) -> torch.Tensor:
        """Background attention (N,), token colours and view-0 ray directions
        (N, 3), view 0's camera -> (A, 3): attention-weighted colour seen around
        each anchor; the mean colour counts as one extra token (anchors view 0
        does not see get the mean). Only the attention's shape matters."""
        attn = attn / attn.mean().clamp_min(1e-12)
        mean = (attn[:, None] * colours).sum(0) / attn.sum()
        theta = OBJECT_RADIUS[1] / origin.norm()  # largest object's angle at the centre
        d_anchor = F.normalize(self.anchors - origin, dim=-1)
        w = torch.exp(-torch.cdist(d_anchor, F.normalize(rays, dim=-1)).square() / (2 * theta**2))
        w = w * attn  # (A, N)
        return (w @ colours + mean) / (w.sum(-1, keepdim=True) + 1.0)

    def forward(
        self, attn: torch.Tensor, colours: torch.Tensor, rays: torch.Tensor, frame
    ) -> dict[str, torch.Tensor]:
        """Background attention (N,), token colours and view-0 ray directions
        (N, 3), view 0's SceneFrame -> Gaussians."""
        base = self.base_colour(attn, colours, rays, frame.origin)  # (A, 3)
        yaw = frame.yaw
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)]).to(base.dtype)
        n_anchor = self.anchors.shape[0]
        pe = positional_encoding(self.anchors / _HALF, _PE_BANDS)
        feat = self.mlp(torch.cat([pe, azimuth.expand(n_anchor, -1)], -1))
        raw = self.out(feat).view(n_anchor, GAUSSIANS_PER_ANCHOR, _N_OUT)

        xy = self.anchors[:, None, :2] + self.pattern + 0.5 * _SPACING * torch.tanh(raw[..., :2])
        size = _SPACING / math.sqrt(GAUSSIANS_PER_ANCHOR) * torch.exp(2.0 * torch.tanh(raw[..., 2:4]))
        rgb = torch.sigmoid(torch.logit(base.clamp(0.02, 0.98))[:, None] + raw[..., 5:8])
        n = n_anchor * GAUSSIANS_PER_ANCHOR
        return {
            "mu": torch.cat([xy, torch.zeros_like(xy[..., :1])], -1).reshape(n, 3),
            "scale": torch.cat([size, torch.full_like(size[..., :1], BACKGROUND_THICKNESS)], -1).reshape(n, 3),
            "quat": xy.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1),
            "opacity": torch.sigmoid(raw[..., 4]).reshape(n),
            "rgb": rgb.reshape(n, 3),
            "sh1": raw[..., 8:].reshape(n, 3, 3),
        }
