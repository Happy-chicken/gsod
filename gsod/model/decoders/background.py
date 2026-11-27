"""Background decoder: background slot -> flat Gaussians on the floor.

A 10 x 10 grid of anchors over [-FLOOR_HALF, FLOOR_HALF]^2, each with GAUSSIANS_PER_ANCHOR
flat Gaussians (z = 0); walls are drawn on the far floor. Colour = the
background's mean view-0 colour + MLP(PE(anchor), latent, view-0 azimuth)
residual, which also gives per-Gaussian offsets, sizes and opacity. Nothing is
read per pixel, so the background cannot copy an object. World-aligned.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gsod.model.layers import logit, positional_encoding, shift_colour

ANCHOR_SIDE = 10
GAUSSIANS_PER_ANCHOR = 64
FLOOR_HALF = 3.0  # past view 0's camera
_SPACING = 2.0 * FLOOR_HALF / ANCHOR_SIDE
_PE_BANDS = 8
BACKGROUND_THICKNESS = 0.005
FLOOR_OPACITY_INIT = 0.95
# Per Gaussian: offset (2), log size (2), opacity, rgb residual (3), sh1 (9).
_N_OUT = 17


def _anchors() -> torch.Tensor:
    """(A, 3) anchor centres on the floor."""
    lin = torch.linspace(-FLOOR_HALF + 0.5 * _SPACING, FLOOR_HALF - 0.5 * _SPACING, ANCHOR_SIDE)
    y, x = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([x.flatten(), y.flatten(), torch.zeros(x.numel())], -1)


def _cell_pattern() -> torch.Tensor:
    """(G, 2) initial offsets: a sunflower pattern filling one cell."""
    i = torch.arange(GAUSSIANS_PER_ANCHOR, dtype=torch.float32) + 0.5
    r = 0.5 * _SPACING * (i / GAUSSIANS_PER_ANCHOR).sqrt()
    angle = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * angle.cos(), r * angle.sin()], -1)


class BackgroundDecoder(nn.Module):
    """Background slots -> flat floor Gaussians, one set per scene."""

    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        d_in = 3 + 6 * _PE_BANDS + z_dim + 2  # PE(anchor), latent, azimuth
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)
        )
        self.to_out = nn.Linear(hidden_dim, GAUSSIANS_PER_ANCHOR * _N_OUT)
        with torch.no_grad():
            self.to_out.weight.mul_(0.1)
            bias = self.to_out.bias.view(GAUSSIANS_PER_ANCHOR, _N_OUT)
            bias.zero_()
            bias[:, 4] = logit(FLOOR_OPACITY_INIT)
            self.to_out.weight.view(GAUSSIANS_PER_ANCHOR, _N_OUT, -1)[:, 8:].zero_()  # sh1: view-independent at first
        self.register_buffer("anchors", _anchors(), persistent=False)
        self.register_buffer("pattern", _cell_pattern(), persistent=False)

    def forward(
        self, latent: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame
    ) -> dict[str, torch.Tensor]:
        """latent (B, C), background attention (B, N), token colours (B, N, 3),
        view 0's SceneFrames -> Gaussians (B, A*G, ...)."""
        b = latent.shape[0]
        mean = (attn[..., None] * colours).sum(-2) / attn.sum(-1, keepdim=True).clamp_min(1e-6)  # (B, 3)
        yaw = frame.yaw
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1).to(latent.dtype)
        n_anchor = self.anchors.shape[0]
        pe = positional_encoding(self.anchors / FLOOR_HALF, _PE_BANDS).expand(b, -1, -1)
        scene = torch.cat([latent, azimuth], -1)[:, None].expand(-1, n_anchor, -1)
        feat = self.mlp(torch.cat([pe, scene], -1))
        raw = self.to_out(feat).view(b, n_anchor, GAUSSIANS_PER_ANCHOR, _N_OUT)

        xy = self.anchors[:, None, :2] + self.pattern + 0.5 * _SPACING * torch.tanh(raw[..., :2])
        size = _SPACING / math.sqrt(GAUSSIANS_PER_ANCHOR) * torch.exp(2.0 * torch.tanh(raw[..., 2:4]))
        rgb = shift_colour(mean[:, None, None], raw[..., 5:8])
        n = n_anchor * GAUSSIANS_PER_ANCHOR
        return {
            "mu": torch.cat([xy, torch.zeros_like(xy[..., :1])], -1).reshape(b, n, 3),
            "scale": torch.cat([size, torch.full_like(size[..., :1], BACKGROUND_THICKNESS)], -1).reshape(b, n, 3),
            "quat": xy.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(b, n, -1),
            "opacity": torch.sigmoid(raw[..., 4]).reshape(b, n),
            "rgb": rgb.reshape(b, n, 3),
            "sh1": raw[..., 8:].reshape(b, n, 3, 3),
        }
