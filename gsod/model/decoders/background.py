"""Background decoder: background slot -> floor discs + fill colour.

Disc colour = the background's mean view-0 colour + MLP(PE(x), latent,
view-0 azimuth) residual. Nothing is read per pixel, so the background
cannot copy an object onto the floor. World-aligned.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gsod.model.decoders.objects import OBJECT_REGION, positional_encoding

_INNER_SIDE = 101  # grid over the object region (fine enough for texture)
_OUTER_SIDE = 61  # grid over [-_HALF, _HALF]^2, outside the region
_HALF = 3.0  # past view 0's camera
_PE_BANDS = 8
BACKGROUND_THICKNESS = 0.005
FLOOR_OPACITY_INIT = 0.95


def _mlp(d_in: int, hidden: int) -> nn.Sequential:
    """Two ReLU layers; BackgroundHead holds the output layers."""
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(True), nn.Linear(hidden, hidden), nn.ReLU(True)
    )


class BackgroundHead(nn.Module):
    """BackgroundDecoder's output layers."""

    def __init__(self, d_model: int, d_scene: int):
        super().__init__()
        self.opacity_out = nn.Linear(d_model, 1)
        self.colour_out = nn.Linear(d_model, 3)
        self.sh_out = nn.Linear(d_model, 9)
        self.fill_out = nn.Linear(d_scene, 3)
        with torch.no_grad():
            self.opacity_out.weight.mul_(0.1)
            self.opacity_out.bias.fill_(math.log(FLOOR_OPACITY_INIT / (1.0 - FLOOR_OPACITY_INIT)))
            self.colour_out.weight.mul_(0.1)
            self.colour_out.bias.zero_()
            self.sh_out.weight.zero_()  # view-independent at first
            self.sh_out.bias.zero_()
            self.fill_out.weight.zero_()  # the mean colour at first
            self.fill_out.bias.zero_()

    def opacity(self, feat: torch.Tensor) -> torch.Tensor:
        """feat (N, d) -> opacity (N,)."""
        return torch.sigmoid(self.opacity_out(feat)[:, 0])

    def fill(self, scene: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        """scene (d_scene,) -> fill colour (3,)."""
        return torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + self.fill_out(scene))

    def colour(self, feat: torch.Tensor, mean: torch.Tensor) -> dict[str, torch.Tensor]:
        """feat (N, d) -> rgb (residual on a base colour, (3,) or (N, 3)), sh1 (N, 3, 3)."""
        residual = self.colour_out(feat)
        return {
            "rgb": torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + residual),
            "sh1": self.sh_out(feat).unflatten(-1, (3, 3)),
        }


def floor_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Disc centres (N, 3) and spacing (N,): fine inside the object region."""
    def grid(half: float, side: int):
        lin = torch.linspace(-half, half, side)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x.flatten(), y.flatten()], -1), 2.0 * half / (side - 1)

    inner, s_in = grid(OBJECT_REGION, _INNER_SIDE)
    outer, s_out = grid(_HALF, _OUTER_SIDE)
    outer = outer[outer.abs().amax(-1) > OBJECT_REGION + 0.5 * s_in]
    xy = torch.cat([inner, outer])
    mu = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
    spacing = torch.cat([torch.full((len(inner),), s_in), torch.full((len(outer),), s_out)])
    return mu, spacing


class BackgroundDecoder(nn.Module):
    """Background slot -> floor discs and a fill colour."""

    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        # Inputs: PE(x), latent, azimuth.
        self.appearance = _mlp(3 + 6 * _PE_BANDS + z_dim + 2, hidden_dim)
        self.head = BackgroundHead(hidden_dim, z_dim + 2)
        mu, spacing = floor_grid()
        self.register_buffer("floor_mu", mu, persistent=False)
        scale = torch.stack([spacing, spacing, torch.full_like(spacing, BACKGROUND_THICKNESS)], -1)
        self.register_buffer("floor_scale", scale, persistent=False)

    def forward(
        self, latent: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame
    ) -> dict[str, torch.Tensor]:
        """latent (C,), attn (s, s) background attention, colours (3, s, s),
        view 0's SceneFrame and focal ratio -> Gaussians and "fill" (3,)."""
        mean = (attn * colours).sum((1, 2)) / attn.sum().clamp_min(1e-6)
        yaw = frame.yaw
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)]).to(latent.dtype)
        scene = torch.cat([latent, azimuth])  # (C + 2,)
        n = self.floor_mu.shape[0]
        pe = positional_encoding(self.floor_mu / _HALF, _PE_BANDS)
        feat = self.appearance(torch.cat([pe, scene.expand(n, -1)], -1))
        return {
            "mu": self.floor_mu,
            "scale": self.floor_scale,
            "quat": self.floor_mu.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1),
            "opacity": self.head.opacity(feat),
            **self.head.colour(feat, mean),
            "fill": self.head.fill(scene, mean),
        }
