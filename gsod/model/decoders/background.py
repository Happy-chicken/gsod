"""Background decoder: background slot -> discs on the floor plane, colour
MLP(PE(x), latent, view-0 azimuth), plus one fill colour. Walls are drawn
on the floor too. Nothing pixel-aligned, so it cannot draw a missed object;
with pixel_features each disc also reads view 0 where it projects, weighted
by the (detached) background attention. Scene frame, world-aligned.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gsod.model.decoders.objects import OBJECT_REGION

# (inner, outer) grid sides, by pixel_features.
_GRID_SIDES = {False: (51, 41), True: (101, 61)}
_HALF = 3.0  # past view 0's camera
_PE_BANDS = 8
BACKGROUND_THICKNESS = 0.005
FLOOR_OPACITY_INIT = 0.95


def _mlp(d_in: int, hidden: int) -> nn.Sequential:
    """Two ReLU layers; BackgroundHead holds the output layers."""
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.ReLU(True), nn.Linear(hidden, hidden), nn.ReLU(True)
    )


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


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
        """feat (N, d) -> rgb (residual on mean, (3,) or (N, 3)), sh1 (N, 3, 3)."""
        residual = self.colour_out(feat)
        return {
            "rgb": torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + residual),
            "sh1": self.sh_out(feat).unflatten(-1, (3, 3)),
        }


def floor_grid(inner_side: int, outer_side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Disc centres (N, 3) and spacing (N,): fine inside the object region."""
    def grid(half: float, side: int):
        lin = torch.linspace(-half, half, side)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x.flatten(), y.flatten()], -1), 2.0 * half / (side - 1)

    inner, s_in = grid(OBJECT_REGION, inner_side)
    outer, s_out = grid(_HALF, outer_side)
    outer = outer[outer.abs().amax(-1) > OBJECT_REGION + 0.5 * s_in]
    xy = torch.cat([inner, outer])
    mu = torch.cat([xy, torch.zeros_like(xy[:, :1])], -1)
    spacing = torch.cat([torch.full((len(inner),), s_in), torch.full((len(outer),), s_out)])
    return mu, spacing


class BackgroundDecoder(nn.Module):
    """Background slot -> floor discs and a fill colour."""

    def __init__(self, z_dim: int, hidden_dim: int, pixel_features: bool = False):
        super().__init__()
        self.pixel_features = bool(pixel_features)
        d_pixel = z_dim + 4 if self.pixel_features else 0
        self.appearance = _mlp(3 + 6 * _PE_BANDS + z_dim + 2 + d_pixel, hidden_dim)
        self.head = BackgroundHead(hidden_dim, z_dim + 2)
        mu, spacing = floor_grid(*_GRID_SIDES[self.pixel_features])
        self.register_buffer("floor_mu", mu, persistent=False)
        scale = torch.stack([spacing, spacing, torch.full_like(spacing, BACKGROUND_THICKNESS)], -1)
        self.register_buffer("floor_scale", scale, persistent=False)

    def forward(
        self,
        latent: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        yaw: torch.Tensor,
        pixels: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """latent (C,), attn (s, s) background attention, colours (3, s, s),
        yaw: view 0's azimuth; pixels: GSODModel.floor_pixels -> Gaussians
        and "fill" (3,)."""
        mean = (attn * colours).sum((1, 2)) / attn.sum().clamp_min(1e-6)
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)]).to(latent.dtype)
        scene = torch.cat([latent, azimuth])  # (C + 2,)
        n = self.floor_mu.shape[0]
        pe = positional_encoding(self.floor_mu / _HALF, _PE_BANDS)
        inputs = [pe, scene.expand(n, -1)]
        base = mean
        if self.pixel_features:
            w = pixels["w"][:, None]
            inputs += [w * pixels["features"], w * pixels["colours"], w]
            base = w * pixels["colours"] + (1.0 - w) * mean
        feat = self.appearance(torch.cat(inputs, -1))
        return {
            "mu": self.floor_mu,
            "scale": self.floor_scale,
            "quat": self.floor_mu.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1),
            "opacity": self.head.opacity(feat),
            **self.head.colour(feat, base),
            "fill": self.head.fill(scene, mean),
        }
