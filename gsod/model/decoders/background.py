"""Background decoder: global slot -> floor discs + fill colour (disc colour
= mean view-0 background colour + MLP(PE(x), latent, azimuth)); detail slots
-> a few 2D Gaussians on the floor that cover the discs' colour where they lie.
No pixel is read. World-aligned.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gsod.model.decoders.objects import OBJECT_RADIUS, OBJECT_REGION, positional_encoding

_INNER_SIDE = 101  # grid over the object region (fine enough for texture)
_OUTER_SIDE = 61  # grid over [-_HALF, _HALF]^2, outside the region
_HALF = 3.0  # past view 0's camera
_PE_BANDS = 8
BACKGROUND_THICKNESS = 0.005
FLOOR_OPACITY_INIT = 0.95
# Detail slots: 2D Gaussians each, initial opacity, initial spread radius (R).
DETAIL_GAUSSIANS = 128
DETAIL_OPACITY_INIT = 0.05
DETAIL_RADIUS_INIT = 0.3


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


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class DetailDecoder(nn.Module):
    """Detail slots -> DETAIL_GAUSSIANS 2D Gaussians each on the floor (z = 0)."""

    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(DETAIL_GAUSSIANS, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, 9),  # xy 2, log scale 2, angle 1, opacity 1, rgb 3
        )
        self.radius = nn.Linear(z_dim, 1)
        self.scale_range = (2.0 * OBJECT_REGION / (_INNER_SIDE - 1), OBJECT_RADIUS[1])
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.bias.zero_()
            out.bias[5] = _logit(DETAIL_OPACITY_INIT)
            self.radius.weight.zero_()
            self.radius.bias.fill_(math.log(math.expm1(DETAIL_RADIUS_INIT)))

    def forward(
        self, latents: torch.Tensor, anchors: torch.Tensor, mean: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """latents (B, C), floor anchors (B, 2), mean colour (3,) -> Gaussians (B, G, ...)."""
        b, g = latents.shape[0], DETAIL_GAUSSIANS
        raw = self.mlp(torch.cat([self.queries.expand(b, -1, -1), latents[:, None].expand(-1, g, -1)], -1))
        radius = F.softplus(self.radius(latents))[:, None]  # (B, 1, 1)
        lo, hi = (math.log(s) for s in self.scale_range)
        return {
            "xy": anchors[:, None] + radius * torch.tanh(raw[..., 0:2]),
            "scale": torch.exp(lo + (hi - lo) * torch.sigmoid(raw[..., 2:4])),
            "angle": math.pi * torch.tanh(raw[..., 4]),
            "opacity": torch.sigmoid(raw[..., 5]),
            "rgb": torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + raw[..., 6:9]),
        }


def cover(points: torch.Tensor, detail: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Floor points (N, 2), detail Gaussians -> coverage (N, 1), colour (N, 3)."""
    xy = detail["xy"].flatten(0, 1)
    c, s = detail["angle"].flatten().cos(), detail["angle"].flatten().sin()
    d = points[:, None] - xy  # (N, M, 2)
    u = (c * d[..., 0] + s * d[..., 1]) / detail["scale"].flatten(0, 1)[:, 0]
    v = (-s * d[..., 0] + c * d[..., 1]) / detail["scale"].flatten(0, 1)[:, 1]
    w = detail["opacity"].flatten() * torch.exp(-0.5 * (u.pow(2) + v.pow(2)))  # (N, M)
    alpha = 1.0 - torch.exp(torch.log1p(-w.clamp(max=1.0 - 1e-6)).sum(-1, keepdim=True))
    colour = (w @ detail["rgb"].flatten(0, 1)) / w.sum(-1, keepdim=True).clamp_min(1e-8)
    return alpha, colour


class BackgroundDecoder(nn.Module):
    """Background slot -> floor discs and a fill colour."""

    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        # Inputs: PE(x), latent, azimuth.
        self.appearance = _mlp(3 + 6 * _PE_BANDS + z_dim + 2, hidden_dim)
        self.head = BackgroundHead(hidden_dim, z_dim + 2)
        self.detail = DetailDecoder(z_dim, hidden_dim)
        mu, spacing = floor_grid()
        self.register_buffer("floor_mu", mu, persistent=False)
        scale = torch.stack([spacing, spacing, torch.full_like(spacing, BACKGROUND_THICKNESS)], -1)
        self.register_buffer("floor_scale", scale, persistent=False)

    def forward(
        self,
        latents: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame,
        anchors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """latents (1 + B, C) (global, detail), attn (s, s) background attention,
        colours (3, s, s), SceneFrame, detail floor anchors (B, 2) -> Gaussians, "fill"."""
        latent = latents[0]
        mean = (attn * colours).sum((1, 2)) / attn.sum().clamp_min(1e-6)
        yaw = frame.yaw
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)]).to(latent.dtype)
        scene = torch.cat([latent, azimuth])  # (C + 2,)
        n = self.floor_mu.shape[0]
        pe = positional_encoding(self.floor_mu / _HALF, _PE_BANDS)
        feat = self.appearance(torch.cat([pe, scene.expand(n, -1)], -1))
        colour = self.head.colour(feat, mean)
        alpha, detail = cover(self.floor_mu[:, :2], self.detail(latents[1:], anchors, mean))
        colour["rgb"] = alpha * detail + (1.0 - alpha) * colour["rgb"]
        return {
            "mu": self.floor_mu,
            "scale": self.floor_scale,
            "quat": self.floor_mu.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(n, -1),
            "opacity": self.head.opacity(feat),
            **colour,
            "fill": self.head.fill(scene, mean),
        }
