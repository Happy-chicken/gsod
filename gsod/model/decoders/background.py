"""Background decoder: global slot -> floor anchors with flat Gaussians;
detail slots -> flat Gaussians on the floor under the tokens they own (just
above the global group). No pixel is read. World-aligned.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gsod.model.decoders.objects import OBJECT_REGION, positional_encoding

# Anchors: inner grid over the object region, outer grid to +-_HALF.
_INNER_SIDE = 10
_OUTER_SIDE = 8
_HALF = 3.0
_PE_BANDS = 6
ANCHOR_DIM = 32
GLOBAL_PER_ANCHOR = 4
DETAIL_TOKENS = 256
# Scale ranges (global: anchor spacing; detail: R), thickness, detail lift (R).
GLOBAL_SCALE = (1.0 / 4.0, 1.5)
DETAIL_SCALE = (0.005, 0.05)
THICKNESS = 0.005
DETAIL_LIFT = 1e-3
GLOBAL_OPACITY_INIT = 0.95
DETAIL_OPACITY_INIT = 0.5
# Per Gaussian: offset 2, log scale 2, angle 1, opacity 1, rgb 3, sh1 9.
_ATTRS = 18


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def anchor_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor centres (A, 2) and spacing (A,)."""
    def grid(half: float, side: int):
        lin = torch.linspace(-half, half, side)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x.flatten(), y.flatten()], -1), 2.0 * half / (side - 1)

    inner, s_in = grid(OBJECT_REGION, _INNER_SIDE)
    outer, s_out = grid(_HALF, _OUTER_SIDE)
    outer = outer[outer.abs().amax(-1) > OBJECT_REGION + 0.5 * s_in]
    spacing = torch.cat([torch.full((len(inner),), s_in), torch.full((len(outer),), s_out)])
    return torch.cat([inner, outer]), spacing


class GaussianGroup(nn.Module):
    """[anchor feature, PE, latent] -> n flat Gaussians per anchor."""

    def __init__(self, d_in: int, hidden: int, n: int, scale: tuple[float, float], opacity: float):
        super().__init__()
        self.n, self.scale = n, scale
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, n * _ATTRS),
        )
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight.view(n, _ATTRS, -1)[:, 9:].zero_()  # view-independent at first
            bias = out.bias.view(n, _ATTRS)
            bias.zero_()
            side = math.ceil(math.sqrt(n))  # start on a sub-grid about the anchor
            for i in range(n):
                cell = torch.tensor([i % side, i // side], dtype=bias.dtype)
                bias[i, 0:2] = torch.atanh((cell + 0.5) / side - 0.5)
            bias[:, 5] = _logit(opacity)

    def forward(
        self, inp: torch.Tensor, centre: torch.Tensor, spacing: torch.Tensor, mean: torch.Tensor, z: float
    ) -> dict[str, torch.Tensor]:
        """inp (A, d_in), centres (A, 2), spacing (A,), mean colour (3,) -> Gaussians (A * n, ...)."""
        raw = self.mlp(inp).view(-1, self.n, _ATTRS)
        s = spacing[:, None, None]
        xy = centre[:, None] + s * torch.tanh(raw[..., 0:2])
        lo, hi = (math.log(v) for v in self.scale)
        scale_xy = s * torch.exp(lo + (hi - lo) * torch.sigmoid(raw[..., 2:4]))
        half = 0.5 * math.pi * torch.tanh(raw[..., 4])
        zero = torch.zeros_like(half)
        out = {
            "mu": torch.cat([xy, torch.full_like(xy[..., :1], z)], -1),
            "scale": torch.cat([scale_xy, torch.full_like(scale_xy[..., :1], THICKNESS)], -1),
            "quat": torch.stack([half.cos(), zero, zero, half.sin()], -1),
            "opacity": torch.sigmoid(raw[..., 5]),
            "rgb": torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + raw[..., 6:9]),
            "sh1": raw[..., 9:18].unflatten(-1, (3, 3)),
        }
        return {k: v.flatten(0, 1) for k, v in out.items()}


class DetailTokens(nn.Module):
    """Detail slots + their tokens' floor points -> flat Gaussians there."""

    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.query = nn.Linear(2 + 4 * _PE_BANDS, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + z_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 16),  # log scale 2, angle 1, opacity 1, rgb 3, sh1 9
        )
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight[7:].zero_()
            out.bias.zero_()
            out.bias[3] = _logit(DETAIL_OPACITY_INIT)

    def forward(
        self, latents: torch.Tensor, xy: torch.Tensor, uv: torch.Tensor, mean: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """latents (B, C), floor xy, uv (B, T, 2), mean colour (3,) -> Gaussians (B * T, ...)."""
        q = self.query(positional_encoding(uv.flatten(0, 1), _PE_BANDS)).view(*uv.shape[:2], -1)
        raw = self.mlp(torch.cat([q, latents[:, None].expand(-1, q.shape[1], -1)], -1))
        lo, hi = (math.log(v) for v in DETAIL_SCALE)
        scale_xy = torch.exp(lo + (hi - lo) * torch.sigmoid(raw[..., 0:2]))
        half = 0.5 * math.pi * torch.tanh(raw[..., 2])
        zero = torch.zeros_like(half)
        out = {
            "mu": torch.cat([xy, torch.full_like(xy[..., :1], DETAIL_LIFT)], -1),
            "scale": torch.cat([scale_xy, torch.full_like(scale_xy[..., :1], THICKNESS)], -1),
            "quat": torch.stack([half.cos(), zero, zero, half.sin()], -1),
            "opacity": torch.sigmoid(raw[..., 3]),
            "rgb": torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + raw[..., 4:7]),
            "sh1": raw[..., 7:16].unflatten(-1, (3, 3)),
        }
        return {k: v.flatten(0, 1) for k, v in out.items()}


class BackgroundDecoder(nn.Module):
    """Background slots -> anchored floor Gaussians and a fill colour."""

    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        centre, spacing = anchor_grid()
        self.register_buffer("centre", centre, persistent=False)
        self.register_buffer("spacing", spacing, persistent=False)
        self.feature = nn.Parameter(0.1 * torch.randn(len(centre), ANCHOR_DIM))
        d_pe = 2 + 4 * _PE_BANDS
        self.global_group = GaussianGroup(
            ANCHOR_DIM + d_pe + z_dim + 2, hidden_dim, GLOBAL_PER_ANCHOR, GLOBAL_SCALE, GLOBAL_OPACITY_INIT
        )
        self.detail = DetailTokens(z_dim, hidden_dim)
        self.fill_out = nn.Linear(z_dim + 2, 3)
        with torch.no_grad():
            self.fill_out.weight.zero_()  # the mean colour at first
            self.fill_out.bias.zero_()

    def forward(
        self,
        latents: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame,
        detail_xy: torch.Tensor,
        detail_uv: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """latents (1 + B, C) (global, detail), attn (s, s) background attention,
        colours (3, s, s), SceneFrame, detail tokens' floor xy and uv (B, T, 2)
        -> Gaussians, "fill"."""
        mean = (attn * colours).sum((1, 2)) / attn.sum().clamp_min(1e-6)
        yaw = frame.yaw
        azimuth = torch.stack([torch.cos(yaw), torch.sin(yaw)]).to(latents.dtype)
        scene = torch.cat([latents[0], azimuth])
        n = len(self.centre)
        base = torch.cat([self.feature, positional_encoding(self.centre / _HALF, _PE_BANDS)], -1)
        glob = self.global_group(torch.cat([base, scene.expand(n, -1)], -1), self.centre, self.spacing, mean, 0.0)
        detail = self.detail(latents[1:], detail_xy, detail_uv, mean)

        out = {k: torch.cat([glob[k], detail[k]]) for k in glob}
        out["fill"] = torch.sigmoid(torch.logit(mean.clamp(0.02, 0.98)) + self.fill_out(scene))
        return out
