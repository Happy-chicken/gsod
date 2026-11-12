"""Object decoder: slot -> floor-standing box -> G Gaussians free in it.

G learned queries attend among themselves and to the slot. Opacity is
normalised per slot (the draw factor scales the slot's layer); colour = the
colour at the slot's attention anchor + residual. Scene frame, units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest box half width (units of r_max); the half height has no upper limit.
OBJECT_WIDTH = 1.5
# Initial box half width (share of its range) and half height (units of r_max).
EXTENT_INIT = 0.9
HEIGHT_INIT = 1.2
# Height of boxes and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Gaussian scale range and initial scale (units of r_max).
SCALE_RANGE = (1.0 / 40.0, 1.0)
SCALE_INIT = 1.0 / 8.0
# Spread of the learned queries (how far apart Gaussians start).
QUERY_STD = 2.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


class RegionHead(nn.Module):
    """Slot latent -> box shift (along / across view 0's ray) and half sizes:
    width in [r_min, OBJECT_WIDTH r_max], height r_min + r_max softplus(.)."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.mlp = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 5))
        self.register_buffer("shift_bound", torch.tensor([0.5 * OBJECT_REGION, 0.5 * r_max]), persistent=False)
        self.r_min, self.r_max = r_min, r_max
        self.width_hi = OBJECT_WIDTH * r_max
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[2:4] = _logit(EXTENT_INIT)
            self.mlp[-1].bias[4] = math.log(math.expm1(HEIGHT_INIT - r_min / r_max))

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> centre, half sizes (K, 3)."""
        raw = self.mlp(slots)
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        t = self.shift_bound * torch.tanh(raw[:, :2])
        xy = anchor[:, :2] + t[:, :1] * ray_xy + t[:, 1:] * across
        width = self.r_min + (self.width_hi - self.r_min) * torch.sigmoid(raw[:, 2:4])
        height = self.r_min + self.r_max * F.softplus(raw[:, 4:5])
        extent = torch.cat([width, height], -1)
        return torch.cat([xy, MIN_HEIGHT + extent[:, 2:]], -1), extent


class ObjectHead(nn.Module):
    """Gaussian feature -> raw parameters (position 3, scale 3, quat 4,
    opacity 1, rgb 3, sh1 9) -> one Gaussian."""

    def __init__(self, d_model: int):
        super().__init__()
        r_max = OBJECT_RADIUS[1]
        self.scale_range = (SCALE_RANGE[0] * r_max, SCALE_RANGE[1] * r_max)
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, 23))
        lo, hi = (math.log(s) for s in self.scale_range)
        with torch.no_grad():
            out = self.project[-1]
            out.weight[3:].mul_(0.1)
            out.weight[14:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[3:6] = _logit((math.log(SCALE_INIT * r_max) - lo) / (hi - lo))
            out.bias[6] = 1.0  # identity quaternion (wxyz)

    def forward(
        self, feat: torch.Tensor, centre: torch.Tensor, extent: torch.Tensor, colour: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), box centre / half sizes / colour (K, 3) -> Gaussians (K, G, ...)."""
        raw = self.project(feat)
        mu = centre[:, None] + extent[:, None] * torch.tanh(raw[..., 0:3])
        mu = torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1)
        opacity = torch.sigmoid(raw[..., 10])
        opacity = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "mu": mu,
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": opacity,
            "rgb": torch.sigmoid(torch.logit(colour.clamp(0.02, 0.98))[:, None] + raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class DecoderLayer(nn.Module):
    """Self-attention among the Gaussians, cross-attention to the slot tokens,
    FFN; pre-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q (K, G, d), slot tokens k, v (K, M, d) -> (K, G, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        return q + self.ffn(self.norm_ff(q))


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians, scene frame."""

    def __init__(
        self,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        n_layers: int = 2,
    ):
        super().__init__()
        d = int(hidden_dim)
        self.M, self.d = int(n_tokens), d
        self.queries = nn.Parameter(torch.randn(int(gaussians_per_slot), d) * QUERY_STD)
        self.k_proj = nn.Linear(z_dim, n_tokens * d)
        self.v_proj = nn.Linear(z_dim, n_tokens * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads) for _ in range(n_layers))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, colour: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray, colour (K, 3) -> Gaussians, centres, half sizes."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        q = self.queries.expand(K, -1, -1)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, centre, extent, colour), centre, extent
