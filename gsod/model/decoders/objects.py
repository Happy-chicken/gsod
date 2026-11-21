"""Object decoder: slot -> box at the attention anchor -> G Gaussians in it.
Units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.layers import logit, mlp, shift_colour

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Initial box half sizes x, y, z (units of r_max); no upper limit.
EXTENT_INIT = (1.375, 1.375, 1.2)
# Height of boxes and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Gaussian scale range and initial scale (units of r_max).
SCALE_RANGE = (1.0 / 40.0, 1.0)
SCALE_INIT = 1.0 / 8.0
# Spread of the learned queries (how far apart Gaussians start).
QUERY_STD = 2.0
# Transformer: heads, slot tokens, layers; RegionHead MLP width.
N_HEADS, N_TOKENS, N_LAYERS = 4, 8, 2
REGION_HIDDEN = 64


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


class RegionHead(nn.Module):
    """Slot latent -> box shift (along / across view 0's ray) and half sizes
    r_min + r_max softplus(.) (no upper limit)."""

    def __init__(self, z_dim: int):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.mlp = mlp(z_dim, REGION_HIDDEN, 5)
        self.register_buffer("shift_bound", torch.tensor([0.5 * OBJECT_REGION, 0.5 * r_max]), persistent=False)
        self.r_min, self.r_max = r_min, r_max
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            for i, size in enumerate(EXTENT_INIT):
                self.mlp[-1].bias[2 + i] = math.log(math.expm1(size - r_min / r_max))

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> centre, half sizes (K, 3)."""
        raw = self.mlp(slots)
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        t = self.shift_bound * torch.tanh(raw[:, :2])
        xy = anchor[:, :2] + t[:, :1] * ray_xy + t[:, 1:] * across
        extent = self.r_min + self.r_max * F.softplus(raw[:, 2:])
        return torch.cat([xy, anchor[:, 2:]], -1), extent


class ObjectHead(nn.Module):
    """Gaussian feature -> one Gaussian."""

    def __init__(self, d_model: int):
        super().__init__()
        r_max = OBJECT_RADIUS[1]
        self.scale_range = (SCALE_RANGE[0] * r_max, SCALE_RANGE[1] * r_max)
        self.mlp = mlp(d_model, d_model, 23)
        lo, hi = (math.log(s) for s in self.scale_range)
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight[3:].mul_(0.1)
            out.weight[14:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[3:6] = logit((math.log(SCALE_INIT * r_max) - lo) / (hi - lo))
            out.bias[6] = 1.0  # identity quaternion (wxyz)

    def forward(
        self,
        feat: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), box centre / half sizes / colour (K, 3) -> Gaussians (K, G, ...)."""
        raw = self.mlp(feat)
        t = torch.tanh(raw[..., 0:3])
        xy = centre[:, None, :2] + extent[:, None, :2] * t[..., :2]
        # Height: over [centre - extent, centre + extent] cut at the floor.
        lo = (centre[:, 2] - extent[:, 2]).clamp_min(MIN_HEIGHT)[:, None, None]
        hi = (centre[:, 2] + extent[:, 2])[:, None, None]
        mu = torch.cat([xy, lo + (hi - lo) * 0.5 * (t[..., 2:] + 1.0)], -1)

        opacity = torch.sigmoid(raw[..., 10])
        opacity = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "mu": mu,
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": opacity,
            "rgb": shift_colour(colour[:, None], raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class DecoderLayer(nn.Module):
    """Self-attention among the Gaussians, cross-attention to the slot tokens,
    FFN; pre-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = mlp(d_model, 2 * d_model, d_model, act=nn.GELU)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q (K, G, d), slot tokens k, v (K, M, d) -> (K, G, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross_attn(self.norm_cross(q), k, v, need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians, scene frame."""

    def __init__(self, z_dim: int, gaussians_per_slot: int, hidden_dim: int):
        super().__init__()
        d = int(hidden_dim)
        self.M, self.d = N_TOKENS, d
        self.queries = nn.Parameter(torch.randn(int(gaussians_per_slot), d) * QUERY_STD)
        self.to_k = nn.Linear(z_dim, N_TOKENS * d)
        self.to_v = nn.Linear(z_dim, N_TOKENS * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, N_HEADS) for _ in range(N_LAYERS))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, colour: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """slots (K, C); anchor, ray, colour (K, 3) -> Gaussians (K, G, ...), half sizes."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        q = self.queries.expand(K, -1, -1)
        k = self.norm_kv(self.to_k(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.to_v(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, centre, extent, colour), extent
