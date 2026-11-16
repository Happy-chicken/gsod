"""Object decoder: visible surface (on token rays, opacity = attention) and
completion (fixed queries in the box, x attention where they project). Units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import SceneFrame, project_and_sample

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
# Visible surface: tokens per slot, uv PE bands.
VISIBLE_TOKENS = 1024
VISIBLE_PE_BANDS = 6


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
    """Slot latent -> box shift (along / across view 0's ray) and half sizes
    r_min + r_max softplus(.) (no upper limit)."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.mlp = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 5))
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
        self,
        feat: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), box centre / half sizes / colour (K, 3) -> Gaussians (K, G, ...)."""
        raw = self.project(feat)
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
            "rgb": torch.sigmoid(torch.logit(colour.clamp(0.02, 0.98))[:, None] + raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class VisibleSurface(nn.Module):
    """[slot, PE(uv)] -> a Gaussian on that token's view-0 ray; opacity = attention."""

    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 2 + 4 * VISIBLE_PE_BANDS, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 13),  # depth 1, rgb 3, sh1 9
        )
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight[4:].zero_()  # view-independent at first
            out.bias.zero_()

    def forward(
        self,
        slots: torch.Tensor,
        uv: torch.Tensor,
        dirs: torch.Tensor,
        attn: torch.Tensor,
        origin: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
        token_width: float,
    ) -> dict[str, torch.Tensor]:
        """slots (K, C); tokens' uv, rays, attention (K, T, .); box, colour -> Gaussians (K, T, ...)."""
        pe = positional_encoding(uv.flatten(0, 1), VISIBLE_PE_BANDS).view(*uv.shape[:2], -1)
        raw = self.mlp(torch.cat([slots[:, None].expand(-1, uv.shape[1], -1), pe], -1))
        # Depth: box centre's on this ray +- the box half width.
        t0 = ((centre - origin)[:, None] * dirs).sum(-1)
        depth = (t0 + extent[:, None, :2].mean(-1) * torch.tanh(raw[..., 0])).clamp_min(1e-3)
        mu = origin + depth[..., None] * dirs
        mu = torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1)
        size = 0.5 * depth * token_width  # covers its own token only
        quat = torch.zeros_like(mu[..., :1]).expand(-1, -1, 4).clone()
        quat[..., 0] = 1.0
        return {
            "mu": mu,
            "scale": size[..., None].expand(-1, -1, 3),
            "quat": quat,
            "opacity": attn,
            "rgb": torch.sigmoid(torch.logit(colour.clamp(0.02, 0.98))[:, None] + raw[..., 1:4]),
            "sh1": raw[..., 4:13].unflatten(-1, (3, 3)),
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
        self.visible = VisibleSurface(z_dim, d)

    def forward(
        self,
        slots: torch.Tensor,
        anchor: torch.Tensor,
        ray: torch.Tensor,
        colour: torch.Tensor,
        uv: torch.Tensor,
        dirs: torch.Tensor,
        attn: torch.Tensor,
        attn_maps: torch.Tensor,
        frame: SceneFrame,
        focal_ratio: tuple[float, float],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """slots (K, C); anchor, ray, colour (K, 3); tokens' uv, rays, attention (K, T, .);
        attention maps (K, 1, s, s) -> Gaussians (K, T + G, ...) (visible, completion), extent."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        q = self.queries.expand(K, -1, -1)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        completion = self.head(q, centre, extent, colour)
        support = project_and_sample(completion["mu"], attn_maps, frame, focal_ratio)[..., 0]
        completion["opacity"] = completion["opacity"] * support
        token_width = 1.0 / (focal_ratio[0] * attn_maps.shape[-1])
        visible = self.visible(slots, uv, dirs, attn, frame.origin, centre, extent, colour, token_width)
        return {key: torch.cat([visible[key], completion[key]], 1) for key in completion}, extent
