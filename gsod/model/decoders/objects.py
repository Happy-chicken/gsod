"""Object decoder: slot -> box at the attention anchor -> G Gaussians in it.
Units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Initial box half sizes x, y, z (only an initial value; no bounds). Boxes
# rest on the floor: height [0, 2 * e_z].
EXTENT_INIT = (0.275, 0.275, 0.18)
# Initial height of the plane the anchors' rays are cut at (learned).
ANCHOR_HEIGHT_INIT = 0.1
# Box shift along view 0's ray: at most this fraction of the camera distance.
ALONG_RAY = 0.5
# Height of boxes and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Initial Gaussian scale, as a fraction of its box's half sizes.
SCALE_INIT = 1.0 / 8.0
# Spread of the learned queries (how far apart Gaussians start).
QUERY_STD = 2.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


class RegionHead(nn.Module):
    """Slot latent -> box half sizes softplus(.) (no bounds) and a shift of the
    anchor along view 0's ray (up to ALONG_RAY of the camera distance) and
    across it (up to the box's half size); the box rests on the floor."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 5))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            for i, size in enumerate(EXTENT_INIT):
                self.mlp[-1].bias[2 + i] = math.log(math.expm1(size))

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, view: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, view = anchor - camera (K, 3) -> centre, half sizes (K, 3)."""
        raw = self.mlp(slots)
        extent = F.softplus(raw[:, 2:])
        ray_xy = F.normalize(view[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        along = ALONG_RAY * view[:, :2].norm(dim=-1, keepdim=True) * torch.tanh(raw[:, :1])
        side = extent[:, :2].mean(-1, keepdim=True) * torch.tanh(raw[:, 1:2])
        xy = anchor[:, :2] + along * ray_xy + side * across
        return torch.cat([xy, extent[:, 2:]], -1), extent


class ObjectHead(nn.Module):
    """Gaussian feature -> one Gaussian; its scale is a fraction of its box's
    half sizes."""

    def __init__(self, d_model: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, 23))
        with torch.no_grad():
            out = self.project[-1]
            out.weight[3:].mul_(0.1)
            out.weight[14:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[3:6] = _logit(SCALE_INIT)
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
            "scale": extent[:, None] * torch.sigmoid(raw[..., 3:6]),
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
        self.anchor_height_raw = nn.Parameter(torch.tensor(math.log(math.expm1(ANCHOR_HEIGHT_INIT))))

    def anchor_height(self) -> torch.Tensor:
        """Height (> 0) of the plane the anchors' token rays are cut at."""
        return F.softplus(self.anchor_height_raw)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, view: torch.Tensor, colour: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """slots (K, C); anchor, view = anchor - camera, colour (K, 3) -> Gaussians (K, G, ...), half sizes."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, view)
        q = self.queries.expand(K, -1, -1)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, centre, extent, colour), extent
