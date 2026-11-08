"""Object decoder: object slot -> floor-standing region -> side^3 cells ->
one Gaussian per cell.

Each cell reads view 0 where it projects (the slot's attention-weighted
features and attention; region and attention detached), passes self-attention
among cells, cross-attention to the slot and FFN. Cells over background
pixels are pruned (visual hull). Scene frame, lengths in units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import read_view0

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest region half width / height (units of r_max).
OBJECT_WIDTH = 1.5
OBJECT_HEIGHT = 1.2
# Initial region size (share of its range).
EXTENT_INIT = 0.9
# Height of regions and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Read features are divided by max(attention, this).
MIN_ATTN = 0.3
# Frequency bands of the cells' position encoding.
CELL_PE_BANDS = 4
# Gaussian scale range and max offset, in cell half sizes.
CELL_SCALE = (0.25, 4.0)
CELL_REACH = 1.5


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


class RegionHead(nn.Module):
    """Slot latent -> region shift (along / across view 0's ray) and half sizes."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.mlp = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 5))
        self.register_buffer("shift_bound", torch.tensor([0.5 * OBJECT_REGION, 0.5 * r_max]), persistent=False)
        self.register_buffer("size_lo", torch.tensor([r_min, r_min, r_min]), persistent=False)
        self.register_buffer("size_hi", torch.tensor(
            [OBJECT_WIDTH * r_max, OBJECT_WIDTH * r_max, OBJECT_HEIGHT * r_max]), persistent=False)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[2:] = _logit(EXTENT_INIT)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> centre, half sizes (K, 3)."""
        raw = self.mlp(slots)
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        t = self.shift_bound * torch.tanh(raw[:, :2])
        xy = anchor[:, :2] + t[:, :1] * ray_xy + t[:, 1:] * across
        extent = self.size_lo + (self.size_hi - self.size_lo) * torch.sigmoid(raw[:, 2:])
        return torch.cat([xy, MIN_HEIGHT + extent[:, 2:]], -1), extent


class ObjectHead(nn.Module):
    """Cell feature -> raw Gaussian parameters: offset 3, scale 3, quat 4,
    opacity 1, rgb 3, sh1 9."""

    def __init__(self, d_model: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, 23))
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            self.project[-1].weight[3:].mul_(0.1)  # all but the offset start small
            bias = self.project[-1].bias
            bias.zero_()
            bias[3:6] = _logit(-lo / (hi - lo))  # scale = half a cell at first
            bias[6] = 1.0  # identity quaternion (wxyz)
            self.project[-1].weight[14:].zero_()  # view-independent at first

    def forward(
        self,
        feat: torch.Tensor,
        cells: torch.Tensor,
        cell_half: torch.Tensor,
        attn: torch.Tensor,
        keep: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), cells (K, G, 3), cell half sizes (K, 3), attention,
        keep (K, G) -> Gaussians (K, G, ...)."""
        raw = self.project(feat)
        half = cell_half[:, None]
        mu = cells + CELL_REACH * half * torch.tanh(raw[..., 0:3])
        return {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": half * _log_uniform(raw[..., 3:6], *CELL_SCALE),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(torch.logit(attn.clamp(0.02, 0.98)) + raw[..., 10]) * keep,
            "rgb": torch.sigmoid(raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class DecoderLayer(nn.Module):
    """Self-attention among the cells (occlusion, depth, parts),
    cross-attention to the slot tokens, FFN; pre-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q (K, G, d), slot tokens k, v (K, M, d) -> (K, G, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        return q + self.ffn(self.norm_ff(q))


class ObjectDecoder(nn.Module):
    """Object slots -> regions and Gaussians, scene frame."""

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
        """gaussians_per_slot: side**3 cells; hidden_dim: the attention width."""
        super().__init__()
        d_model = int(hidden_dim)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        side = round(gaussians_per_slot ** (1.0 / 3.0))
        assert side**3 == gaussians_per_slot, "gaussians_per_slot must be a cube"
        self.side = side
        self.M = int(n_tokens)
        self.d = d_model

        # Cell centres in the region, in [-1, 1]^3.
        lin = (torch.arange(side) + 0.5) / side * 2.0 - 1.0
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("cell_grid", grid, persistent=False)
        # Query: read feature and attention + encoded position in the region.
        self.register_buffer("cell_pe", positional_encoding(grid, CELL_PE_BANDS), persistent=False)
        self.cell_in = nn.Linear(z_dim + 1 + self.cell_pe.shape[1], d_model)
        self.k_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.v_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d_model)

    @staticmethod
    def _read_view0(cells: torch.Tensor, view0: dict):
        """Cells (K, G, 3) -> read features (K, G, C), attention (K, G), keep (K, G)."""
        attn, features = view0["attn"], view0["features"]
        K, C = attn.shape[0], features.shape[0]
        a = attn[:, None]
        maps = torch.cat([a * features[None], a, view0["bg_attn"][None, None].expand(K, -1, -1, -1)], 1)
        out = read_view0(cells, maps, view0["frame"], view0["focal_ratio"])  # (K, G, C + 2)
        a = out[..., C]
        return out[..., :C] / a[..., None].clamp_min(MIN_ATTN), a, 1.0 - out[..., C + 1]

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: "features" (C, s, s),
        the slots' "attn" (K, s, s), the background's
        "bg_attn" (s, s), "frame", "focal_ratio" -> Gaussians (K, G, ...),
        region centre and half sizes (K, 3)."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        cells = centre[:, None] + extent[:, None] * self.cell_grid
        # Detached: reading view 0 cannot move the region or regroup.
        view0 = {**view0, "attn": view0["attn"].detach(), "bg_attn": view0["bg_attn"].detach()}
        feat, a, keep = self._read_view0(cells.detach(), view0)
        q = self.cell_in(torch.cat([feat, a[..., None], self.cell_pe.expand(K, -1, -1)], -1))
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, cells, extent / self.side, a, keep), centre, extent
