"""Object decoder: object slot -> region + one Gaussian per region cell.

The region (RegionHead, an MLP on the slot latent) stands just above the
floor around the attention anchor, shifted along view 0's ray and sized. It
holds side^3 cells. Each cell reads view 0 where it projects (region and
attention detached): the slot's attention-weighted features (dim where the
slot holds little), its attention, and the background's. Its query is its
own embedding plus the features and attention. It passes n_layers of
self-attention among cells, cross-attention to tokens from the slot latent
and FFN, and emits a Gaussian near its centre whose opacity starts at the
attention read there. A cell over a background pixel is empty: view 0 sees the
background through it (visual hull), so its opacity is scaled by
1 - background attention. Scene frame, lengths in units of R.
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
# Regions and their Gaussians stay this high above the floor.
MIN_HEIGHT = 0.01
# Features read from view 0 are divided by max(attention, this),
# so a cell sees little of pixels its slot hardly holds.
MIN_ATTN = 0.3
# Cell Gaussians: scale range in units of the cell's half size, and how far
# (in cell half sizes) a Gaussian may move from its cell centre.
CELL_SCALE = (0.25, 4.0)
CELL_REACH = 1.5


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def _projection(d_model: int, n_params: int) -> nn.Sequential:
    """Feature -> raw parameters; all but the position rows start small."""
    mlp = nn.Sequential(
        nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, n_params)
    )
    with torch.no_grad():
        mlp[-1].weight[3:].mul_(0.1)
        mlp[-1].bias.zero_()
    return mlp


class RegionHead(nn.Module):
    """Slot latent -> the slot's region: a small MLP gives the shift along
    view 0's ray and across it (on the floor) and the half sizes (x, y, z).
    The output layer starts at zero: every region starts at the anchor, at
    EXTENT_INIT of its size range."""

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
        """slots (K, C), anchor, ray (K, 3) -> region centre, half sizes (K, 3).
        The region stands on the floor: z from MIN_HEIGHT up twice its half height."""
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
        self.project = _projection(d_model, 23)
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            bias = self.project[-1].bias
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
        """feat (K, G, d), cell centres (K, G, 3), cell half sizes (K, 3), view-0
        attention and visual-hull factor read at the cells (K, G) -> one
        Gaussian per cell (K, G, ...)."""
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
        # Cell query: its own embedding (its identity and position; large enough
        # to tell apart cells on one view-0 ray, which read the same pixels)
        # + the view-0 feature and attention read there.
        self.cell_embed = nn.Parameter(torch.randn(len(grid), d_model))
        self.cell_in = nn.Linear(z_dim + 1, d_model)
        self.k_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.v_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d_model)

    @staticmethod
    def _read_view0(cells: torch.Tensor, view0: dict):
        """Cells (K, G, 3) -> where each projects in view 0: the slot's
        attention-weighted features (K, G, C), its attention (K, G) and the
        visual-hull factor 1 - background attention (K, G)."""
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
        q = self.cell_embed + self.cell_in(torch.cat([feat, a[..., None]], -1))
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, cells, extent / self.side, a, keep), centre, extent
