"""Object decoder: object slot -> region + one Gaussian per region cell.

The region stands on the floor around the attention anchor (shifted along
view 0's ray), sized from the slot latent. It holds side^3 cells. Each cell
reads view 0 where it projects (the slot's attention-weighted features, its
attention and colour; region and attention detached), passes n_layers of
self-attention among cells, cross-attention to tokens from the slot latent
and FFN, and emits a Gaussian near its centre whose opacity starts at the
sampled attention. Scene frame, lengths in units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import view0_grid

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest region half width / height (units of r_max).
OBJECT_WIDTH = 1.5
OBJECT_HEIGHT = 1.2
# Initial region size (share of its range).
EXTENT_INIT = 0.9
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


class ObjectHead(nn.Module):
    """The region from the slot latent; per cell raw parameters: offset 3,
    scale 3, quat 4, opacity 1, rgb 3, sh1 9."""

    def __init__(self, z_dim: int, d_model: int):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.project = _projection(d_model, 23)
        # Region shift along (view 0's ray, across it) on the floor, and half sizes.
        self.shift = nn.Linear(z_dim, 2)
        self.size = nn.Linear(z_dim, 3)
        self.register_buffer("shift_bound", torch.tensor([0.5 * OBJECT_REGION, 0.5 * r_max]), persistent=False)
        self.register_buffer("size_lo", torch.tensor([r_min, r_min, r_min]), persistent=False)
        self.register_buffer("size_hi", torch.tensor(
            [OBJECT_WIDTH * r_max, OBJECT_WIDTH * r_max, OBJECT_HEIGHT * r_max]), persistent=False)
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            bias = self.project[-1].bias
            bias[3:6] = _logit(-lo / (hi - lo))  # scale = half a cell at first
            bias[6] = 1.0  # identity quaternion (wxyz)
            self.project[-1].weight[14:].zero_()  # view-independent at first
            self.shift.weight.zero_()
            self.shift.bias.zero_()
            self.size.weight.zero_()
            self.size.bias.fill_(_logit(EXTENT_INIT))

    def region(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> region centre, half sizes (K, 3).
        The region stands on the floor: z from 0 to twice its half height."""
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        t = self.shift_bound * torch.tanh(self.shift(slots))
        xy = anchor[:, :2] + t[:, :1] * ray_xy + t[:, 1:] * across
        extent = self.size_lo + (self.size_hi - self.size_lo) * torch.sigmoid(self.size(slots))
        return torch.cat([xy, extent[:, 2:]], -1), extent

    def forward(
        self,
        feat: torch.Tensor,
        cells: torch.Tensor,
        cell_half: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), cell centres (K, G, 3), cell half sizes (K, 3), view-0
        attention and colours sampled at the cells (K, G), (K, G, 3) -> one
        Gaussian per cell (K, G, ...); opacity starts at the attention."""
        raw = self.project(feat)
        half = cell_half[:, None]
        return {
            "mu": cells + CELL_REACH * half * torch.tanh(raw[..., 0:3]),
            "scale": half * _log_uniform(raw[..., 3:6], *CELL_SCALE),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(torch.logit(attn.clamp(0.02, 0.98)) + raw[..., 10]),
            "rgb": torch.sigmoid(torch.logit(colours.clamp(0.02, 0.98)) + raw[..., 11:14]),
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
        # Cell query: its own embedding + view-0 feature, attention, colour, position.
        self.cell_embed = nn.Parameter(torch.randn(len(grid), d_model) * 0.02)
        self.cell_in = nn.Linear(z_dim + 7, d_model)
        self.k_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.v_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.head = ObjectHead(z_dim, d_model)

    @property
    def extent_range(self) -> tuple[float, float]:
        """xy half width range (units of R)."""
        return float(self.head.size_lo[0]), float(self.head.size_hi[0])

    @staticmethod
    def _sample(cells: torch.Tensor, view0: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cells (K, G, 3) -> where each projects in view 0: the slot's
        features (attention-weighted, K, G, C), attention (K, G), colours (K, G, 3)."""
        grid, front = view0_grid(cells, view0["frame"], view0["focal_ratio"])
        attn, features, colours = view0["attn"], view0["features"], view0["colours"]
        K, C = attn.shape[0], features.shape[0]
        maps = torch.cat(
            [attn[:, None] * features[None], attn[:, None], colours[None].expand(K, -1, -1, -1)], 1
        )
        out = F.grid_sample(maps, grid[:, None], mode="bilinear", align_corners=False)[:, :, 0]
        out = out.transpose(1, 2) * front[..., None]  # (K, G, C + 4)
        a = out[..., C]
        return out[..., :C] / a[..., None].clamp_min(1e-4), a, out[..., C + 1 :]

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: "features" (C, s, s),
        "colours" (3, s, s), the slots' "attn" (K, s, s), "frame",
        "focal_ratio" -> Gaussians (K, G, ...), region centre and half sizes (K, 3)."""
        K = slots.shape[0]
        centre, extent = self.head.region(slots, anchor, ray)
        cells = centre[:, None] + extent[:, None] * self.cell_grid
        # Detached: reading view 0 cannot move the region or regroup.
        view0 = {**view0, "attn": view0["attn"].detach()}
        feat, a, colours = self._sample(cells.detach(), view0)
        position = self.cell_grid.expand(K, -1, -1)
        q = self.cell_embed + self.cell_in(torch.cat([feat, a[..., None], colours, position], -1))
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, cells, extent / self.side, a, colours), centre, extent
