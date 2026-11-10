"""Object decoder: object slot -> fixed floor-standing box -> side^3 cells ->
one Gaussian per cell.

The box sits at the slot's attention anchor, shifted along view 0's ray by
the slot. Cells (encoded position) attend among themselves and to the slot.
Each Gaussian's opacity is the slot's view-0 attention where its cell
projects times a learned occupancy: attention gives the silhouette, the slot
the depth. Colour is view 0's there plus a residual. Scene frame, units of R.
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
# Box half sizes.
BOX_HALF = (1.5 * OBJECT_RADIUS[1], 1.5 * OBJECT_RADIUS[1], 1.2 * OBJECT_RADIUS[1])
# Height of boxes and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Gaussian scale range and max offset, in cell half sizes.
CELL_SCALE = (0.25, 4.0)
CELL_REACH = 1.5
CELL_PE_BANDS = 4


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


class DecoderLayer(nn.Module):
    """Self-attention among the cells, cross-attention to the slot tokens,
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
        """gaussians_per_slot: side**3 cells; hidden_dim: the attention width."""
        super().__init__()
        d = int(hidden_dim)
        side = round(gaussians_per_slot ** (1.0 / 3.0))
        assert side**3 == gaussians_per_slot, "gaussians_per_slot must be a cube"
        self.side, self.M, self.d = side, int(n_tokens), d

        lin = (torch.arange(side) + 0.5) / side * 2.0 - 1.0
        grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        self.register_buffer("cell_grid", grid, persistent=False)
        self.register_buffer("cell_pe", positional_encoding(grid, CELL_PE_BANDS), persistent=False)
        self.register_buffer("box_half", torch.tensor(BOX_HALF), persistent=False)
        self.cell_in = nn.Linear(self.cell_pe.shape[1], d)
        self.k_proj = nn.Linear(z_dim, n_tokens * d)
        self.v_proj = nn.Linear(z_dim, n_tokens * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads) for _ in range(n_layers))
        # Box shift along / across view 0's ray.
        self.shift = nn.Linear(z_dim, 2)
        self.register_buffer("shift_bound", torch.tensor([0.5 * OBJECT_REGION, 0.5 * OBJECT_RADIUS[1]]), persistent=False)
        # Offset 3, scale 3, quat 4, occupancy 1, rgb 3, sh1 9.
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, 23))
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            self.shift.weight.zero_()
            self.shift.bias.zero_()
            out = self.head[-1]
            out.weight[3:].mul_(0.1)
            out.weight[14:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[3:6] = _logit(-lo / (hi - lo))  # scale = half a cell
            out.bias[6] = 1.0  # identity quaternion (wxyz)
            out.bias[10] = _logit(0.9)  # occupied: attention sets the opacity

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: the slots' "attn" (K, s, s),
        "colours" (3, s, s), "frame", "focal_ratio" -> Gaussians (K, G, ...),
        box centres (K, 3)."""
        K = slots.shape[0]
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        t = self.shift_bound * torch.tanh(self.shift(slots))
        xy = anchor[:, :2] + t[:, :1] * ray_xy + t[:, 1:] * across
        centre = torch.cat([xy, (MIN_HEIGHT + self.box_half[2]).expand(K, 1)], -1)
        cells = centre[:, None] + self.box_half * self.cell_grid

        # Not detached: reconstruction trains the attention through the opacity.
        attn, colours = view0["attn"], view0["colours"]
        maps = torch.cat([attn[:, None], colours[None].expand(K, -1, -1, -1)], 1)
        read = read_view0(cells.detach(), maps, view0["frame"], view0["focal_ratio"])  # (K, G, 4)
        a, c = read[..., 0], read[..., 1:]

        q = self.cell_in(self.cell_pe).expand(K, -1, -1)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        raw = self.head(q)

        half = self.box_half / self.side
        mu = cells + CELL_REACH * half * torch.tanh(raw[..., 0:3])
        gaussians = {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": half * _log_uniform(raw[..., 3:6], *CELL_SCALE),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": a.clamp(0.0, 1.0) * torch.sigmoid(raw[..., 10]),
            "rgb": torch.sigmoid(torch.logit(c.clamp(0.02, 0.98)) + raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }
        return gaussians, centre
