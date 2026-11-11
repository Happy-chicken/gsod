"""Object decoder: object slot -> floor-standing box -> side^3 cells ->
one Gaussian per cell.

The box (RegionHead) sits at the slot's attention anchor, shifted along
view 0's ray and sized by the slot; its first axis follows that ray, so a
column of cells along it is (nearly) one view-0 ray. Cells (encoded
position) attend among themselves and to the slot. Each column is occupied
on one interval: every cell scores being its front and its back (softmax
along the column), occupancy = P(front <= cell) * P(back >= cell). Opacity
= the slot's view-0 attention where the cell projects x occupancy (x above
the floor): attention gives the silhouette, the interval the depth.
Colour is view 0's there plus a residual. Scene frame, units of R.
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
# Largest box half width / height (units of r_max).
OBJECT_WIDTH = 1.5
OBJECT_HEIGHT = 1.2
# Initial box size (share of its range).
EXTENT_INIT = 0.9
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


class RegionHead(nn.Module):
    """Slot latent -> box shift (along / across view 0's ray) and half sizes."""

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
    """Cell feature -> raw Gaussian parameters (offset 3, scale 3, quat 4,
    front 1, back 1, rgb 3, sh1 9) -> one Gaussian per cell."""

    def __init__(self, d_model: int, side: int):
        super().__init__()
        self.side = side
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, 24))
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            out = self.project[-1]
            out.weight[3:].mul_(0.1)
            out.weight[15:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[3:6] = _logit(-lo / (hi - lo))  # scale = half a cell
            out.bias[6] = 1.0  # identity quaternion (wxyz)

    def occupancy(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        """front, back scores (K, G) -> occupancy (K, G): one interval per
        column (cells along the box's first axis)."""
        K, S = front.shape[0], self.side
        p_front = front.view(K, S, S * S).softmax(1)
        p_back = back.view(K, S, S * S).softmax(1)
        started = p_front.cumsum(1)  # P(front <= cell)
        ended = p_back.cumsum(1) - p_back  # P(back < cell)
        return (started * (1.0 - ended)).view(K, -1)

    def forward(
        self,
        feat: torch.Tensor,
        cells: torch.Tensor,
        axes: torch.Tensor,
        cell_half: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), cells (K, G, 3), box axes (K, 3, 3) (rows), cell half
        sizes (K, 3) along them, view-0 attention (K, G) and colours (K, G, 3)
        at the cells -> Gaussians (K, G, ...)."""
        raw = self.project(feat)
        half = cell_half[:, None]
        offset = (CELL_REACH * half * torch.tanh(raw[..., 0:3]))[..., None, :] @ axes[:, None]
        mu = cells + offset[..., 0, :]
        above = torch.sigmoid((cells[..., 2] - MIN_HEIGHT) / half.amin(-1))  # the floor
        return {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": half * _log_uniform(raw[..., 3:6], *CELL_SCALE),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": attn.clamp(0.0, 1.0) * self.occupancy(raw[..., 10], raw[..., 11]) * above,
            "rgb": torch.sigmoid(torch.logit(colours.clamp(0.02, 0.98)) + raw[..., 12:15]),
            "sh1": raw[..., 15:24].unflatten(-1, (3, 3)),
        }


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
        self.cell_in = nn.Linear(self.cell_pe.shape[1], d)
        self.k_proj = nn.Linear(z_dim, n_tokens * d)
        self.v_proj = nn.Linear(z_dim, n_tokens * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads) for _ in range(n_layers))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d, side)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: the slots' "attn" (K, s, s),
        "colours" (3, s, s), "frame", "focal_ratio" -> Gaussians (K, G, ...),
        box centres and half sizes (K, 3)."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        # Box axes: along view 0's ray, across it (horizontal), and up-ish.
        along = F.normalize(ray, dim=-1)
        up = torch.zeros_like(along)
        up[:, 2] = 1.0
        across = F.normalize(torch.cross(up, along, dim=-1), dim=-1)
        axes = torch.stack([along, across, torch.cross(along, across, dim=-1)], 1)  # (K, 3, 3)
        cells = centre[:, None] + (extent[:, None] * self.cell_grid) @ axes

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
        return self.head(q, cells, axes, extent / self.side, a, c), centre, extent
