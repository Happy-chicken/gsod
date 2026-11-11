"""Object decoder: object slot -> floor-standing box -> side^3 cubes ->
k Gaussians per cube.

The box (RegionHead) sits at the slot's attention anchor, shifted along
view 0's ray and sized by the slot; it is split into side^3 cubes along the
scene axes. One token per cube (encoded position) attends to the other
cubes and to the slot, then emits the cube's occupancy and k Gaussians that
move freely inside it. A Gaussian's opacity is the slot's view-0 attention
where it projects x its cube's occupancy x its own factor: attention gives
the silhouette, occupancy where the object is, the Gaussians its shape.
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
# Gaussian scale range, in cube half sizes; Gaussians stay inside their cube.
GAUSS_SCALE = (0.1, 1.0)
# Initial cube occupancy and Gaussian opacity factor.
OCCUPANCY_INIT = 0.9
OPACITY_INIT = 0.7
# Per Gaussian: offset 3, scale 3, quat 4, opacity 1, rgb 3, sh1 9.
GAUSS_PARAMS = 23
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
    """Cube feature -> occupancy and k Gaussians inside the cube."""

    def __init__(self, d_model: int, per_cube: int):
        super().__init__()
        sub = round(per_cube ** (1.0 / 3.0))
        assert sub**3 == per_cube, "gaussians_per_cell must be a cube"
        self.k = per_cube
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, 1 + per_cube * GAUSS_PARAMS)
        )
        lin = (torch.arange(sub) + 0.5) / sub * 2.0 - 1.0  # sub-grid of the cube
        start = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
        lo, hi = (math.log(s) for s in GAUSS_SCALE)
        with torch.no_grad():
            out = self.project[-1]
            out.weight.mul_(0.1)
            out.bias.zero_()
            out.bias[0] = _logit(OCCUPANCY_INIT)
            w = out.weight[1:].view(per_cube, GAUSS_PARAMS, d_model)
            w[:, 14:] = 0.0  # view-independent at first
            g = out.bias[1:].view(per_cube, GAUSS_PARAMS)
            g[:, 0:3] = torch.atanh(start)  # spread over the cube
            g[:, 3:6] = _logit((math.log(1.0 / sub) - lo) / (hi - lo))  # one sub-cell
            g[:, 6] = 1.0  # identity quaternion (wxyz)
            g[:, 10] = _logit(OPACITY_INIT)

    def forward(self, feat: torch.Tensor, centres: torch.Tensor, half: torch.Tensor, read) -> dict[str, torch.Tensor]:
        """feat (K, G, d), cube centres (K, G, 3), cube half sizes (K, 3),
        read: points (K, P, 3) -> view-0 attention (K, P), colours (K, P, 3)
        -> Gaussians (K, G * k, ...)."""
        K, G = feat.shape[:2]
        raw = self.project(feat)
        occupancy = torch.sigmoid(raw[..., 0])  # (K, G)
        p = raw[..., 1:].view(K, G, self.k, GAUSS_PARAMS)
        h = half[:, None, None]
        mu = centres[:, :, None] + h * torch.tanh(p[..., 0:3])
        mu = torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1)
        a, c = read(mu.detach().flatten(1, 2))
        a, c = a.view(K, G, self.k), c.view(K, G, self.k, 3)
        out = {
            "mu": mu,
            "scale": h * _log_uniform(p[..., 3:6], *GAUSS_SCALE),
            "quat": F.normalize(p[..., 6:10], dim=-1),
            "opacity": a.clamp(0.0, 1.0) * occupancy[..., None] * torch.sigmoid(p[..., 10]),
            "rgb": torch.sigmoid(torch.logit(c.clamp(0.02, 0.98)) + p[..., 11:14]),
            "sh1": p[..., 14:23].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(1, 2) for name, value in out.items()}


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
        cell_side: int,
        gaussians_per_cell: int,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        n_layers: int = 2,
    ):
        """cell_side^3 cubes, gaussians_per_cell Gaussians each; hidden_dim:
        the attention width."""
        super().__init__()
        d = int(hidden_dim)
        side = int(cell_side)
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
        self.head = ObjectHead(d, int(gaussians_per_cell))

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: the slots' "attn" (K, s, s),
        "colours" (3, s, s), "frame", "focal_ratio" -> Gaussians (K, G, ...),
        box centres and half sizes (K, 3)."""
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        cells = centre[:, None] + extent[:, None] * self.cell_grid

        # Not detached: reconstruction trains the attention through the opacity.
        maps = torch.cat([view0["attn"][:, None], view0["colours"][None].expand(K, -1, -1, -1)], 1)

        def read(points: torch.Tensor):
            out = read_view0(points, maps, view0["frame"], view0["focal_ratio"])  # (K, P, 4)
            return out[..., 0], out[..., 1:]

        q = self.cell_in(self.cell_pe).expand(K, -1, -1)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        return self.head(q, cells, extent / self.side, read), centre, extent
