"""Object decoder: object slot -> frustum grid along view 0's rays -> one
Gaussian per cell.

Per slot, S x S view-0 rays cover the slot's attention region (image box
from the attention's moments); each holds D cells over a depth range
centred near where the slot touches the floor (its lowest attended rays
meet z = 0), shifted and sized by the slot (RegionHead). One token per ray
(features, attention, colour read on it, encoded position, floor depth)
passes self-attention among rays, cross-attention to the slot and FFN, and
predicts the ray's interval [t_front, t_back] plus each cell's Gaussian.
Cells are occupied inside the interval (and above the floor) only; opacity
= the slot's attention on the ray x occupancy, attention not detached.
Colour is view 0's on the ray plus a residual. Scene frame, units of R;
t is view-0 camera depth.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import camera_directions

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest depth half range (units of r_max) and its initial share.
OBJECT_DEPTH = 1.5
DEPTH_INIT = 0.9
# Height of Gaussians above the floor.
MIN_HEIGHT = 0.01
# Contact: attention-weighted floor depth of the rays, leaning to the
# lowest rows by exp(CONTACT_SHARPNESS * v / H).
CONTACT_SHARPNESS = 30.0
# Image box: attention mean +- this many standard deviations, at least
# MIN_BOX pixels.
BOX_STD = 2.0
MIN_BOX = 4.0
# Interval edge softness, in cells; Gaussian scale range, in cell sizes.
EDGE = 0.5
CELL_SCALE = (0.5, 2.0)
PE_BANDS = 4
# Per ray: front, back. Per cell: depth offset 1, scale 3, quat 4, rgb 3, sh1 9.
RAY_PARAMS = 2
CELL_PARAMS = 20


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


def _image_grid(side: int, image_size: tuple[int, int], device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel centres (u, v) of a side x side token grid over the image, (side*side,) each."""
    h, w = image_size
    idx = torch.arange(side, device=device, dtype=dtype)
    v, u = torch.meshgrid((idx + 0.5) * h / side - 0.5, (idx + 0.5) * w / side - 0.5, indexing="ij")
    return u.flatten(), v.flatten()


class RegionHead(nn.Module):
    """Slot latent -> depth range (centre shift from the contact depth, half range)."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 2))
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
            self.mlp[-1].bias[1] = _logit(DEPTH_INIT)

    def forward(self, slots: torch.Tensor, contact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), contact depth (K,) -> depth centre, half range (K,).
        The contact is the object's front; its centre is about r behind."""
        r_min, r_max = OBJECT_RADIUS
        raw = self.mlp(slots)
        centre = contact + 0.5 * (r_min + r_max) + r_max * torch.tanh(raw[:, 0])
        half = r_min + (OBJECT_DEPTH * r_max - r_min) * torch.sigmoid(raw[:, 1])
        return centre, half


class ObjectHead(nn.Module):
    """Ray token -> its interval and its cells' Gaussians."""

    def __init__(self, d_model: int, depths: int):
        super().__init__()
        self.depths = depths
        self.project = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, RAY_PARAMS + depths * CELL_PARAMS)
        )
        lo, hi = (math.log(s) for s in CELL_SCALE)
        with torch.no_grad():
            out = self.project[-1]
            out.weight.mul_(0.1)
            out.bias.zero_()
            out.bias[0] = _logit(0.3)  # front at 30% of the range
            out.bias[1] = _logit(0.5)  # back half way from the front to the end
            cell = out.bias[RAY_PARAMS:].view(depths, CELL_PARAMS)
            cell[:, 1:4] = _logit(-lo / (hi - lo))  # scale = one cell
            cell[:, 4] = 1.0  # identity quaternion (wxyz)

    def forward(
        self,
        feat: torch.Tensor,
        origin: torch.Tensor,
        rays: torch.Tensor,
        t_lo: torch.Tensor,
        t_hi: torch.Tensor,
        t_floor: torch.Tensor,
        spacing: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, N, d); camera origin (3,); rays (K, N, 3) at unit depth;
        depth range t_lo, t_hi (K,); floor depth (K, N); lateral ray spacing
        per unit depth (K,); attention (K, N), colours (K, N, 3) on the rays
        -> Gaussians (K, N * depths, ...)."""
        raw = self.project(feat)
        cell = raw[..., RAY_PARAMS:].unflatten(-1, (self.depths, CELL_PARAMS))  # (K, N, D, P)
        span = (t_hi - t_lo)[:, None]  # (K, 1)
        front = torch.sigmoid(raw[..., 0])
        back = front + (1.0 - front) * torch.sigmoid(raw[..., 1])
        t_front = t_lo[:, None] + front * span
        t_back = t_lo[:, None] + back * span  # (K, N)

        bin_ = span / self.depths  # (K, 1)
        idx = torch.arange(self.depths, device=feat.device, dtype=feat.dtype)
        t = t_lo[:, None, None] + (idx + 0.5) * bin_[..., None] + 0.5 * bin_[..., None] * torch.tanh(cell[..., 0])
        edge = EDGE * bin_[..., None]
        occupancy = (
            torch.sigmoid((t - t_front[..., None]) / edge)
            * torch.sigmoid((t_back[..., None] - t) / edge)
            * torch.sigmoid((t_floor[..., None] - t) / edge)  # above the floor
        )
        mu = origin + t[..., None] * rays[:, :, None]
        size = torch.maximum(spacing[:, None, None] * t, bin_[..., None])[..., None]
        out = {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": size * _log_uniform(cell[..., 1:4], *CELL_SCALE),
            "quat": F.normalize(cell[..., 4:8], dim=-1),
            "opacity": attn.clamp(0.0, 1.0)[..., None] * occupancy,
            "rgb": torch.sigmoid(torch.logit(colours.clamp(0.02, 0.98))[:, :, None] + cell[..., 8:11]),
            "sh1": cell[..., 11:20].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(1, 2) for name, value in out.items()}


class DecoderLayer(nn.Module):
    """Self-attention among the rays, cross-attention to the slot tokens,
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
        """q (K, N, d), slot tokens k, v (K, M, d) -> (K, N, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        return q + self.ffn(self.norm_ff(q))


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians in a frustum grid, scene frame."""

    def __init__(
        self,
        z_dim: int,
        ray_side: int,
        ray_depths: int,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        n_layers: int = 2,
    ):
        """ray_side: S rays per box side; ray_depths: D cells per ray."""
        super().__init__()
        d = int(hidden_dim)
        self.side, self.depths, self.M, self.d = int(ray_side), int(ray_depths), int(n_tokens), d
        # Ray token: features, attention 1, colour 3, encoded box position, floor depth 1.
        self.ray_in = nn.Linear(z_dim + 1 + 3 + (3 + 6 * PE_BANDS) + 1, d)
        self.k_proj = nn.Linear(z_dim, n_tokens * d)
        self.v_proj = nn.Linear(z_dim, n_tokens * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, n_heads) for _ in range(n_layers))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d, self.depths)
        lin = (torch.arange(self.side) + 0.5) / self.side * 2.0 - 1.0
        vv, uu = torch.meshgrid(lin, lin, indexing="ij")
        box = torch.stack([uu.flatten(), vv.flatten()], -1)  # (S*S, 2) in [-1, 1]
        self.register_buffer("box_grid", box, persistent=False)
        pe = positional_encoding(torch.cat([box, torch.zeros_like(box[:, :1])], -1), PE_BANDS)
        self.register_buffer("box_pe", pe, persistent=False)

    def forward(self, slots: torch.Tensor, view0: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """slots (K, C); view0: the slots' "attn" (K, s, s), "colours" (3, s, s),
        "features" (C, s, s), "frame", "focal_ratio", "image_size" ->
        Gaussians (K, S*S*D, ...), depth centre and half range (K, 2)."""
        K = slots.shape[0]
        frame, fr, (h, w) = view0["frame"], view0["focal_ratio"], view0["image_size"]
        attn = view0["attn"]
        s = attn.shape[-1]
        rot, origin = frame.rot, frame.origin

        def floor_depth(directions: torch.Tensor) -> torch.Tensor:
            """Camera depth where unit-depth rays meet z = 0 (far if they do not)."""
            dz = directions[..., 2]
            return torch.where(dz < -1e-6, -origin[2] / dz.clamp(max=-1e-6), torch.full_like(dz, 1e3))

        # Contact depth and image box from the attention (geometry: detached).
        a = attn.detach().flatten(1)  # (K, s*s)
        u, v = _image_grid(s, (h, w), a.device, a.dtype)
        token_dirs = camera_directions(u, v, (h, w), fr) @ rot.T
        lean = torch.log(a.clamp_min(1e-8)) * 2.0 + CONTACT_SHARPNESS * v / h
        contact = (lean.softmax(-1) * floor_depth(token_dirs).clamp(max=1e2)).sum(-1)  # (K,)
        p = a.pow(2) / a.pow(2).sum(-1, keepdim=True).clamp_min(1e-8)
        lo, hi = [], []
        for c, size in ((u, w), (v, h)):
            mean = (p * c).sum(-1)
            std = (p * (c - mean[:, None]) ** 2).sum(-1).sqrt()
            half = (BOX_STD * std).clamp_min(0.5 * MIN_BOX)
            lo.append((mean - half).clamp(-0.5, size - 0.5 - MIN_BOX))
            hi.append(torch.maximum(mean + half, lo[-1] + MIN_BOX).clamp(max=size - 0.5))

        # Rays: S x S over the box.
        g = 0.5 * (self.box_grid + 1.0)  # (N, 2) in [0, 1]
        ru = lo[0][:, None] + g[:, 0] * (hi[0] - lo[0])[:, None]  # (K, N)
        rv = lo[1][:, None] + g[:, 1] * (hi[1] - lo[1])[:, None]
        rays = camera_directions(ru, rv, (h, w), fr) @ rot.T  # (K, N, 3)
        t_floor = floor_depth(rays)
        centre, half = self.region(slots, contact)
        t_lo = (centre - half).clamp_min(1e-2)
        t_hi = centre + half
        spacing = torch.sqrt((hi[0] - lo[0]) * (hi[1] - lo[1])) / (self.side * math.sqrt(fr[0] * fr[1] * w * h))

        # View 0 on the rays; attention not detached.
        grid = torch.stack([(2.0 * ru + 1.0) / w - 1.0, (2.0 * rv + 1.0) / h - 1.0], -1)[:, None]  # (K, 1, N, 2)
        sample = lambda maps: F.grid_sample(maps, grid, align_corners=False)[:, :, 0].transpose(1, 2)
        a_ray = sample(attn[:, None])[..., 0]  # (K, N)
        c_ray = sample(view0["colours"][None].expand(K, -1, -1, -1))
        f_ray = sample(view0["features"][None].expand(K, -1, -1, -1))
        floor_rel = ((t_floor - centre[:, None]) / half[:, None]).clamp(-2.0, 2.0)
        q = self.ray_in(
            torch.cat([f_ray, a_ray[..., None], c_ray, self.box_pe.expand(K, -1, -1), floor_rel[..., None]], -1)
        )
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v_ = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        for layer in self.layers:
            q = layer(q, k, v_)
        gaussians = self.head(q, origin, rays, t_lo, t_hi, t_floor, spacing, a_ray, c_ray)
        return gaussians, torch.stack([centre, half], -1)
