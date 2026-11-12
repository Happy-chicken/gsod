"""Object decoder: geometry lifted from view 0's pixels, slots own it.

Every pixel of a side x side grid of view 0 carries `layers` Gaussians along
its ray (visible surface, back), predicted from the pixel's feature mixed
with the slots it belongs to. Ownership w = softmax over slots (background
first) of query(pixel) . key(slot). A Gaussian owned by object slot k is
drawn only inside k's box (RegionHead, at k's attention anchor) and by k's
draw factor:

    opacity = base * sum_k w_k * inside_k * draw_k     (k = objects)

so reconstruction depends on the ownership. Colour = view 0's at the pixel
+ residual. Scene frame, units of R; t is view-0 camera depth.
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
# Initial box half sizes x, y, z (units of r_max); no upper limit.
EXTENT_INIT = (1.375, 1.375, 1.2)
# Height of boxes and Gaussians above the floor.
MIN_HEIGHT = 0.01
# Softness of a box's faces (units of R).
BOX_EDGE = 0.02
# A Gaussian's depth: t = t_floor * (1 - DEPTH_SPAN * sigmoid(raw)); layer starts.
DEPTH_SPAN = 0.6
DEPTH_INIT = (0.3, 0.15)
# Gaussian scale range, in pixel footprints at its depth.
PIXEL_SCALE = (0.5, 4.0)
OPACITY_INIT = 0.9
# Per Gaussian: depth 1, scale 3, quat 4, opacity 1, rgb 3, sh1 9.
GAUSS_PARAMS = 21


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def positional_encoding(x: torch.Tensor, bands: int) -> torch.Tensor:
    """x (N, 3) in [-1, 1] -> (N, 3 + 6 * bands): x, sin / cos(2^k pi x)."""
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    angles = x[..., None] * freqs  # (N, 3, bands)
    return torch.cat([x, angles.sin().flatten(1), angles.cos().flatten(1)], -1)


def inside_boxes(mu: torch.Tensor, centre: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    """Points (N, 3), boxes (K, 3) -> soft inside (N, K); heights cut at the floor."""
    d = (mu[:, None, :2] - centre[None, :, :2]).abs()
    side = torch.sigmoid((extent[None, :, :2] - d) / BOX_EDGE).prod(-1)
    lo = (centre[:, 2] - extent[:, 2]).clamp_min(MIN_HEIGHT)
    hi = centre[:, 2] + extent[:, 2]
    z = mu[:, None, 2]
    return side * torch.sigmoid((z - lo) / BOX_EDGE) * torch.sigmoid((hi - z) / BOX_EDGE)


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
    """Gaussian feature -> raw parameters -> Gaussians along view 0's rays."""

    def __init__(self, d_model: int, layers: int):
        super().__init__()
        self.layers = layers
        self.project = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, GAUSS_PARAMS))
        lo, hi = (math.log(s) for s in PIXEL_SCALE)
        with torch.no_grad():
            out = self.project[-1]
            out.weight.mul_(0.1)
            out.weight[12:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[1:4] = _logit(-lo / (hi - lo))  # one pixel footprint
            out.bias[4] = 1.0  # identity quaternion (wxyz)
            out.bias[8] = _logit(OPACITY_INIT)
        self.depth_bias = nn.Parameter(torch.tensor([_logit(d) for d in DEPTH_INIT]))

    def forward(
        self, feat: torch.Tensor, origin: torch.Tensor, rays: torch.Tensor, t_floor: torch.Tensor,
        spacing: float, colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (N, L, d); camera origin (3,); rays (N, 3) at unit depth; floor
        depth (N,); pixel spacing per unit depth; colours (N, 3) -> Gaussians
        (N * L, ...), opacity before ownership."""
        raw = self.project(feat)  # (N, L, P)
        t = t_floor[:, None] * (1.0 - DEPTH_SPAN * torch.sigmoid(raw[..., 0] + self.depth_bias))
        mu = origin + t[..., None] * rays[:, None]
        out = {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": (spacing * t)[..., None] * _log_uniform(raw[..., 1:4], *PIXEL_SCALE),
            "quat": F.normalize(raw[..., 4:8], dim=-1),
            "opacity": torch.sigmoid(raw[..., 8]),
            "rgb": torch.sigmoid(torch.logit(colours.clamp(0.02, 0.98))[:, None] + raw[..., 9:12]),
            "sh1": raw[..., 12:21].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(0, 1) for name, value in out.items()}


class ObjectDecoder(nn.Module):
    """Pixel features + slots -> owned Gaussians along view 0's rays."""

    def __init__(self, z_dim: int, ray_side: int, ray_layers: int, hidden_dim: int):
        super().__init__()
        d = int(hidden_dim)
        self.side, self.layers = int(ray_side), int(ray_layers)
        self.layer_embed = nn.Parameter(torch.randn(self.layers, z_dim) * 0.02)
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.mix = nn.Sequential(nn.Linear(2 * z_dim, d), nn.ReLU(True), nn.Linear(d, d), nn.ReLU(True))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d, self.layers)

    def forward(
        self, slots: torch.Tensor, bg_slot: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor,
        draw: torch.Tensor, view0: dict,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """slots (K-1, C), bg_slot (C,), anchor / ray (K-1, 3), draw (K-1,);
        view0: "features" (C, s, s), "colours" (3, s, s), "frame",
        "focal_ratio", "image_size" -> Gaussians (N, ...), ownership of
        object slots (N, K-1), box centres and half sizes (K-1, 3)."""
        frame, fr, (h, w) = view0["frame"], view0["focal_ratio"], view0["image_size"]
        r = self.side
        feat = F.adaptive_avg_pool2d(view0["features"][None], r)[0].flatten(1).T  # (r*r, C)
        colours = F.adaptive_avg_pool2d(view0["colours"][None], r)[0].flatten(1).T

        # View 0's rays through the grid; depth where they meet the floor.
        idx = torch.arange(r, device=feat.device, dtype=feat.dtype)
        v, u = torch.meshgrid((idx + 0.5) * h / r - 0.5, (idx + 0.5) * w / r - 0.5, indexing="ij")
        rays = camera_directions(u.flatten(), v.flatten(), (h, w), fr) @ frame.rot.T  # (N, 3)
        far = frame.origin.norm() + 2.0 * OBJECT_REGION
        dz = rays[:, 2]
        t_floor = torch.where(dz < -1e-6, (MIN_HEIGHT - frame.origin[2]) / dz.clamp(max=-1e-6), far).clamp(max=far)
        spacing = 1.0 / (r * math.sqrt(fr[0] * fr[1]))

        # Ownership over [background, objects]; features mixed with the owners.
        x = feat[:, None] + self.layer_embed  # (N, L, C)
        every = torch.cat([bg_slot[None], slots], 0)  # (K, C)
        w_own = (self.query(x) @ self.key(every).T / math.sqrt(x.shape[-1])).softmax(-1)  # (N, L, K)
        hidden = self.mix(torch.cat([x, w_own @ every], -1))
        gaussians = self.head(hidden, frame.origin, rays, t_floor, spacing, colours)

        centre, extent = self.region(slots, anchor, ray)
        inside = inside_boxes(gaussians["mu"], centre, extent)  # (N*L, K-1)
        share = w_own.flatten(0, 1)[:, 1:] * inside * draw  # (N*L, K-1)
        total = share.sum(-1)
        gaussians["opacity"] = gaussians["opacity"] * total
        owner = share / total[:, None].clamp_min(1e-8)
        return gaussians, owner, centre, extent
