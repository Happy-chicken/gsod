"""Object decoder: object slot -> floor-standing box -> Gaussians along
view 0's rays.

The box (RegionHead) sits at the slot's attention anchor, shifted along
view 0's ray and sized by the slot. Each ray of a side x side grid that
crosses the box carries `layers` Gaussians at increasing depths inside it:
the first is the visible surface, the others the back. A conv net over the
grid (token features, slot, attention, colour, box range) predicts them.
Opacity is the slot's view-0 attention on the ray times a learned value:
attention gives the silhouette, the slot and features the depth. Colour is
view 0's on the ray plus a residual. Scene frame, units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import pixel_rays

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
# Gaussian scale range, in ray spacings at its depth.
RAY_SCALE = (0.5, 3.0)
# Initial opacity factor of the first (visible) and the back layers.
FRONT_OPACITY = 0.9
BACK_OPACITY = 0.3
# Per layer: depth step 1, scale 3, quat 4, opacity 1, rgb 3, sh1 9.
N_PARAMS = 21


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
    """Raw per-ray parameters -> Gaussians along the rays."""

    def __init__(self, layers: int):
        super().__init__()
        self.layers = layers

    def init_bias(self, bias: torch.Tensor) -> None:
        """Output bias (layers * N_PARAMS,): evenly spaced depths, identity
        quaternion, front / back opacity."""
        b = bias.view(self.layers, N_PARAMS)
        b.zero_()
        for layer in range(self.layers):
            b[layer, 0] = _logit(1.0 / (self.layers + 1 - layer))  # depth fraction (l+1)/(L+1)
        b[:, 4] = 1.0
        b[0, 8] = _logit(FRONT_OPACITY)
        b[1:, 8] = _logit(BACK_OPACITY)

    def forward(
        self,
        raw: torch.Tensor,
        origin: torch.Tensor,
        rays: torch.Tensor,
        t_in: torch.Tensor,
        t_out: torch.Tensor,
        spacing: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """raw (K, N, layers, N_PARAMS); camera origin (3,), rays (N, 3) at unit
        depth; box range t_in, t_out (K, N); ray spacing per unit depth;
        attention (K, N, 1) (0 off the box) and colours (N, 3) on the rays ->
        Gaussians (K, N * layers, ...)."""
        # Increasing depth fractions in [0, 1] of the box range.
        steps = torch.sigmoid(raw[..., 0])
        fractions = 1.0 - torch.cumprod(1.0 - steps, -1)
        t = t_in[..., None] + fractions * (t_out - t_in)[..., None]  # (K, N, L)
        mu = origin + t[..., None] * rays[:, None]
        width = (spacing * t)[..., None]
        out = {
            "mu": torch.cat([mu[..., :2], mu[..., 2:].clamp_min(MIN_HEIGHT)], -1),
            "scale": width * _log_uniform(raw[..., 1:4], *RAY_SCALE),
            "quat": F.normalize(raw[..., 4:8], dim=-1),
            "opacity": attn * torch.sigmoid(raw[..., 8]),
            "rgb": torch.sigmoid(torch.logit(colours.clamp(0.02, 0.98))[:, None] + raw[..., 9:12]),
            "sh1": raw[..., 12:21].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(1, 2) for name, value in out.items()}


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians along view 0's rays, scene frame."""

    def __init__(self, z_dim: int, ray_side: int, layers: int, hidden_dim: int):
        """ray_side: rays per image side; layers: Gaussians per ray and slot."""
        super().__init__()
        self.side, self.layers = int(ray_side), int(layers)
        self.region = RegionHead(z_dim)
        # Inputs: features, slot, attention, colour, box range (t_in, t_out - t_in).
        d = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2 * z_dim + 1 + 3 + 2, d, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(d, self.layers * N_PARAMS, 1),
        )
        self.head = ObjectHead(self.layers)
        with torch.no_grad():
            out = self.net[-1]
            out.weight.mul_(0.1)
            self.head.init_bias(out.bias)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor, view0: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray (K, 3); view0: the slots' "attn" (K, s, s),
        "colours" (3, s, s), "features" (C, s, s), "frame", "focal_ratio",
        "image_size" -> Gaussians (K, side * side * layers, ...), box centres
        and half sizes (K, 3)."""
        K, r = slots.shape[0], self.side
        frame, focal_ratio = view0["frame"], view0["focal_ratio"]
        centre, extent = self.region(slots, anchor, ray)

        # Where each ray crosses each box (slab test); t is camera depth.
        origin = frame.origin
        rays = pixel_rays(frame, r, view0["image_size"], focal_ratio)  # (N, 3)
        safe = torch.where(rays.abs() < 1e-6, torch.full_like(rays, 1e-6), rays)
        t1 = (centre[:, None] - extent[:, None] - origin) / safe
        t2 = (centre[:, None] + extent[:, None] - origin) / safe
        t_in = torch.minimum(t1, t2).amax(-1).clamp_min(1e-3)  # (K, N)
        t_out = torch.maximum(t1, t2).amin(-1)
        crosses = (t_out > t_in).to(t_in.dtype)
        t_out = torch.maximum(t_out, t_in)

        # View-0 maps at the rays; attention not detached.
        pool = lambda x: F.adaptive_avg_pool2d(x, r)
        attn = pool(view0["attn"][:, None])  # (K, 1, r, r)
        colours = pool(view0["colours"][None])  # (1, 3, r, r)
        features = pool(view0["features"][None])
        grid = lambda x: x.view(K, 1, r, r)
        maps = torch.cat(
            [
                features.expand(K, -1, -1, -1),
                slots[:, :, None, None].expand(-1, -1, r, r),
                attn,
                colours.expand(K, -1, -1, -1),
                grid(t_in),
                grid(t_out - t_in),
            ],
            1,
        )
        raw = self.net(maps).view(K, self.layers, N_PARAMS, r * r).permute(0, 3, 1, 2)
        spacing = 1.0 / (r * math.sqrt(focal_ratio[0] * focal_ratio[1]))
        gate = attn.flatten(1)[..., None] * crosses[..., None]  # (K, N, 1)
        gaussians = self.head(raw, origin, rays, t_in, t_out, spacing, gate, colours[0].flatten(1).T)
        return gaussians, centre, extent
