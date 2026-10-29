"""Pixel-aligned Gaussians whose appearance the slots compete for.

Geometry and appearance come from different inputs:

* ``PixelGeometry``: view 0's encoder features -> G Gaussians on the ray of
  every feature cell (depth along the ray, scale, rotation). It knows
  nothing about slots, so every visible surface gets Gaussians, whether or
  not a slot has claimed it.
* ``SlotMixer``: each Gaussian's position + every slot -> colour and density
  per slot, mixed as in uORF (Yu et al., ICLR 2022):

      sigma_k = softplus(MLP(slot_k, PE(x)))     object slots: 0 outside the
                                                  locality box
      opacity = 1 - exp(-sum_k sigma_k)
      pi_k    = sigma_k / sum_j sigma_j           slot probabilities
      rgb     = sum_k pi_k * rgb_k

  Every slot is asked at every Gaussian and the pi_k sum to one there, so
  the slots compete for each point by how well they explain it, and a slot
  that loses still gets a gradient. The mixer never sees the encoder
  features: what a Gaussian looks like must go through the slots, or the
  reconstruction would not depend on them. For the same reason it reads the
  positions detached: otherwise PixelGeometry learns to code each pixel's
  colour into tiny depth changes, which the Fourier features of one field
  (the background's) decode, and the slots are bypassed through the
  geometry. Geometry is trained by the rendering alone.

Positions are in the azimuth frame of view 0 (see gsod.py).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from gsod.model.decoders.parameter_head import SceneBounds

# Raw values per Gaussian from PixelGeometry: depth 1, log-scale 3, quat 4.
_GEOMETRY_PARAMS = 8
# Gaussian std relative to the footprint of its feature cell at its depth.
_SCALE_RANGE = (0.25, 4.0)
# Initial density of every slot at every Gaussian: all slots start equal.
_SIGMA_INIT = 0.3


class PixelGeometry(nn.Module):
    """Feature map (C, s, s) -> per_pixel Gaussians on each cell's ray.

    Layer 0 is the visible surface: its depth starts where the ray meets the
    floor (rays at or above the horizon start at the far bound) and moves
    log-uniformly within [t_near, t_far]. Layers 1.. lie behind it, at most an
    object diameter further, for the sides view 0 does not see. Depths are
    camera z-depths (the rays have camera z = 1).
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        per_pixel: int,
        bounds: SceneBounds,
        near: float,
        cell: float,
    ):
        """cell: width of one feature cell at depth 1 (its footprint per unit
        depth)."""
        super().__init__()
        self.per_pixel = int(per_pixel)
        self.cell = float(cell)
        self.floor_z = bounds.floor_z
        self.t_near = max(2.0 * near, bounds.distance_range[0] - 1.5 * bounds.fg_xy)
        self.t_far = bounds.reach
        self.behind = 2.0 * bounds.r_max
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, hidden_dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, self.per_pixel * _GEOMETRY_PARAMS, 1),
        )
        with torch.no_grad():
            last = self.net[-1]
            last.weight.mul_(0.1)
            last.bias.view(self.per_pixel, _GEOMETRY_PARAMS).zero_()
            # Identity quaternion (w first, gsplat convention).
            last.bias.view(self.per_pixel, _GEOMETRY_PARAMS)[:, 4] = 1.0

    def forward(
        self, feat: torch.Tensor, origin: torch.Tensor, directions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """feat: (C, s, s); origin: (3,); directions: (s*s, 3), row-major.

        Returns (mu / scale / quat of per_pixel * s*s Gaussians, layer-major,
        and their depths (per_pixel, s*s)).
        """
        n_pix = directions.shape[0]
        raw = self.net(feat[None])[0].reshape(self.per_pixel, _GEOMETRY_PARAMS, n_pix)

        log_lo, log_hi = math.log(self.t_near), math.log(self.t_far)
        dz = directions[:, 2]
        t_floor = torch.where(
            dz < -1e-6,
            (self.floor_z - origin[2]) / dz.clamp(max=-1e-6),
            torch.full_like(dz, self.t_far),
        )
        frac = (t_floor.clamp(self.t_near, self.t_far).log() - log_lo) / (log_hi - log_lo)
        prior = torch.logit(frac.clamp(0.02, 0.98))
        surface = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(raw[0, 0] + prior))
        depth = torch.cat(
            [surface[None], surface + self.behind * torch.sigmoid(raw[1:, 0])]
        )  # (per_pixel, s*s)

        mu = origin + depth[..., None] * directions  # (per_pixel, s*s, 3)
        lo, hi = (math.log(r) for r in _SCALE_RANGE)
        rel = torch.exp(lo + (hi - lo) * torch.sigmoid(raw[:, 1:4]))  # (G, 3, s*s)
        scale = (depth * self.cell)[:, None] * rel
        geometry = {
            "mu": mu.reshape(-1, 3),
            "scale": scale.transpose(1, 2).reshape(-1, 3),
            "quat": F.normalize(raw[:, 4:8], dim=1).transpose(1, 2).reshape(-1, 4),
        }
        return geometry, depth


class _SlotField(nn.Module):
    """(slot, encoded position) -> rgb logits (3) and density logit (1).

    The first layer is split into a slot and a position part, so the (K, M)
    concatenated input is never built.
    """

    def __init__(self, z_dim: int, pe_dim: int, hidden_dim: int):
        super().__init__()
        self.slot_in = nn.Linear(z_dim, hidden_dim)
        self.pos_in = nn.Linear(pe_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 4),
        )
        with torch.no_grad():
            self.mlp[-1].bias[3] = math.log(math.expm1(_SIGMA_INIT))

    def forward(self, slots: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """slots: (K, C); pe: (M, P) -> (K, M, 4)."""
        return self.mlp(self.slot_in(slots)[:, None] + self.pos_in(pe)[None])


class SlotMixer(nn.Module):
    """Gaussian positions (M, 3) + slots (K, C), 0 = background ->
    (rgb (M, 3), opacity (M,), slot probabilities (M, K)).

    Object slots share one field, the background slot has its own (uORF).
    Object densities are zero outside the box objects can occupy, so the
    floor and everything far away is the background's.
    """

    def __init__(self, z_dim: int, hidden_dim: int, bounds: SceneBounds, n_freqs: int = 4):
        super().__init__()
        self.fg_xy = bounds.fg_xy
        self.fg_z = bounds.fg_z
        self.register_buffer(
            "freqs", math.pi * 2.0 ** torch.arange(n_freqs, dtype=torch.float32),
            persistent=False,
        )
        pe_dim = 3 * (1 + 2 * n_freqs)
        self.bg = _SlotField(z_dim, pe_dim, hidden_dim)
        self.obj = _SlotField(z_dim, pe_dim, hidden_dim)

    def encode_position(self, mu: torch.Tensor) -> torch.Tensor:
        """Positions scaled so the object box spans [-1, 1] in xy, plus
        Fourier features."""
        x = mu / self.fg_xy
        angles = (x[..., None] * self.freqs.to(x.dtype)).flatten(-2)
        return torch.cat([x, angles.sin(), angles.cos()], dim=-1)

    def forward(
        self, mu: torch.Tensor, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detached: no colour path from the pixel features through the depth.
        pe = self.encode_position(mu.detach())
        out = torch.cat([self.bg(slots[:1], pe), self.obj(slots[1:], pe)])  # (K, M, 4)
        sigma = F.softplus(out[..., 3])
        z_lo, z_hi = self.fg_z
        inside = (
            (mu[:, :2].abs() <= self.fg_xy).all(-1) & (mu[:, 2] >= z_lo) & (mu[:, 2] <= z_hi)
        ).to(sigma.dtype)
        sigma = torch.cat([sigma[:1], sigma[1:] * inside])
        total = sigma.sum(0)  # (M,)
        probs = sigma / total.clamp_min(1e-8)  # (K, M)
        rgb = (probs[..., None] * torch.sigmoid(out[..., :3])).sum(0)
        return rgb, 1.0 - torch.exp(-total), probs.T
