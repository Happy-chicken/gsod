"""Gaussian parameter heads for slots.

A head turns the decoder's per-Gaussian features into Gaussians, in the
*azimuth frame of the input view* (yaw-rotated so view 0 looks along +x):

    mu / scale / quat / opacity / rgb, each (K, G, ...) for K slots

There is no per-slot frame: where a slot's Gaussians go is learned, inside
fixed regions that are the structural prior (no masks):

ObjectHead      object slots: anywhere in the object box, the volume objects
                can occupy. Its Gaussians start spread over the whole box, so
                every object slot can reach every object and the slots
                compete for them (render-time compositing, gsod.py). Colour:
                one base colour per slot plus a small per-Gaussian residual
                ("rgb_residual", penalised in losses.py), so one slot is a
                poor fit for two objects of different colours.
BackgroundHead  background slot: a floor layer and a far shell, both outside
                the object box, with a lower bound on scale, so it cannot
                draw an object (at most a blurred decal on the floor, which
                the other views reject).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

OPACITY_INIT = -2.2  # sigmoid ~= 0.10
OBJECT_SCALE_INIT = 0.1
FLOOR_THICKNESS = 0.01


@dataclass(frozen=True)
class SceneBounds:
    """Scene geometry priors from a SceneGenConfig dict (meta.json["config"]).

    radius_range is the objects' half size: they rest on the floor with their
    centres at z = floor + r. fg_xy / fg_z: the object box (the placement
    area widened by the largest object; floor to the top of the largest
    object). floor_reach: how far the floor layer extends (beyond the
    cameras). reach: radius of the far shell.
    """

    floor_z: float
    r_min: float
    r_max: float
    fg_xy: float
    fg_z: tuple[float, float]
    floor_reach: float
    reach: float

    @classmethod
    def from_config(cls, gen: dict) -> "SceneBounds":
        r_min, r_max = (float(r) for r in gen["radius_range"])
        floor_z = float(gen.get("floor_z", 0.0))
        # A cube turned about z reaches sqrt(2) * r from its centre.
        fg_xy = float(gen["placement_radius"]) + 1.5 * r_max
        d_max = float(gen["distance_range"][1])
        return cls(
            floor_z=floor_z,
            r_min=r_min,
            r_max=r_max,
            fg_xy=fg_xy,
            fg_z=(floor_z, floor_z + 2.0 * r_max + 0.25),
            floor_reach=d_max + fg_xy,
            reach=3.0 * (d_max + float(gen.get("target_height", 0.4))),
        )


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def _projection(d_model: int, n_params: int) -> nn.Sequential:
    """Feature -> raw parameters. The first three rows (positions) keep their
    default weight scale, so the spread of the decoder's Gaussian queries
    spreads the Gaussians; the other rows start small."""
    mlp = nn.Sequential(
        nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, n_params)
    )
    with torch.no_grad():
        mlp[-1].weight[3:].mul_(0.1)
        mlp[-1].bias.zero_()
    return mlp


class ObjectHead(nn.Module):
    """Raw parameters: position 3, scale 3, quat 4, opacity 1, rgb residual 3."""

    def __init__(self, z_dim: int, d_model: int, bounds: SceneBounds):
        super().__init__()
        self.xy = bounds.fg_xy
        self.z = bounds.fg_z
        self.scale_range = (0.02, bounds.r_max)
        self.project = _projection(d_model, 14)
        # Base colour of each slot, from its latent; starts at grey.
        self.base = nn.Linear(z_dim, 3)
        lo, hi = (math.log(s) for s in self.scale_range)
        with torch.no_grad():
            bias = self.project[-1].bias
            bias[3:6] = _logit((math.log(OBJECT_SCALE_INIT) - lo) / (hi - lo))
            bias[6] = 1.0  # identity quaternion (w first, gsplat convention)
            bias[10] = OPACITY_INIT
            self.base.weight.zero_()
            self.base.bias.zero_()

    def forward(self, feat: torch.Tensor, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """feat: (K, G, d); slots: (K, C) -> Gaussians (K, G, ...), with the
        colour residuals as "rgb_residual"."""
        raw = self.project(feat)
        z_lo, z_hi = self.z
        mu = torch.cat(
            [
                self.xy * torch.tanh(raw[..., 0:2]),
                z_lo + (z_hi - z_lo) * torch.sigmoid(raw[..., 2:3]),
            ],
            dim=-1,
        )
        residual = raw[..., 11:14]
        return {
            "mu": mu,
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(raw[..., 10]),
            "rgb": torch.sigmoid(self.base(slots)[:, None] + residual),
            "rgb_residual": residual,
        }


class BackgroundHead(nn.Module):
    """The first half of the Gaussians form a flat floor layer at floor_z
    within floor_reach; the second half a shell around the scene centre from
    just outside the object box out to reach. Scales are at least half the
    smallest object's size.

    Raw parameters: position / direction 3, shell radius 1, scale 3, quat 4,
    opacity 1, rgb 3.
    """

    def __init__(self, d_model: int, bounds: SceneBounds, n_gaussians: int):
        super().__init__()
        self.n_floor = n_gaussians // 2
        self.floor_z = bounds.floor_z
        self.floor_reach = bounds.floor_reach
        # The shell starts at the object box's farthest corner.
        self.shell = (
            math.sqrt(2.0 * bounds.fg_xy**2 + (bounds.fg_z[1] - bounds.floor_z) ** 2),
            bounds.reach,
        )
        s_min = 0.5 * bounds.r_min
        self.floor_scale = (s_min, bounds.floor_reach / 4.0)
        self.shell_scale = (s_min, bounds.reach / 4.0)
        self.project = _projection(d_model, 15)
        with torch.no_grad():
            bias = self.project[-1].bias
            bias[7] = 1.0  # identity quaternion
            bias[11] = OPACITY_INIT

    def forward(self, feat: torch.Tensor, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """feat: (1, G, d) -> Gaussians (1, G, ...)."""
        raw = self.project(feat)
        floor, shell = raw[:, : self.n_floor], raw[:, self.n_floor :]

        floor_xy = self.floor_reach * torch.tanh(floor[..., 0:2])
        floor_mu = torch.cat([floor_xy, torch.full_like(floor_xy[..., :1], self.floor_z)], -1)
        floor_scale = _log_uniform(floor[..., 4:6], *self.floor_scale)
        floor_scale = torch.cat(
            [floor_scale, torch.full_like(floor_scale[..., :1], FLOOR_THICKNESS)], -1
        )
        floor_quat = torch.zeros_like(floor[..., 7:11])
        floor_quat[..., 0] = 1.0  # flat in xy

        r_in, r_out = self.shell
        radius = r_in + (r_out - r_in) * torch.sigmoid(shell[..., 3:4])
        centre = shell.new_tensor([0.0, 0.0, self.floor_z])
        shell_mu = centre + radius * F.normalize(shell[..., 0:3], dim=-1)

        return {
            "mu": torch.cat([floor_mu, shell_mu], 1),
            "scale": torch.cat(
                [floor_scale, _log_uniform(shell[..., 4:7], *self.shell_scale)], 1
            ),
            "quat": torch.cat([floor_quat, F.normalize(shell[..., 7:11], dim=-1)], 1),
            "opacity": torch.sigmoid(raw[..., 11]),
            "rgb": torch.sigmoid(raw[..., 12:15]),
        }
