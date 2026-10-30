"""Gaussian parameter heads for slots.

A head turns the decoder's per-Gaussian features into Gaussians, in the
*azimuth frame of the input view* (yaw-rotated so view 0 looks along +x):

    mu / scale / quat / opacity / rgb, each (K, G, ...) for K slots

Only the object slots are decoded:

ObjectHead      object slots: around the region the slot's attention covers
                (centre and extent per slot, from GSODModel), starting with
                that region's mean colour, so a slot's Gaussians depend on
                the input through its attention from the first step. While
                the attention is still spread, so are the Gaussians. Colour:
                region colour + a per-slot correction from the latent + a
                small per-Gaussian residual ("rgb_residual", penalised in
                losses.py), so one slot is a poor fit for two objects of
                different colours.

The background is not decoded: it is the empty space behind the objects
(gsod.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

OPACITY_INIT = -2.2  # sigmoid ~= 0.10
OBJECT_SCALE_INIT = 0.1


@dataclass(frozen=True)
class SceneBounds:
    """Scene geometry priors from a SceneGenConfig dict (meta.json["config"]).

    radius_range is the objects' half size: they rest on the floor with their
    centres at z = floor + r. fg_xy: half width of the object box (the
    placement area widened by the largest object). d_max: the farthest
    camera distance.
    """

    floor_z: float
    r_min: float
    r_max: float
    fg_xy: float
    d_max: float

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
            d_max=d_max,
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
        self.scale_range = (0.02, bounds.r_max)
        self.project = _projection(d_model, 14)
        # Colour correction of each slot, from its latent; starts at zero.
        self.base = nn.Linear(z_dim, 3)
        lo, hi = (math.log(s) for s in self.scale_range)
        with torch.no_grad():
            bias = self.project[-1].bias
            bias[3:6] = _logit((math.log(OBJECT_SCALE_INIT) - lo) / (hi - lo))
            bias[6] = 1.0  # identity quaternion (w first, gsplat convention)
            bias[10] = OPACITY_INIT
            self.base.weight.zero_()
            self.base.bias.zero_()

    def forward(
        self,
        feat: torch.Tensor,
        slots: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat: (K, G, d); slots: (K, C); per slot, its attention region:
        centre (K, 3), extent (K, 3) and mean colour (K, 3) -> Gaussians
        (K, G, ...), with the colour residuals as "rgb_residual"."""
        raw = self.project(feat)
        residual = raw[..., 11:14]
        region = torch.logit(colour.clamp(0.02, 0.98)) + self.base(slots)
        return {
            "mu": centre[:, None] + extent[:, None] * torch.tanh(raw[..., 0:3]),
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(raw[..., 10]),
            "rgb": torch.sigmoid(region[:, None] + residual),
            "rgb_residual": residual,
        }
