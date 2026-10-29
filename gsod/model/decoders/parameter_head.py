"""Gaussian parameter head for slots.

The head turns per-Gaussian features into the common output interface:

    mu / scale / quat / opacity / rgb

GSODModel uses it in the background decoder (the Gaussians outside view 0;
see pixel_gaussians.py for the rest). Slot ids here are local to the
decoder, 0..K-1 over the slots it decodes.

The first three values are offsets inside a per-slot box (``SlotFrame``) in
the *azimuth frame of the input view* (yaw-rotated so view 0 looks along +x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

PARAMS_PER_GAUSSIAN = 14  # mu/offset 3, log-scale 3, quat 4, opacity 1, rgb 3

LOG_SCALE_INIT = math.log(0.1)
OPACITY_INIT = -2.2  # sigmoid ~= 0.10


def _init_head(head: nn.Linear) -> None:
    """Shared projection head of the attention decoder.

    It has a single bias row for all Gaussians, so the spatial spread must come
    from the per-Gaussian queries. The offset rows therefore keep their default
    weight scale; only the other rows are shrunk.
    """
    with torch.no_grad():
        head.weight[3:].mul_(0.1)
        head.bias.zero_()
        head.bias[3:6] = LOG_SCALE_INIT
        head.bias[6] = 1.0  # identity quaternion (w first, gsplat convention)
        head.bias[10] = OPACITY_INIT


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class SceneBounds:
    """Scene geometry priors from a SceneGenConfig dict (meta.json["config"]).

    fg_xy / fg_z: the box objects can occupy (the placement area widened by
    the largest object; from the floor to the top of the largest object),
    the locality box of the object slots. reach: beyond the farthest camera,
    so the background can surround every view. radius_range is the objects'
    half size: they rest on the floor with their centres at z = floor + r.
    """

    floor_z: float
    r_max: float
    fg_xy: float
    fg_z: tuple[float, float]
    distance_range: tuple[float, float]
    reach: float

    @classmethod
    def from_config(cls, gen: dict) -> "SceneBounds":
        r_max = float(gen["radius_range"][1])
        floor_z = float(gen.get("floor_z", 0.0))
        d_lo, d_hi = (float(d) for d in gen["distance_range"])
        return cls(
            floor_z=floor_z,
            r_max=r_max,
            # A cube turned about z reaches sqrt(2) * r from its centre.
            fg_xy=float(gen["placement_radius"]) + 1.5 * r_max,
            fg_z=(floor_z - 0.1, floor_z + 2.0 * r_max + 0.25),
            distance_range=(d_lo, d_hi),
            reach=3.0 * (d_hi + float(gen.get("target_height", 0.4))),
        )


class SlotFrame(nn.Module):
    """slot latent -> (centre, extent), bounded by dataset priors.

    Everything is predicted from the slot latent alone: centre xy, softly
    bounded to [-xy, xy]; centre z in [z[0], z[1]]; extent in
    [extent[0], extent[1]], log-uniform in the predicted value and
    initialised at extent_init. Gaussian centres: centre + extent *
    tanh(offset).

    GSODModel uses it for the background decoder only
    (``from_scene_config``): a frame from the scene's size out beyond the
    farthest camera, so its Gaussians can fill what view 0 does not see.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        *,
        xy: float,
        z: tuple[float, float],
        extent: tuple[float, float],
        extent_init: float,
    ):
        super().__init__()
        self.xy = float(xy)
        self.z = (float(z[0]), float(z[1]))
        self.extent = (float(extent[0]), float(extent[1]))

        self.fg = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 6)
        )
        log_lo, log_hi = math.log(self.extent[0]), math.log(self.extent[1])
        with torch.no_grad():
            self.fg[-1].bias.zero_()
            self.fg[-1].bias[3:6] = _logit(
                (math.log(extent_init) - log_lo) / (log_hi - log_lo)
            )

    @classmethod
    def from_scene_config(cls, z_dim: int, gen: dict) -> "SlotFrame":
        """Background frame from a SceneGenConfig dict."""
        b = SceneBounds.from_config(gen)
        return cls(
            z_dim,
            xy=b.fg_xy,
            z=b.fg_z,
            extent=(b.fg_xy, b.reach),
            extent_init=math.sqrt(b.fg_xy * b.reach),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: (K, C) slots -> centre (K, 3), extent (K, 3)."""
        out = self.fg(slots)
        z_lo, z_hi = self.z
        z = z_lo + (z_hi - z_lo) * torch.sigmoid(out[:, 2:3])
        xy = self.xy * torch.tanh(out[:, 0:2] / self.xy)
        log_lo, log_hi = math.log(self.extent[0]), math.log(self.extent[1])
        extent = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(out[:, 3:6]))
        return torch.cat([xy, z], dim=-1), extent


class GaussianParameterHead(nn.Module):
    """Decode per-Gaussian features into the common Gaussian dict.

    The features are projected to the 14 raw parameters; their first three
    values are offsets inside each slot's frame (SlotFrame).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        frame: SlotFrame,
        frame_scale_min: float = 0.05,
        frame_scale_max: float = 0.5,
    ):
        super().__init__()
        self.frame_scale_min = float(frame_scale_min)
        self.frame_scale_max = float(frame_scale_max)
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, PARAMS_PER_GAUSSIAN),
        )
        _init_head(self.project[-1])
        self.frame = frame

    def forward(
        self, raw: torch.Tensor, slots: torch.Tensor, slot_ids: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """raw: (M, in_dim); slots: (K, C); slot_ids: (M,) in 0..K-1.

        Returns the Gaussian dict and (M, K) one-hot slot probabilities.
        """
        raw = self.project(raw)
        assert raw.shape[0] == slot_ids.shape[0]

        centre, extent = self.frame(slots)
        c, e = centre[slot_ids], extent[slot_ids]
        rel_scale = self.frame_scale_min + (
            self.frame_scale_max - self.frame_scale_min
        ) * torch.sigmoid(raw[:, 3:6])
        gaussians = {
            "mu": c + e * torch.tanh(raw[:, 0:3]),
            "scale": e * rel_scale,
            "quat": F.normalize(raw[:, 6:10], dim=-1),
            "opacity": torch.sigmoid(raw[:, 10]),
            "rgb": torch.sigmoid(raw[:, 11:14]),
        }
        slot_probs = F.one_hot(slot_ids, slots.shape[0]).to(dtype=slots.dtype)
        return gaussians, slot_probs
