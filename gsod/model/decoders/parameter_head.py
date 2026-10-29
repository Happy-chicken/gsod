"""Gaussian parameter head for slots.

The head turns per-Gaussian features into the common output interface:

    mu / scale / quat / opacity / rgb

GSODModel uses two decoders with this head: one for the object slots and one
for the background slot, which differ in their SlotFrame bounds. Slot ids
here are local to the decoder, 0..K-1 over the slots it decodes.

The first three values are offsets inside each slot's box (``SlotFrame``)
in the *azimuth frame of the input view* (yaw-rotated so view 0 looks along
+x).
"""

from __future__ import annotations

import math

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


class SlotFrame(nn.Module):
    """slot latent -> (centre, extent), bounded by dataset priors.

    Everything is predicted from the slot latent alone: centre xy, softly
    bounded to [-fg_xy, fg_xy]; centre z in [fg_z[0], fg_z[1]]; extent in
    [fg_extent[0], fg_extent[1]], log-uniform in the predicted value and
    initialised at fg_extent_init. Gaussian centres: centre + extent *
    tanh(offset).

    Ray anchoring (``self.ray`` and ``self.anchor_weight``, set by GSODModel
    for object slots): the ray of view 0 through the centroid of the slot's
    2D mask gives where the slot is seen; the anchored centre is where that
    ray reaches the predicted height z, plus a learned residual of at most
    anchor_residual. The centre used is (1 - w) * free + w * anchored, so the
    coarse 2D decomposition places the 3D frames while the residual leaves
    the 3D reconstruction room to correct them. Without a ray (or w = 0) the
    centre is the free prediction.

    The bounds are the structural prior (uORF's locality): object frames are
    object-sized and inside the placement area, so the floor and everything
    far away can only be drawn by the background slot, whose frame spans the
    scene out beyond the cameras (``from_scene_config(..., background=True)``).
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        *,
        fg_xy: float = 2.1,
        fg_z: tuple[float, float] = (0.0, 0.8),
        fg_extent: tuple[float, float] = (0.1, 0.75),
        fg_extent_init: float = 0.5,
        anchor_residual: float = 0.5,
    ):
        super().__init__()
        # How far the learned residual may move an anchored centre in xy
        # away from the point where the slot's ray reaches its height.
        self.anchor_residual = float(anchor_residual)
        # Set by GSODModel for the current forward pass: rays of view 0
        # (origin (3,), directions (K, 3)) in the view-0 azimuth frame, and
        # the weight of the anchored centre.
        self.ray: tuple[torch.Tensor, torch.Tensor] | None = None
        self.anchor_weight = 0.0
        self.fg_xy = float(fg_xy)
        self.fg_z = (float(fg_z[0]), float(fg_z[1]))
        self.fg_extent = (float(fg_extent[0]), float(fg_extent[1]))

        self.fg = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 8)
        )
        log_lo, log_hi = math.log(self.fg_extent[0]), math.log(self.fg_extent[1])
        with torch.no_grad():
            # Keep default weight scale so different slot latents map to
            # different centres from the start (no x0.1 here).
            self.fg[-1].bias.zero_()
            # Outputs 6:8, the anchored centre's residual, start at zero (on
            # the ray); they are separate from the free centre (0:2), which
            # is an absolute position and would saturate the residual's tanh.
            self.fg[-1].weight[6:8].zero_()
            self.fg[-1].bias[3:6] = _logit(
                (math.log(fg_extent_init) - log_lo) / (log_hi - log_lo)
            )

    @classmethod
    def from_scene_config(
        cls, z_dim: int, gen: dict, background: bool = False, **kwargs
    ) -> "SlotFrame":
        """Priors from a SceneGenConfig dict (meta.json["config"]).

        Object frames: up to an object's size (tanh rarely reaches 1, so the
        extent must exceed the half size). Background frame: from the scene's
        size out beyond the farthest camera, so it can surround every view.
        """
        r_max = float(gen["radius_range"][1])
        fg_xy = float(gen["placement_radius"]) + 0.3
        floor_z = float(gen.get("floor_z", 0.0))
        if background:
            reach = 3.0 * (
                float(gen["distance_range"][1]) + float(gen.get("target_height", 0.4))
            )
            extent, extent_init = (fg_xy, reach), math.sqrt(fg_xy * reach)
        else:
            extent, extent_init = (0.1, 1.35 * r_max), min(0.5, 1.2 * r_max)
        return cls(
            z_dim,
            fg_xy=fg_xy,
            fg_z=(floor_z, floor_z + r_max + 0.25),
            fg_extent=extent,
            fg_extent_init=extent_init,
            **kwargs,
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: (K, C) slots -> centre (K, 3), extent (K, 3)."""
        out = self.fg(slots)
        z_lo, z_hi = self.fg_z
        z = z_lo + (z_hi - z_lo) * torch.sigmoid(out[:, 2:3])
        xy = out[:, 0:2]
        w = self.anchor_weight
        if self.ray is not None and w > 0:
            origin, direction = self.ray
            # Rays at or above the horizon never come down to z; keep them
            # finite (the xy bound below then clamps the result).
            dz = direction[:, 2:3].clamp(max=-1e-2)
            t = ((z - origin[2]) / dz).clamp(min=0.0)
            anchored = origin[0:2] + t * direction[:, 0:2]
            anchored = anchored + self.anchor_residual * torch.tanh(out[:, 6:8])
            xy = (1.0 - w) * xy + w * anchored
        xy = self.fg_xy * torch.tanh(xy / self.fg_xy)
        log_lo, log_hi = math.log(self.fg_extent[0]), math.log(self.fg_extent[1])
        extent = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(out[:, 3:6]))
        return torch.cat([xy, z], dim=-1), extent


class GaussianParameterHead(nn.Module):
    """Project per-Gaussian features (in_dim) to the 14 raw parameters and
    decode them into the common Gaussian dict."""

    def __init__(
        self,
        frame: SlotFrame,
        in_dim: int,
        hidden_dim: int,
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
