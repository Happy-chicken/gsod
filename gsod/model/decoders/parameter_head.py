"""Gaussian parameter head for slots.

The head turns per-Gaussian features into the common output interface:

    mu / scale / quat / opacity / rgb

Every slot is decoded the same way; there is no separate background model.
Slot ids are 0..K-1.

By default the first three values are absolute centres. An optional
``SlotFrame`` plugin interprets them as offsets inside a per-slot box in the
*azimuth frame of the input view* (yaw-rotated so view 0 looks along +x).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

PARAMS_PER_GAUSSIAN = 14  # mu/offset 3, log-scale 3, quat 4, opacity 1, rgb 3

LOG_SCALE_INIT = math.log(0.1)
OPACITY_INIT = -2.2  # sigmoid ~= 0.10
OFFSET_SPREAD = 0.8  # initial tanh(offset) ~ U(-0.8, 0.8) inside the frame


def init_raw_bias(
    bias: torch.Tensor,
    *,
    offset_spread: float = OFFSET_SPREAD,
    opacity: float = OPACITY_INIT,
) -> None:
    """Initialise an (n_gaussians, 14) bias view in place.

    Offsets get a fixed random spread, so every Gaussian starts at a different
    point of its slot frame instead of all collapsing onto the frame centre.
    """
    n = bias.shape[0]
    bias.zero_()
    if offset_spread > 0:
        u = torch.empty(n, 3, dtype=bias.dtype, device=bias.device)
        u.uniform_(-offset_spread, offset_spread)
        bias[:, 0:3] = torch.atanh(u)
    bias[:, 3:6] = LOG_SCALE_INIT
    bias[:, 6] = 1.0  # identity quaternion (w first, gsplat convention)
    bias[:, 10] = opacity


def _init_head(head: nn.Linear) -> None:
    """Shared projection head of the attention decoder.

    It has a single bias row for all Gaussians, so the spatial spread must come
    from the per-Gaussian queries. The offset rows therefore keep their default
    weight scale; only the other rows are shrunk.
    """
    with torch.no_grad():
        head.weight[3:].mul_(0.1)
        init_raw_bias(head.bias.view(1, PARAMS_PER_GAUSSIAN), offset_spread=0.0)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class SlotFrame(nn.Module):
    """slot latent -> (centre, extent), bounded by dataset priors.

    centre z in [fg_z[0], fg_z[1]] is predicted first. With a ray per slot
    (``self.ray``, set by GSODModel from the attention centroid), centre xy is
    where that ray reaches height z, plus a learned residual of at most
    anchor_residual; so the centre stays on the line of sight of what the slot
    attends to, whatever height is predicted. centre xy is softly bounded to
    [-fg_xy, fg_xy]. extent in [fg_extent[0], fg_extent[1]], log-uniform in
    the predicted value and initialised at fg_extent_init. Gaussian centres:
    centre + extent * tanh(offset).

    The extent range spans an object (~0.1) up to the reach of the scene
    (beyond the cameras), so the same slot model can draw a small object or
    the background around the cameras; the log scale keeps the small end as
    finely controllable as the large one.

    Defaults match ``SceneGenConfig`` defaults; use ``from_scene_config`` to
    derive them from a dataset's meta.json.
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 64,
        *,
        fg_xy: float = 2.1,
        fg_z: tuple[float, float] = (0.0, 0.8),
        fg_extent: tuple[float, float] = (0.1, 20.0),
        fg_extent_init: float = 0.5,
        anchor_residual: float = 0.5,
    ):
        super().__init__()
        # How far the learned residual may move a frame centre in xy away
        # from the point where the slot's ray reaches the predicted height.
        self.anchor_residual = float(anchor_residual)
        # Set by GSODModel for the current forward pass: rays through the
        # attention centroids in the view-0 azimuth frame, (origin (3,),
        # directions (K, 3)); None falls back to free centres.
        self.ray: tuple[torch.Tensor, torch.Tensor] | None = None
        self.fg_xy = float(fg_xy)
        self.fg_z = (float(fg_z[0]), float(fg_z[1]))
        self.fg_extent = (float(fg_extent[0]), float(fg_extent[1]))

        self.fg = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 6)
        )
        log_lo, log_hi = math.log(self.fg_extent[0]), math.log(self.fg_extent[1])
        with torch.no_grad():
            # Keep default weight scale so different slot latents map to
            # different centres from the start (no x0.1 here).
            self.fg[-1].bias.zero_()
            self.fg[-1].bias[3:6] = _logit(
                (math.log(fg_extent_init) - log_lo) / (log_hi - log_lo)
            )

    @classmethod
    def from_scene_config(cls, z_dim: int, gen: dict, **kwargs) -> "SlotFrame":
        """Priors from a SceneGenConfig dict (meta.json["config"])."""
        r_max = float(gen["radius_range"][1])
        place = float(gen["placement_radius"])
        floor_z = float(gen.get("floor_z", 0.0))
        # Beyond the farthest camera, so a slot can surround every view.
        reach = 3.0 * (
            float(gen["distance_range"][1]) + float(gen.get("target_height", 0.4))
        )
        return cls(
            z_dim,
            fg_xy=place + 0.3,
            fg_z=(floor_z, floor_z + r_max + 0.25),
            fg_extent=(0.1, reach),
            fg_extent_init=min(0.5, 1.2 * r_max),
            **kwargs,
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: (K, C) slots -> centre (K, 3), extent (K, 3)."""
        out = self.fg(slots)
        z_lo, z_hi = self.fg_z
        z = z_lo + (z_hi - z_lo) * torch.sigmoid(out[:, 2:3])
        if self.ray is not None:
            # Centre = where the attention-centroid ray reaches height z, plus
            # a bounded residual. Slots attending to different objects get
            # different frames; the rendering loss reaches the attention maps
            # through the ray and the predicted height through t.
            origin, direction = self.ray
            # Rays at or above the horizon never come down to z; keep them
            # finite (the xy bound below then clamps the result).
            dz = direction[:, 2:3].clamp(max=-1e-2)
            t = ((z - origin[2]) / dz).clamp(min=0.0)
            xy = origin[0:2] + t * direction[:, 0:2]
            xy = xy + self.anchor_residual * torch.tanh(out[:, 0:2])
        else:
            xy = out[:, 0:2]
        xy = self.fg_xy * torch.tanh(xy / self.fg_xy)
        log_lo, log_hi = math.log(self.fg_extent[0]), math.log(self.fg_extent[1])
        extent = torch.exp(log_lo + (log_hi - log_lo) * torch.sigmoid(out[:, 3:6]))
        return torch.cat([xy, z], dim=-1), extent


class GaussianParameterHead(nn.Module):
    """Decode per-Gaussian raw parameters into the common Gaussian dict.

    in_dim is 14 when the caller already produced raw Gaussian parameters
    (the MLP decoders). Attention decoders pass their feature width and the
    head projects it to 14 first.
    """

    def __init__(
        self,
        in_dim: int = PARAMS_PER_GAUSSIAN,
        hidden_dim: int = 128,
        frame: SlotFrame | None = None,
        frame_scale_min: float = 0.05,
        frame_scale_max: float = 0.5,
    ):
        super().__init__()
        self.frame_scale_min = float(frame_scale_min)
        self.frame_scale_max = float(frame_scale_max)
        if in_dim == PARAMS_PER_GAUSSIAN:
            self.project = nn.Identity()
        else:
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
        if raw.dim() == 3:
            raw = raw.reshape(-1, PARAMS_PER_GAUSSIAN)
        assert raw.shape[-1] == PARAMS_PER_GAUSSIAN
        assert raw.shape[0] == slot_ids.shape[0]

        if self.frame is None:
            mu = raw[:, 0:3]
            scale = torch.exp(raw[:, 3:6].clamp(-7.0, 0.0))
        else:
            centre, extent = self.frame(slots)
            c, e = centre[slot_ids], extent[slot_ids]
            mu = c + e * torch.tanh(raw[:, 0:3])
            rel_scale = self.frame_scale_min + (
                self.frame_scale_max - self.frame_scale_min
            ) * torch.sigmoid(raw[:, 3:6])
            scale = e * rel_scale
        gaussians = {
            "mu": mu,
            "scale": scale,
            "quat": F.normalize(raw[:, 6:10], dim=-1),
            "opacity": torch.sigmoid(raw[:, 10]),
            "rgb": torch.sigmoid(raw[:, 11:14]),
        }
        slot_probs = F.one_hot(slot_ids, slots.shape[0]).to(dtype=slots.dtype)
        return gaussians, slot_probs
