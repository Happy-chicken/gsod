"""Gaussian parameter head for object slots.

The head turns per-Gaussian features into the common output interface:

    mu / scale / quat / opacity / rgb

Decoders only see object slots; the background is a separate static module
(gsod.model.background.BackgroundShell). Slot ids here are local, 0..K-1 over
the object slots; GSODModel shifts them by one so that 0 is the background.

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
    """object-slot latent -> (centre, extent), bounded by dataset priors.

    centre xy in [-fg_xy, fg_xy], centre z in [fg_z[0], fg_z[1]],
    extent in [fg_extent[0], fg_extent[1]], initialised at fg_extent_init.
    Gaussian centres: centre + extent * tanh(offset).

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
        fg_extent: tuple[float, float] = (0.1, 0.75),
        fg_extent_init: float = 0.5,
        anchor_height: float = 0.45,
        anchor_residual: float = 0.5,
    ):
        super().__init__()
        # Height of the plane the attention centroid ray is intersected with
        # (typical object-centre height), and how far the learned residual may
        # move a frame centre away from its anchor in xy.
        self.anchor_height = float(anchor_height)
        self.anchor_residual = float(anchor_residual)
        # Set by GSODModel for the current forward pass: (K, 3) points in the
        # view-0 azimuth frame, or None to fall back to free centres.
        self.anchor: torch.Tensor | None = None
        self.fg_xy = float(fg_xy)
        self.fg_z = (float(fg_z[0]), float(fg_z[1]))
        self.fg_extent = (float(fg_extent[0]), float(fg_extent[1]))

        self.fg = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(True), nn.Linear(hidden_dim, 6)
        )
        lo, hi = self.fg_extent
        with torch.no_grad():
            # Keep default weight scale so different slot latents map to
            # different centres from the start (no x0.1 here).
            self.fg[-1].bias.zero_()
            self.fg[-1].bias[3:6] = _logit((fg_extent_init - lo) / (hi - lo))

    @classmethod
    def from_scene_config(cls, z_dim: int, gen: dict, **kwargs) -> "SlotFrame":
        """Priors from a SceneGenConfig dict (meta.json["config"])."""
        r_min, r_max = (float(x) for x in gen["radius_range"])
        place = float(gen["placement_radius"])
        floor_z = float(gen.get("floor_z", 0.0))
        return cls(
            z_dim,
            fg_xy=place + 0.3,
            fg_z=(floor_z, floor_z + r_max + 0.25),
            # tanh rarely reaches 1, so the extent must exceed the half size.
            fg_extent=(0.1, 1.35 * r_max),
            fg_extent_init=min(0.5, 1.2 * r_max),
            # Objects rest on z = floor_z (drawn or not).
            anchor_height=floor_z + 0.5 * (r_min + r_max),
            **kwargs,
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """slots: (K, C) object slots -> centre (K, 3), extent (K, 3)."""
        out = self.fg(slots)
        if self.anchor is not None:
            # Centre = attention centroid on the ground-parallel plane + a
            # bounded learned residual. Slots attending to different objects
            # get different frames, and the rendering loss reaches the
            # attention maps through the centroid.
            xy = self.anchor[:, 0:2] + self.anchor_residual * torch.tanh(out[:, 0:2])
        else:
            xy = out[:, 0:2]
        xy = self.fg_xy * torch.tanh(xy / self.fg_xy)
        z_lo, z_hi = self.fg_z
        z = z_lo + (z_hi - z_lo) * torch.sigmoid(out[:, 2:3])
        e_lo, e_hi = self.fg_extent
        extent = e_lo + (e_hi - e_lo) * torch.sigmoid(out[:, 3:6])
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
        """raw: (M, in_dim); slots: (K, C) object slots; slot_ids: (M,) in 0..K-1.

        Returns the Gaussian dict and (M, K) one-hot object-slot probabilities.
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
