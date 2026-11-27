"""Object decoder: one local Splatter Image per object slot, as if each slot reconstructed its
object alone.

1. window   from the slot's attention: a square around its centre, WINDOW_STD x its spread
            (at least MIN_HALF of the image); resampled to WINDOW x WINDOW points of view 0
2. anchor   the attention-weighted mean of where the view-0 token rays meet the floor; its camera
            depth is where the slot's Gaussians start (a start only)
3. Gaussians  M per window point, along the point's ray (+ a 3D offset):
    geometry    depth, offset, opacity = MLP(slot, PE(point in the window)): the shape in the
                window's own coordinates, no pixel features
    appearance  scale, rotation, colour residual, sh1 = MLP(slot, U-Net features at the point);
                colour = view 0's colour there + the residual
    A slot owns its Gaussians and makes them transparent where it is not. Depth: up to where the
    ray meets the floor z = 0. Window and anchor carry no gradient. Scene frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.background import FLOOR_HALF
from gsod.model.encoders import UNetEncoder
from gsod.model.layers import logit, positional_encoding, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions

WINDOW = 48  # window points per side
WINDOW_STD, MIN_HALF = 2.5, 0.06  # window half side: this x the attention's spread, at least this x the image
ANCHOR_SHARPNESS = 2.0  # anchor weights: attention ** this
MAX_DEPTH = 12.0  # rays that never meet the floor stop here (scene units)
OPACITY_START = (0.5, 0.1)  # the first Gaussian of a point, the others
DEPTH_STEP = 0.05  # Gaussian j starts j steps (fractions of the floor depth) behind the anchor
OFFSET_UNIT = 0.1  # 3D offset per unit of output (scene units)
SCALE_POINT = 0.5  # initial size: this fraction of a window point's footprint
PE_BANDS = 8
# Per Gaussian. Geometry: depth, offset (3), opacity. Appearance: log scale (3), quat (4), rgb residual (3), sh1 (9).
N_GEOMETRY, N_APPEARANCE = 5, 19


class Head(nn.Module):
    """Slots (B, K, C) + a per-point input (broadcast to B, K, P, D) -> each slot's output at each
    of its points (B, K, P, n_out)."""

    def __init__(self, z_dim: int, d_point: int, hidden: int, n_out: int):
        super().__init__()
        self.from_slot = nn.Linear(z_dim, hidden)
        self.from_point = nn.Linear(d_point, hidden)
        self.hidden = nn.Sequential(nn.ReLU(True), nn.Linear(hidden, hidden), nn.ReLU(True))
        self.out = nn.Linear(hidden, n_out)

    def forward(self, slots: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(self.from_slot(slots)[:, :, None] + self.from_point(point)))


class ObjectDecoder(nn.Module):
    """Object slots + their attention + view 0 -> Gaussians (B, K, WINDOW*WINDOW*M, ...)."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        gaussians_per_point: int,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float],
    ):
        super().__init__()
        self.m = m = gaussians_per_point
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        self.encoder = UNetEncoder(z_dim)  # pixel features of view 0, full resolution
        self.geometry = Head(z_dim, 2 + 4 * PE_BANDS, hidden_dim, m * N_GEOMETRY)
        self.appearance = Head(z_dim, z_dim, hidden_dim, m * N_APPEARANCE)
        with torch.no_grad():  # at first: at the anchor's depth, point-sized, view 0's colour
            out = self.geometry.out
            out.weight.zero_()
            bias = out.bias.view(m, N_GEOMETRY)
            bias.zero_()
            for j in range(m):
                bias[j, 0] = 4.0 * j * DEPTH_STEP  # ~ j steps behind the anchor (sigmoid slope <= 1/4)
                bias[j, 4] = logit(OPACITY_START[min(j, 1)])
            out = self.appearance.out
            out.weight.zero_()
            out.bias.zero_()
            out.bias.view(m, N_APPEARANCE)[:, 3] = 1.0  # quat w

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, image: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Object slots (B, K, C), their attention (B, K, s*s), view-0 image (B, 3, H, W), view 0's
        SceneFrames -> Gaussians (B, K, WINDOW*WINDOW*M, ...); scene frame."""
        h, w = self.image_size
        size = image.new_tensor([w, h])
        side = int(round(attn.shape[-1] ** 0.5))

        # Token pixel centres and rays; the attention's centre, spread and anchor (no gradient).
        idx = torch.arange(side, device=image.device, dtype=image.dtype) + 0.5
        tv, tu = torch.meshgrid(idx * h / side - 0.5, idx * w / side - 0.5, indexing="ij")
        token_uv = torch.stack([tu, tv], -1).reshape(-1, 2)  # (N, 2) pixels
        with torch.no_grad():
            p = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)  # (B, K, N)
            centre = p @ token_uv  # (B, K, 2)
            spread = (p @ token_uv.pow(2) - centre.pow(2)).clamp_min(0).sqrt().amax(-1)  # (B, K)
            half = (WINDOW_STD * spread).clamp_min(MIN_HALF * max(h, w))
            token_rays = self.rays(token_uv, frame)  # (B, N, 3)
            xy = (self.floor_depth(token_rays, frame)[..., None] * token_rays + frame.origin[:, None])[..., :2]
            hit = F.pad(xy.clamp(-FLOOR_HALF, FLOOR_HALF), (0, 1))  # (B, N, 3) on the floor
            sharp = (ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(-1)
            anchor = sharp @ hit  # (B, K, 3) on the floor
            anchor_depth = ((anchor - frame.origin[:, None]) * frame.rot[:, None, :, 2]).sum(-1)  # (B, K)

        # Window points: local coordinates in [-1, 1], their view-0 pixels and rays.
        g = (torch.arange(WINDOW, device=image.device, dtype=image.dtype) + 0.5) / WINDOW * 2 - 1
        gv, gu = torch.meshgrid(g, g, indexing="ij")
        local = torch.stack([gu, gv], -1).reshape(-1, 2)  # (P, 2)
        uv = centre[:, :, None] + half[..., None, None] * local  # (B, K, P, 2) pixels
        b, k, n_points = uv.shape[:3]
        rays = self.rays(uv.flatten(1, 2), frame).unflatten(1, (k, n_points))  # (B, K, P, 3)
        floor = self.floor_depth(rays.flatten(1, 2), frame).unflatten(1, (k, n_points))  # (B, K, P)

        def sample(x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B, K, P, C) at the window points
            grid = ((uv + 0.5) / size * 2 - 1).flatten(1, 2)[:, :, None]  # (B, K*P, 1, 2)
            out = F.grid_sample(x, grid, padding_mode="border", align_corners=False)[..., 0]
            return out.transpose(1, 2).unflatten(1, (k, n_points))

        geometry = self.geometry(slots, positional_encoding(local, PE_BANDS)[None, None])
        geometry = geometry.unflatten(-1, (self.m, N_GEOMETRY))  # (B, K, P, M, N_GEOMETRY)
        appearance = self.appearance(slots, sample(self.encoder(image))).unflatten(-1, (self.m, N_APPEARANCE))

        # Depth along each point's ray, up to the floor, starting at the anchor's depth.
        start = torch.logit((anchor_depth[..., None] / floor).clamp(0.02, 0.98))  # (B, K, P)
        depth = floor[..., None] * torch.sigmoid(start[..., None] + geometry[..., 0])  # (B, K, P, M)
        footprint = depth * (2 * half[..., None, None] / WINDOW) / (w * self.focal_ratio[0])  # a point's width
        colour = sample(image)[..., None, :]  # (B, K, P, 1, 3)
        gaussians = {
            "mu": frame.origin[:, None, None, None]
            + depth[..., None] * rays[..., None, :]
            + OFFSET_UNIT * geometry[..., 1:4],
            "scale": SCALE_POINT * footprint[..., None] * torch.exp(appearance[..., 0:3]),
            "quat": F.normalize(appearance[..., 3:7], dim=-1),
            "opacity": torch.sigmoid(geometry[..., 4]),
            "rgb": shift_colour(colour, appearance[..., 7:10]),
            "sh1": appearance[..., 10:].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(2, 3) for name, value in gaussians.items()}

    def rays(self, uv: torch.Tensor, frame: SceneFrame) -> torch.Tensor:
        """View-0 pixels (..., N, 2) -> rays (B, N, 3), scene frame, camera z = 1."""
        cam = camera_directions(uv[..., 0], uv[..., 1], self.image_size, self.focal_ratio)
        return cam @ frame.rot.mT

    @staticmethod
    def floor_depth(rays: torch.Tensor, frame: SceneFrame) -> torch.Tensor:
        """Rays (B, N, 3) -> camera depth where they meet the floor (B, N), at most MAX_DEPTH."""
        return (-frame.origin[:, None, 2] / rays[..., 2].clamp(max=-1e-6)).clamp(max=MAX_DEPTH)
