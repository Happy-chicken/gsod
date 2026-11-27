"""Pixel decoder (Splatter Image style), one splatter image per slot: every slot puts M Gaussians
on every view-0 pixel, placed by depth along the pixel's ray (+ a 3D offset). A slot owns its
Gaussians; where it is not, it should make them transparent, so the grouping comes from the
rendering (who is opaque, who is in front), as in uORF.

Geometry comes from the slot, appearance from the pixels:
    geometry    depth, offset, opacity = MLP(slot, PE(pixel position)); no pixel features. The
                background's PE is low-frequency (smooth floor and walls, also behind the
                objects), the objects' high-frequency
    appearance  scale, rotation, colour residual, sh1 = MLP(slot, pixel features); colour = the
                pixel's own colour + the residual
Depth: up to where the ray meets the floor z = 0. Scene frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.encoders import UNetEncoder
from gsod.model.layers import logit, positional_encoding, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions

MAX_DEPTH = 12.0  # rays that never meet the floor stop here (scene units)
DEPTH_START, DEPTH_STEP = 0.85, 0.05  # Gaussian j starts at this fraction of the floor depth (+ j steps)
# Initial opacity: the background's first Gaussian, its others; every object Gaussian.
BACKGROUND_OPACITY_START, OBJECT_OPACITY_START = (0.9, 0.1), 0.1
OFFSET_UNIT = 0.1  # 3D offset per unit of output (scene units)
SCALE_PIXEL = 0.5  # initial size: this fraction of the pixel's footprint
BACKGROUND_PE_BANDS, OBJECT_PE_BANDS = 2, 8
# Per Gaussian. Geometry: depth, offset (3), opacity. Appearance: log scale (3), quat (4), rgb residual (3), sh1 (9).
N_GEOMETRY, N_APPEARANCE = 5, 19


class Head(nn.Module):
    """Slots (B, K, C) + a per-pixel input (B or 1, P, D) -> each slot's output at each pixel (B, K, P, n_out)."""

    def __init__(self, z_dim: int, d_pixel: int, hidden: int, n_out: int):
        super().__init__()
        self.from_slot = nn.Linear(z_dim, hidden)
        self.from_pixel = nn.Linear(d_pixel, hidden)
        self.hidden = nn.Sequential(nn.ReLU(True), nn.Linear(hidden, hidden), nn.ReLU(True))
        self.out = nn.Linear(hidden, n_out)

    def forward(self, slots: torch.Tensor, pixel: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(self.from_slot(slots)[:, :, None] + self.from_pixel(pixel)[:, None]))


class PixelDecoder(nn.Module):
    """Slots + view 0 -> each slot's Gaussians (B, K*H*W*M, ...) and their slot, one-hot (B, K*H*W*M, K)."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        gaussians_per_pixel: int,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float],
    ):
        super().__init__()
        self.m = m = gaussians_per_pixel
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        # Geometry: slot + pixel position; the background's smooth, the objects' detailed.
        self.background_geometry = Head(z_dim, 2 + 4 * BACKGROUND_PE_BANDS, hidden_dim, m * N_GEOMETRY)
        self.object_geometry = Head(z_dim, 2 + 4 * OBJECT_PE_BANDS, hidden_dim, m * N_GEOMETRY)
        # Appearance: slot + pixel features.
        self.encoder = UNetEncoder(z_dim)  # pixel features, full resolution
        self.appearance = Head(z_dim, z_dim, hidden_dim, m * N_APPEARANCE)
        with torch.no_grad():  # at first: on the ray, pixel-sized, the pixel's colour; objects faint
            for head, opacity in (
                (self.background_geometry, BACKGROUND_OPACITY_START),
                (self.object_geometry, (OBJECT_OPACITY_START,) * 2),
            ):
                head.out.weight.zero_()
                bias = head.out.bias.view(m, N_GEOMETRY)
                bias.zero_()
                for j in range(m):
                    bias[j, 0] = logit(min(DEPTH_START + j * DEPTH_STEP, 0.97))
                    bias[j, 4] = logit(opacity[min(j, 1)])
            out = self.appearance.out
            out.weight.zero_()
            out.bias.zero_()
            out.bias.view(m, N_APPEARANCE)[:, 3] = 1.0  # quat w

    def forward(
        self, slots: torch.Tensor, image: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Slots (B, K, C; slot 0 the background), view-0 image (B, 3, H, W), view 0's SceneFrames
        -> Gaussians (B, K*H*W*M, ...), their slot one-hot (B, K*H*W*M, K)."""
        b, k = slots.shape[:2]
        h, w = self.image_size
        v, u = torch.meshgrid(
            torch.arange(h, device=image.device, dtype=image.dtype),
            torch.arange(w, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        position = torch.stack([u / (w - 1), v / (h - 1)], -1).reshape(-1, 2) * 2 - 1  # (P, 2) in [-1, 1]
        geometry = torch.cat(
            [
                self.background_geometry(slots[:, :1], positional_encoding(position, BACKGROUND_PE_BANDS)[None]),
                self.object_geometry(slots[:, 1:], positional_encoding(position, OBJECT_PE_BANDS)[None]),
            ],
            1,
        ).unflatten(-1, (self.m, N_GEOMETRY))  # (B, K, P, M, N_GEOMETRY)
        pixel = self.encoder(image).flatten(2).transpose(1, 2)  # (B, P, C)
        appearance = self.appearance(slots, pixel).unflatten(-1, (self.m, N_APPEARANCE))  # (B, K, P, M, N_APPEARANCE)

        # Depth along the pixel's ray, up to the floor; camera z = 1 rays, so depth is camera depth.
        rays = camera_directions(u, v, self.image_size, self.focal_ratio).reshape(-1, 3) @ frame.rot.mT  # (B, P, 3)
        floor = (-frame.origin[:, None, 2] / rays[..., 2].clamp(max=-1e-6)).clamp(max=MAX_DEPTH)  # (B, P)
        rays, floor = rays[:, None, :, None], floor[:, None, :, None]  # (B, 1, P, 1, ...)
        depth = floor * torch.sigmoid(geometry[..., 0])  # (B, K, P, M)
        footprint = depth / (w * self.focal_ratio[0])  # a pixel's width at that depth
        colour = image.flatten(2).transpose(1, 2)[:, None, :, None]  # (B, 1, P, 1, 3)
        origin = frame.origin[:, None, None, None]
        gaussians = {
            "mu": origin + depth[..., None] * rays + OFFSET_UNIT * geometry[..., 1:4],
            "scale": SCALE_PIXEL * footprint[..., None] * torch.exp(appearance[..., 0:3]),
            "quat": F.normalize(appearance[..., 3:7], dim=-1),
            "opacity": torch.sigmoid(geometry[..., 4]),
            "rgb": shift_colour(colour, appearance[..., 7:10]),
            "sh1": appearance[..., 10:].unflatten(-1, (3, 3)),
        }
        slot = torch.arange(k, device=image.device).repeat_interleave(h * w * self.m)
        owner = F.one_hot(slot, k).to(image.dtype).expand(b, -1, -1)  # (B, K*P*M, K)
        return {name: value.flatten(1, 3) for name, value in gaussians.items()}, owner
