"""Pixel decoder (Splatter Image style): M Gaussians per view-0 pixel, placed by depth along the
pixel's ray (+ a 3D offset), so the first lies on the visible surface and the others can reach
the hidden side.

Geometry comes from the slots, appearance from the pixels:
    geometry    depth, offset, opacity = MLP(slot, PE(pixel position)); no pixel features, so
                only the slots shape the scene and the grouping matters
    appearance  scale, rotation, colour residual, sh1 = MLP(slot, pixel features); colour = the
                pixel's own colour + the residual
Every slot predicts the pixel's Gaussians as if the pixel were its own; the predictions are mixed
by the slots' attention at the pixel, which is also the Gaussians' (soft) owner. Depth: up to
where the ray meets the floor z = 0. Scene frame.
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
OPACITY_START = (0.9, 0.1)  # the first Gaussian, the others
OFFSET_UNIT = 0.1  # 3D offset per unit of output (scene units)
SCALE_PIXEL = 0.5  # initial size: this fraction of the pixel's footprint
PE_BANDS = 8
# Per Gaussian. Geometry: depth, offset (3), opacity. Appearance: log scale (3), quat (4), rgb residual (3), sh1 (9).
N_GEOMETRY, N_APPEARANCE = 5, 19


def _head(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.ReLU(True), nn.Linear(d_in, hidden), nn.ReLU(True), nn.Linear(hidden, d_out))


class PixelDecoder(nn.Module):
    """Slots + their attention + view 0 -> Gaussians (B, H*W*M, ...) and owners (B, H*W*M, K)."""

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
        # Geometry: slot + pixel position.
        self.slot_to_geometry = nn.Linear(z_dim, hidden_dim)
        self.position_to_geometry = nn.Linear(2 + 4 * PE_BANDS, hidden_dim)
        self.geometry = _head(hidden_dim, hidden_dim, m * N_GEOMETRY)
        # Appearance: slot + pixel features.
        self.encoder = UNetEncoder(z_dim)  # pixel features, full resolution
        self.slot_to_appearance = nn.Linear(z_dim, hidden_dim)
        self.pixel_to_appearance = nn.Linear(z_dim, hidden_dim)
        self.appearance = _head(hidden_dim, hidden_dim, m * N_APPEARANCE)
        with torch.no_grad():  # at first: on the ray, pixel-sized, the pixel's colour
            out = self.geometry[-1]
            out.weight.zero_()
            bias = out.bias.view(m, N_GEOMETRY)
            bias.zero_()
            for j in range(m):
                bias[j, 0] = logit(min(DEPTH_START + j * DEPTH_STEP, 0.97))
                bias[j, 4] = logit(OPACITY_START[min(j, 1)])
            out = self.appearance[-1]
            out.weight.zero_()
            out.bias.zero_()
            out.bias.view(m, N_APPEARANCE)[:, 3] = 1.0  # quat w

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, image: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Slots (B, K, C), their attention (B, K, s*s; slot 0 the background), view-0 image
        (B, 3, H, W), view 0's SceneFrames -> Gaussians (B, H*W*M, ...), owners (B, H*W*M, K)."""
        h, w = self.image_size
        side = int(round(attn.shape[-1] ** 0.5))
        share = F.interpolate(attn.unflatten(2, (side, side)), size=(h, w), mode="bilinear", align_corners=False)
        share = share.flatten(2)
        share = share / share.sum(1, keepdim=True).clamp_min(1e-6)  # (B, K, P)

        # Each slot's take on each pixel, mixed by the slots' share of it (before the last, linear layer).
        def mixed(head: nn.Sequential, per_slot: torch.Tensor, per_pixel: torch.Tensor) -> torch.Tensor:
            hidden = head[:-1](per_slot[:, :, None] + per_pixel[:, None])  # (B, K, P, h)
            return head[-1]((share[..., None] * hidden).sum(1)).unflatten(-1, (self.m, -1))  # (B, P, M, n)

        v, u = torch.meshgrid(
            torch.arange(h, device=image.device, dtype=image.dtype),
            torch.arange(w, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        position = torch.stack([u / (w - 1), v / (h - 1)], -1).reshape(-1, 2) * 2 - 1  # (P, 2) in [-1, 1]
        geometry = mixed(
            self.geometry,
            self.slot_to_geometry(slots),
            self.position_to_geometry(positional_encoding(position, PE_BANDS)),
        )
        pixel = self.encoder(image).flatten(2).transpose(1, 2)  # (B, P, C)
        appearance = mixed(self.appearance, self.slot_to_appearance(slots), self.pixel_to_appearance(pixel))

        # Depth along the pixel's ray, up to the floor; camera z = 1 rays, so depth is camera depth.
        rays = camera_directions(u, v, self.image_size, self.focal_ratio).reshape(-1, 3) @ frame.rot.mT  # (B, P, 3)
        floor = (-frame.origin[:, None, 2] / rays[..., 2].clamp(max=-1e-6)).clamp(max=MAX_DEPTH)  # (B, P)
        depth = floor[..., None] * torch.sigmoid(geometry[..., 0])  # (B, P, M)
        footprint = depth / (w * self.focal_ratio[0])  # a pixel's width at that depth
        colour = image.flatten(2).transpose(1, 2)[:, :, None]  # (B, P, 1, 3)
        gaussians = {
            "mu": frame.origin[:, None, None] + depth[..., None] * rays[:, :, None] + OFFSET_UNIT * geometry[..., 1:4],
            "scale": SCALE_PIXEL * footprint[..., None] * torch.exp(appearance[..., 0:3]),
            "quat": F.normalize(appearance[..., 3:7], dim=-1),
            "opacity": torch.sigmoid(geometry[..., 4]),
            "rgb": shift_colour(colour, appearance[..., 7:10]),
            "sh1": appearance[..., 10:].unflatten(-1, (3, 3)),
        }
        owner = share.transpose(1, 2)[:, :, None].expand(-1, -1, self.m, -1)  # (B, P, M, K)
        return {name: value.flatten(1, 2) for name, value in gaussians.items()}, owner.flatten(1, 2)
