"""Refiner: each object Gaussian reads view 0 around its pixel (deformable sampling) -> a refined
q, decoded on by the object decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import SceneFrame

N_POINTS = 4  # points each Gaussian samples around its view-0 pixel


def sample_own_pixels(
    mu: torch.Tensor,
    offsets: torch.Tensor,
    attn: torch.Tensor,
    enc: torch.Tensor,
    image: torch.Tensor,
    frame: SceneFrame,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> torch.Tensor:
    """Gaussian centres mu (B, K, G, 3), offsets (B, K, G, M, 2) in pixels, slot attention maps
    (B, K, s, s), encoder map (B, C, s, s), view-0 image (B, 3, H, W) -> at each centre's view-0
    pixel + offsets: [features, rgb, its slot's ownership] (B, K, G, M, C + 4); 0 outside the image."""
    b, k, g, m = offsets.shape[:4]
    h, w = image_size
    f = mu.new_tensor([focal_ratio[0] * w, focal_ratio[1] * h])
    size = mu.new_tensor([w, h])
    # Project the centres into view 0 (pixel centres at integers).
    cam = (mu.flatten(1, 2).detach() - frame.origin[:, None]) @ frame.rot  # (B, N, 3)
    uv = f * cam[..., :2] / cam[..., 2:].clamp_min(1e-4) + (size - 1) / 2
    points = uv[:, :, None] + offsets.flatten(1, 2)  # (B, N, M, 2)
    grid = (points + 0.5) / size * 2 - 1

    def sample(x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, N, M, C)
        return F.grid_sample(x, grid, align_corners=False).permute(0, 2, 3, 1)

    own_slot = torch.arange(k, device=mu.device).repeat_interleave(g)  # slot of each Gaussian
    ownership = sample(attn).gather(-1, own_slot.view(1, -1, 1, 1).expand(b, -1, m, 1))
    return torch.cat([sample(enc), sample(image), ownership], -1).unflatten(1, (k, g))


class Refiner(nn.Module):
    """q (B*K, G, d) + what each Gaussian sees in view 0 -> refined q (B*K, G, d)."""

    def __init__(self, z_dim: int, d: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]):
        super().__init__()
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        # Sampled [features, rgb, ownership] -> q (zero at first).
        self.pixel_to_q = nn.Linear(z_dim + 4, d)
        nn.init.zeros_(self.pixel_to_q.weight)
        nn.init.zeros_(self.pixel_to_q.bias)
        # Deformable sampling: q -> N_POINTS offsets (px) and their weights.
        self.to_offsets = nn.Linear(d, N_POINTS * 2)
        self.to_weights = nn.Linear(d, N_POINTS)
        angle = torch.arange(N_POINTS) * 2 * math.pi / N_POINTS
        with torch.no_grad():  # a 1 px ring at first: points that start equal would stay equal
            self.to_offsets.weight.zero_()
            self.to_offsets.bias.copy_(torch.stack([angle.cos(), angle.sin()], -1).flatten())
            self.to_weights.weight.zero_()
            self.to_weights.bias.zero_()

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        image: torch.Tensor,
        enc: torch.Tensor,
        attn: torch.Tensor,
        frame: SceneFrame,
    ) -> torch.Tensor:
        """q (B*K, G, d), Gaussian centres mu (B, K, G, 3), view-0 image (B, 3, H, W), pixel features
        (B, C, s, s), slot attention maps (B, K, s, s) -> q + what each Gaussian sees (B*K, G, d)."""
        shape = mu.shape[:2]
        offsets = self.to_offsets(q).unflatten(-1, (N_POINTS, 2)).unflatten(0, shape)  # (B, K, G, M, 2)
        weights = self.to_weights(q).softmax(-1).unflatten(0, shape)  # (B, K, G, M)
        samples = sample_own_pixels(mu, offsets, attn, enc, image, frame, self.image_size, self.focal_ratio)
        pixel_info = (weights[..., None] * samples).sum(-2).flatten(0, 1)  # (B*K, G, C + 4)
        return q + self.pixel_to_q(pixel_info)
