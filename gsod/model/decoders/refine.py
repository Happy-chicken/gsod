"""Feedback for the refine round: view 0's render -> a feature map (read per
token by the slot refine) and samples of it at each object Gaussian's pixel
(read by the continued object decoding)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.utils.geometry import SceneFrame

# Channels of the feedback map.
FEEDBACK_DIM = 32


class Feedback(nn.Module):
    """[input - render, render, alpha] (B, 7, H, W) -> feedback map (B, 32, H/2, W/2)."""

    def __init__(self, enabled: bool = True):
        """enabled=False: an all-zero map (encoder features only, for ablation)."""
        super().__init__()
        self.enabled = enabled
        self.cnn = nn.Sequential(
            nn.Conv2d(7, FEEDBACK_DIM, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(FEEDBACK_DIM, FEEDBACK_DIM, 3, padding=1), nn.ReLU(),
        )

    def forward(self, image: torch.Tensor, render: dict[str, torch.Tensor]) -> torch.Tensor:
        """View 0: image (B, 3, H, W), render {"rgb", "alpha"} (B, C, H, W)."""
        x = torch.cat([image - render["rgb"], render["rgb"], render["alpha"]], 1).detach()
        out = self.cnn(x)
        return out if self.enabled else torch.zeros_like(out)


def sample_at_gaussians(
    maps: list[torch.Tensor],
    mu: torch.Tensor,
    render: dict[str, torch.Tensor],
    frame: SceneFrame,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> torch.Tensor:
    """View-0 maps (B, C_i, h, w), Gaussian centres mu (B, K, G, 3, scene frame),
    view-0 render {"depth", "alpha"} -> per Gaussian [maps..., occlusion]
    (B, K, G, sum C_i + 1); 0 where a Gaussian falls outside the image."""
    k, g = mu.shape[1:3]
    h, w = image_size
    # Project each centre into view 0 (inverse of the token rays in init_attention_anchor).
    cam = (mu.flatten(1, 2).detach() - frame.origin[:, None]) @ frame.rot  # (B, N, 3)
    z = cam[..., 2:]
    zs = z.clamp_min(1e-4)
    u = focal_ratio[0] * w * cam[..., :1] / zs + (w - 1) / 2
    v = focal_ratio[1] * h * cam[..., 1:2] / zs + (h - 1) / 2
    grid = torch.cat([(u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1], -1)[:, :, None]  # (B, N, 1, 2)

    def sample(x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, N, C)
        return F.grid_sample(x, grid, align_corners=False)[..., 0].transpose(1, 2)

    # Occlusion: the Gaussian's depth - the rendered depth at its pixel (scene units).
    depth = render["depth"] / render["alpha"].clamp_min(1e-4) / frame.scale[:, None, None, None]
    occlusion = z - sample(depth.detach())
    return torch.cat([*(sample(m) for m in maps), occlusion], -1).unflatten(1, (k, g))
