"""Refiner: view 0 rendered -> what each object Gaussian sees at its pixel -> a
correction of its decoder feature q (then ObjectHead again)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.layers import mlp
from gsod.utils.geometry import SceneFrame

# Channels of the feedback feature map.
FEEDBACK_DIM = 32


class Refiner(nn.Module):
    def __init__(
        self, z_dim: int, d_model: int, image_size: tuple[int, int], focal_ratio: tuple[float, float],
        feedback: bool = True,
    ):
        """feedback=False: encoder features only (no rendered residual)."""
        super().__init__()
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        self.feedback = feedback
        # [input - render (3), render (3), alpha (1)] at H x W -> H/2 x W/2.
        self.cnn = nn.Sequential(
            nn.Conv2d(7, FEEDBACK_DIM, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(FEEDBACK_DIM, FEEDBACK_DIM, 3, padding=1), nn.ReLU(),
        )
        # [q, feedback, encoder, occlusion, inside] -> q correction; zero at first.
        self.mlp = mlp(d_model + FEEDBACK_DIM + z_dim + 2, d_model, d_model, norm=True)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        image: torch.Tensor,
        render: dict[str, torch.Tensor],
        enc: torch.Tensor,
        frame: SceneFrame,
    ) -> torch.Tensor:
        """q (B, K, G, d), Gaussian centres mu (B, K, G, 3, scene frame), view 0:
        image (B, 3, H, W), render {"rgb", "depth", "alpha"} (B, C, H, W),
        encoder map (B, z_dim, s, s) -> refined q (B, K, G, d)."""
        b, k, g, d = q.shape
        h, w = self.image_size
        # Project each centre into view 0 (inverse of the token rays in init_attention_anchor).
        cam = (mu.flatten(1, 2).detach() - frame.origin[:, None]) @ frame.rot  # (B, N, 3)
        z = cam[..., 2:]
        zs = z.clamp_min(1e-4)
        u = self.focal_ratio[0] * w * cam[..., :1] / zs + (w - 1) / 2
        v = self.focal_ratio[1] * h * cam[..., 1:2] / zs + (h - 1) / 2
        grid = torch.cat([(u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1], -1)[:, :, None]  # (B, N, 1, 2)
        inside = ((grid[:, :, 0].abs() <= 1).all(-1, keepdim=True) & (z > 0)).to(q.dtype)

        def sample(x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, N, C); 0 outside
            return F.grid_sample(x, grid, align_corners=False)[..., 0].transpose(1, 2)

        # Occlusion: the Gaussian's depth - the rendered depth there (scene units).
        depth = render["depth"] / render["alpha"].clamp_min(1e-4) / frame.scale[:, None, None, None]
        occlusion = z - sample(depth.detach())
        if self.feedback:
            feedback = torch.cat([image - render["rgb"], render["rgb"], render["alpha"]], 1).detach()
            seen = sample(self.cnn(feedback))
        else:
            seen = q.new_zeros(b, k * g, FEEDBACK_DIM)
        x = torch.cat([q.flatten(1, 2), seen, sample(enc), occlusion, inside], -1)
        return q + self.mlp(x).unflatten(1, (k, g))
