"""Background: a shell of Gaussians around the scene, coloured by the background slot.

Its geometry (positions, sizes, rotations, opacities) is plain learnable
parameters, shared by all scenes, in the world frame. The Gaussians live on a
sphere of radius ``radius`` around the origin; every camera is inside it, so
the shell can cover every viewing direction.

The colours depend on the scene through the background slot of slot
attention (uORF's background slot + background NeRF): a small MLP of
[background slot, Fourier(direction)] adds to the learned base colour. Its
last layer starts at zero, so the shell starts scene-independent.

Why the background cannot take the objects: the shell is far away (several
camera distances), so whatever it shows moves with the camera like a sky. It
can paint an object into view 0, but in the other views that paint lands on
the wrong pixels; only Gaussians near the centre (object slots) reproduce an
object in every view. A background slot that attends to objects therefore
costs reconstruction error in the novel views, which pushes the object tokens
back to the object slots.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def fibonacci_directions(n: int) -> torch.Tensor:
    """(n, 3) near-uniform unit vectors on the sphere."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    return torch.stack(
        [
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
        ],
        dim=-1,
    ).float()


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class BackgroundShell(nn.Module):
    """n learnable Gaussians on a sphere of radius ``radius``.

    Sizes are relative to the radius, in scale_range. The mean spacing on the
    sphere is sqrt(4*pi/n) * R (~0.11 R for n = 1024). The initial size follows
    n so that neighbours overlap and the shell starts closed:
    rel_scale_init = 0.07 * sqrt(1024 / n) (0.07 at n = 1024: simulated initial
    coverage over all pixels mean 0.97, min 0.93 at opacity 0.95). It must stay
    below scale_range[1], so n >= ~125 for the default range.
    """

    def __init__(
        self,
        n_gaussians: int = 1024,
        radius: float = 20.0,
        scale_range: tuple[float, float] = (0.02, 0.2),
        rel_scale_init: float | None = None,
        opacity_init: float = 0.95,
        color_init: tuple[float, float, float] = (0.5, 0.5, 0.5),
        cond_dim: int = 64,
        hidden_dim: int = 64,
        n_freqs: int = 4,
    ):
        """cond_dim: size of the background slot; hidden_dim: colour-MLP width;
        n_freqs: Fourier frequencies of the direction fed to the colour MLP."""
        super().__init__()
        self.radius = float(radius)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        lo, hi = self.scale_range
        n = int(n_gaussians)
        if rel_scale_init is None:
            rel_scale_init = 0.07 * math.sqrt(1024.0 / n)
        if not lo < rel_scale_init < hi:
            raise ValueError(
                f"{n} background Gaussians need initial size {rel_scale_init:.3f} R to "
                f"close the shell, outside scale_range {self.scale_range}; use more "
                "Gaussians or widen scale_range"
            )

        self.direction = nn.Parameter(fibonacci_directions(n))
        self.scale_raw = nn.Parameter(
            torch.full((n, 3), _logit((rel_scale_init - lo) / (hi - lo)))
        )
        self.quat_raw = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1))
        self.opacity_raw = nn.Parameter(torch.full((n,), _logit(opacity_init)))
        color = torch.tensor(color_init).clamp(1e-3, 1 - 1e-3)
        self.rgb_raw = nn.Parameter(torch.logit(color).repeat(n, 1))

        freqs = math.pi * 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.color_mlp = nn.Sequential(
            nn.Linear(cond_dim + 3 + 6 * n_freqs, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.color_mlp[-1].weight)
        nn.init.zeros_(self.color_mlp[-1].bias)

    @property
    def n_gaussians(self) -> int:
        return self.direction.shape[0]

    def forward(
        self, bg_slot: torch.Tensor, yaw: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """bg_slot: (D,) background slot; yaw: azimuth of view 0 (the frame the
        slot was encoded in). Returns the shell's Gaussians in the world frame."""
        lo, hi = self.scale_range
        direction = F.normalize(self.direction, dim=-1, eps=1e-6)
        # Directions in the view-0 azimuth frame (rotate by -yaw about +z).
        c, s = torch.cos(yaw), torch.sin(yaw)
        x, y, z = direction.unbind(-1)
        d_az = torch.stack([c * x + s * y, -s * x + c * y, z], dim=-1)
        angles = (d_az[..., None] * self.freqs).flatten(1)
        features = torch.cat(
            [
                bg_slot.expand(direction.shape[0], -1),
                d_az,
                angles.sin(),
                angles.cos(),
            ],
            dim=-1,
        )
        return {
            "mu": self.radius * direction,
            "scale": self.radius * (lo + (hi - lo) * torch.sigmoid(self.scale_raw)),
            "quat": F.normalize(self.quat_raw, dim=-1),
            "opacity": torch.sigmoid(self.opacity_raw),
            "rgb": torch.sigmoid(self.rgb_raw + self.color_mlp(features)),
        }
