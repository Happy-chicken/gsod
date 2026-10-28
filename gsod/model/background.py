"""Static background: a shell of learnable Gaussians around the scene.

The background is the same in every image (a uniform colour), so it is not a
slot and is not decoded from anything: its Gaussians are plain learnable
parameters, shared by all scenes, in the world frame. They live on a sphere of
radius ``radius`` around the origin; every camera is inside it, so the shell
can cover every viewing direction.
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
    ):
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

    @property
    def n_gaussians(self) -> int:
        return self.direction.shape[0]

    def forward(self) -> dict[str, torch.Tensor]:
        lo, hi = self.scale_range
        return {
            "mu": self.radius * F.normalize(self.direction, dim=-1, eps=1e-6),
            "scale": self.radius * (lo + (hi - lo) * torch.sigmoid(self.scale_raw)),
            "quat": F.normalize(self.quat_raw, dim=-1),
            "opacity": torch.sigmoid(self.opacity_raw),
            "rgb": torch.sigmoid(self.rgb_raw),
        }
