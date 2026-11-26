"""Image encoder: pretrained DINOv2 (frozen), its patch features projected to z_dim."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DINO_NAME, DINO_DIM, PATCH = "dinov2_vits14", 384, 14
GRID_DIVISOR = 4  # DINO's patch grid is s / 4: 16 x 16 patches (224 px in) for s = 64
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet


class DinoEncoder(nn.Module):
    """B x 3 x H x W image in [0, 1] -> B x C x s x s features (C = z_dim, s = out_size)."""

    def __init__(self, z_dim: int, out_size: int):
        super().__init__()
        # The hub cache if there (no network, e.g. compute nodes); else downloaded into it.
        local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if local.is_dir():
            dino = torch.hub.load(str(local), DINO_NAME, source="local")
        else:
            dino = torch.hub.load("facebookresearch/dinov2", DINO_NAME)
        self.dino = dino.eval().requires_grad_(False)
        self.out_size, self.grid = out_size, out_size // GRID_DIVISOR
        self.proj = nn.Sequential(nn.Linear(DINO_DIM, z_dim), nn.ReLU(True), nn.Linear(z_dim, z_dim))

    def train(self, mode: bool = True) -> DinoEncoder:
        super().train(mode)
        self.dino.eval()  # frozen: always eval
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images, size=self.grid * PATCH, mode="bilinear", align_corners=False, antialias=True)
        x = (x - x.new_tensor(MEAN)[:, None, None]) / x.new_tensor(STD)[:, None, None]
        with torch.no_grad():
            tokens = self.dino.forward_features(x)["x_norm_patchtokens"]  # (B, g*g, DINO_DIM)
        feat = self.proj(tokens).transpose(1, 2).unflatten(2, (self.grid, self.grid))  # (B, C, g, g)
        return F.interpolate(feat, size=self.out_size, mode="bilinear", align_corners=False)
