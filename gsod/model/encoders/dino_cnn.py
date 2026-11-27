"""Image encoder: DINOv2 features (grouping) + a shallow CNN on the image (colour, detail)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.encoders.dinov2 import DinoEncoder
from gsod.model.layers import conv_block


class DinoCNNEncoder(nn.Module):
    """B x 3 x H x W image in [0, 1] -> B x C x s x s features (C = z_dim, s = out_size)."""

    def __init__(self, z_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.dino = DinoEncoder(z_dim, out_size)
        self.cnn = nn.Sequential(conv_block(3, z_dim), conv_block(z_dim, z_dim))  # full s x s resolution
        self.fuse = nn.Conv2d(2 * z_dim, z_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        small = F.interpolate(images, size=self.out_size, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.dino(images), self.cnn(small)], 1))
