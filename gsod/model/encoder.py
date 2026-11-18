"""Image encoder: conv U-Net over one view (positions are embedded in slot attention)."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Bx3xHxW image -> BxCxHxW feature map (C = z_dim)."""

    def __init__(self, input_nc: int = 3, z_dim: int = 64):
        super().__init__()
        self.enc_down_1 = nn.Sequential(
            nn.Conv2d(input_nc, z_dim, 3, stride=1, padding=1, padding_mode="replicate"), nn.ReLU(True)
        )
        self.enc_down_2 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1, padding_mode="replicate"), nn.ReLU(True)
        )
        self.enc_down_3 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1, padding_mode="replicate"), nn.ReLU(True)
        )
        self.enc_up_3 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.enc_up_2 = nn.Sequential(
            nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1, padding_mode="replicate"),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.enc_up_1 = nn.Sequential(
            nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1, padding_mode="replicate"), nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down_1 = self.enc_down_1(x)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        return self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))
