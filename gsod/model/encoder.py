"""Image encoder: conv U-Net over one view, with pixel-coordinate channels."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Bx3xHxW image -> BxCxHxW feature map (C = z_dim)."""

    def __init__(self, input_nc: int = 3, z_dim: int = 64):
        super().__init__()
        self.enc_down_1 = nn.Sequential(
            nn.Conv2d(input_nc + 4, z_dim, 3, stride=1, padding=1), nn.ReLU(True)
        )
        self.enc_down_2 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1), nn.ReLU(True)
        )
        self.enc_down_3 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1), nn.ReLU(True)
        )
        self.enc_up_3 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.enc_up_2 = nn.Sequential(
            nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.enc_up_1 = nn.Sequential(
            nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1), nn.ReLU(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        y1, x1 = torch.meshgrid(ys, xs, indexing="ij")
        pixel_emb = (
            torch.stack([x1, 2 - x1, y1, 2 - y1])
            .unsqueeze(0)
            .expand(x.shape[0], -1, -1, -1)
        )
        x_ = torch.cat([x, pixel_emb], dim=1)

        x_down_1 = self.enc_down_1(x_)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        return self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))
