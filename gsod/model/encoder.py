"""Image encoder: conv U-Net over one view (positions are embedded in slot attention)."""

from __future__ import annotations

import torch
from torch import nn

from gsod.model.layers import conv_block


class Encoder(nn.Module):
    """Bx3xHxW image -> BxCxHxW feature map (C = z_dim)."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.down_1 = conv_block(3, z_dim)
        self.down_2 = conv_block(z_dim, z_dim, stride=2)
        self.down_3 = conv_block(z_dim, z_dim, stride=2)
        self.up_3 = conv_block(z_dim, z_dim, up=True)
        self.up_2 = conv_block(z_dim * 2, z_dim, up=True)
        self.up_1 = conv_block(z_dim * 2, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_down_1 = self.down_1(x)
        x_down_2 = self.down_2(x_down_1)
        x_down_3 = self.down_3(x_down_2)
        x_up_3 = self.up_3(x_down_3)
        x_up_2 = self.up_2(torch.cat([x_up_3, x_down_2], dim=1))
        return self.up_1(torch.cat([x_up_2, x_down_1], dim=1))
