"""2D spatial broadcast decoder: the slots reconstruct their input image.

The decoder of the original Slot Attention (Locatello et al. 2020; Watters
et al. 2019): every slot is tiled over a grid, a Fourier encoding of the grid
position is appended, and a per-pixel MLP outputs an RGB colour and an alpha
logit. The alphas compete in a softmax over the slots and the image is the
alpha-weighted sum of the colours.

Reconstructing the image through K slots of equal capacity is what makes
slot attention decompose a scene without any target: one slot drawing the
whole image costs more error than several slots drawing one part each.
GSODModel applies it to each encoded view; the 3D model does not read it.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SpatialBroadcastDecoder(nn.Module):
    """(B, K, D) slots -> (rgb (B, 3, size, size), alphas (B, K, size, size))."""

    def __init__(
        self, slot_dim: int, size: int = 64, hidden_dim: int = 128, n_freqs: int = 4
    ):
        super().__init__()
        self.size = size
        lin = torch.linspace(-1.0, 1.0, size)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        xy = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (size*size, 2)
        freqs = math.pi * 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        angles = (xy[:, :, None] * freqs).flatten(1)  # (size*size, 2*n_freqs)
        pos = torch.cat([xy, angles.sin(), angles.cos()], dim=-1)
        self.register_buffer("pos", pos, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + pos.shape[1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = slots.shape
        n_pix = self.pos.shape[0]
        x = torch.cat(
            [
                slots[:, :, None].expand(-1, -1, n_pix, -1),
                self.pos.to(slots.dtype).expand(b, k, -1, -1),
            ],
            dim=-1,
        )
        out = self.mlp(x)  # (B, K, P, 4)
        alphas = out[..., 3].softmax(dim=1)  # (B, K, P)
        recon = (alphas[..., None] * torch.sigmoid(out[..., :3])).sum(1)  # (B, P, 3)
        side = self.size
        return (
            recon.transpose(1, 2).reshape(b, 3, side, side),
            alphas.reshape(b, k, side, side),
        )
