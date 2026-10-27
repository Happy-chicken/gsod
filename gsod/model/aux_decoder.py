"""2D auxiliary decoder (fg_source = "attention").

A spatial broadcast decoder (Watters et al. 2019; the decoder of the original
Slot Attention): every latent is tiled over a small grid, a Fourier encoding
of the grid position is appended, and a per-pixel MLP outputs an RGB colour
and a mask logit. The masks are a softmax over the latents and the image is
the mask-weighted sum of the colours.

It reconstructs view 0 at a low resolution from [background latent; object
slots]. This trains the encoder and slot attention with a pure image
reconstruction signal that needs no foreground target F, which is what lets
the slot attention's own foreground (1 - background row) replace F as the
target of the 3D losses. It is used in training only; the 3D model does not
read its output.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SpatialBroadcastDecoder(nn.Module):
    """(L, D) latents -> (rgb (3, size, size), masks (L, size, size))."""

    def __init__(
        self, slot_dim: int, size: int = 32, hidden_dim: int = 128, n_freqs: int = 4
    ):
        super().__init__()
        self.size = size
        lin = torch.linspace(-1.0, 1.0, size)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        xy = torch.stack([x, y], dim=-1).reshape(-1, 2)  # (size*size, 2)
        freqs = math.pi * 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        angles = (xy[:, :, None] * freqs).flatten(1)  # (size*size, 2*n_freqs)
        self.register_buffer(
            "pos", torch.cat([xy, angles.sin(), angles.cos()], dim=-1), persistent=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim + self.pos.shape[1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_latents, n_pix = latents.shape[0], self.pos.shape[0]
        x = torch.cat(
            [
                latents[:, None].expand(-1, n_pix, -1),
                self.pos[None].expand(n_latents, -1, -1).to(latents.dtype),
            ],
            dim=-1,
        )
        out = self.mlp(x)  # (L, size*size, 4)
        rgb = torch.sigmoid(out[..., :3])
        masks = out[..., 3].softmax(dim=0)  # (L, size*size)
        recon = (masks[..., None] * rgb).sum(0)  # (size*size, 3)
        side = self.size
        return recon.T.reshape(3, side, side), masks.reshape(n_latents, side, side)
