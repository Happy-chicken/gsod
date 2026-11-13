"""2D attention reconstruction: the attention as masks over view 0's tokens.

Every token's colour depends on every slot, so the loss reaches every slot.
"""

from __future__ import annotations

import torch
from torch import nn

from gsod.model.decoders.objects import positional_encoding

BANDS = 4


class AttnDecoder(nn.Module):
    """Object slots -> colour per token (spatial broadcast); mixed by attention."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 2 + 4 * BANDS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, slots: torch.Tensor, attn: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        """slots (K-1, C), attn (K, N) (row 0 background), bg colours (N, 3) -> (N, 3)."""
        side = int(round(attn.shape[1] ** 0.5))
        lin = torch.linspace(-1.0, 1.0, side, device=attn.device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        pe = positional_encoding(torch.stack([x, y], -1).reshape(-1, 2), BANDS)  # (N, P)
        k, n = slots.shape[0], pe.shape[0]
        inp = torch.cat([slots[:, None].expand(-1, n, -1), pe[None].expand(k, -1, -1)], -1)
        colours = torch.cat([bg[None], torch.sigmoid(self.mlp(inp))], 0)  # (K, N, 3)
        return (attn[..., None] * colours).sum(0)
