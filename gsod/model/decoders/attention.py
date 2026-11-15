"""2D reconstruction of view 0 with the attention as masks; an object slot
draws only within an object-sized envelope about its centre, else background.
"""

from __future__ import annotations

import torch
from torch import nn

from gsod.model.decoders.objects import OBJECT_RADIUS, positional_encoding

BANDS = 4
# Envelope radius (units of r_max), projected into view 0.
ENVELOPE = 1.5


class AttnDecoder(nn.Module):
    """Object slots -> colour per token about the slot's centre; mixed by attention."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 2 + 4 * BANDS, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        bg: torch.Tensor,
        uv: torch.Tensor,
        depth: torch.Tensor,
        focal_ratio: tuple[float, float],
    ) -> torch.Tensor:
        """slots (K-1, C), attn (K, N), bg (N, 3), uv (N, 2), depth (K-1,) -> (N, 3)."""
        p = attn[1:] / attn[1:].sum(-1, keepdim=True).clamp_min(1e-8)
        centre = p @ uv  # (K-1, 2)
        focal = torch.tensor(focal_ratio, device=uv.device, dtype=uv.dtype)
        sigma = 2.0 * focal * ENVELOPE * OBJECT_RADIUS[1] / depth[:, None]  # (K-1, 2)
        rel = (uv[None] - centre[:, None]) / sigma[:, None]  # (K-1, N, 2)
        w = torch.exp(-0.5 * rel.pow(2).sum(-1, keepdim=True))
        k, n = rel.shape[:2]
        pe = positional_encoding((rel.clamp(-3.0, 3.0) / 3.0).flatten(0, 1), BANDS).view(k, n, -1)
        inp = torch.cat([slots[:, None].expand(-1, n, -1), pe], -1)
        objects = w * torch.sigmoid(self.mlp(inp)) + (1.0 - w) * bg[None]
        colours = torch.cat([bg[None], objects], 0)  # (K, N, 3)
        return (attn[..., None] * colours).sum(0)
