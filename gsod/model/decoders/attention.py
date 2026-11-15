"""Global 2D component: every slot decodes a colour and a mask logit at every
view-0 token (spatial broadcast); masks are a softmax over slots, so slots
compete everywhere. The background colour is the floor render.
"""

from __future__ import annotations

import torch
from torch import nn

from gsod.model.decoders.objects import positional_encoding

BANDS = 4


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class AttnDecoder(nn.Module):
    """Slots -> view-0 colours (N, 3) mixed by decoded masks."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.objects = _mlp(z_dim + 2 + 4 * BANDS, hidden_dim, 4)  # rgb, logit
        self.background = _mlp(z_dim + 2 + 4 * BANDS, hidden_dim, 1)  # logit

    def forward(
        self, bg_slots: torch.Tensor, slots: torch.Tensor, bg: torch.Tensor, uv: torch.Tensor
    ) -> torch.Tensor:
        """bg_slots (B, C), slots (K-1, C), floor colours (N, 3), uv (N, 2) -> (N, 3)."""
        pe = positional_encoding(uv, BANDS)  # (N, P)
        n, n_bg = pe.shape[0], bg_slots.shape[0]
        everything = torch.cat([bg_slots, slots], 0)
        inp = torch.cat([everything[:, None].expand(-1, n, -1), pe[None].expand(len(everything), -1, -1)], -1)
        obj = self.objects(inp[n_bg:])
        bg_logit = self.background(inp[:n_bg]).logsumexp(0, keepdim=True)
        logits = torch.cat([bg_logit, obj[..., 3:]], 0)  # (K, N, 1)
        colours = torch.cat([bg[None], torch.sigmoid(obj[..., :3])], 0)  # (K, N, 3)
        return (logits.softmax(0) * colours).sum(0)
