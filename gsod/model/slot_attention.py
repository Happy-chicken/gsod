"""Slot attention with K-1 object slots and an empty option.

There is no background slot. The softmax over slots has one more entry, a
fixed logit of 0 with no slot behind it, so a token none of the object
slots wants (the empty background) can go to nothing instead of being
forced into an object slot. Which tokens are empty is learned through the
keys; the option itself has no parameters.

    attention row 0      empty (background)
    attention rows 1..   object slots 1..K-1 (returned slots 0..K-2)

Slot initialisation: BO-QSA (Jia et al., ICLR 2023, "Improving Object-centric
Learning with Query Optimization"). Every slot has its own learned initial
query. The refinement iterations run on detached slots, and the gradient of
the last iteration is passed straight through to the queries. Learned
queries trained without this scheme (plain QSA) are worse than random
initialisation in the paper's ablation; the detach + straight-through pair is
what makes them work.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init


class SlotAttention(nn.Module):
    """BxNxC tokens -> (Bx(K-1)xD object slots, BxKxN attention, softmax
    over the empty option (row 0) and the slots)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        """num_slots: K, the empty option included (K-1 object slots)."""
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim

        # BO-QSA: one learned initial query per object slot.
        self.slot_queries = nn.Parameter(torch.empty(1, num_slots - 1, slot_dim))
        init.xavier_uniform_(self.slot_queries)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False)
        )
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        hidden_dim = max(slot_dim, hidden_dim)
        self.to_res = self._residual_mlp(slot_dim, hidden_dim)
        # Soft position embedding + feature MLP (Locatello et al. 2020).
        self.pos_proj = nn.Linear(4, in_dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
        )
        self.norm_feat = nn.LayerNorm(in_dim)

    @staticmethod
    def _residual_mlp(dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] in [0, 1] for a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        grid = torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1)
        return grid.reshape(1, n_tokens, 4)

    def forward(self, feat: torch.Tensor):
        b = feat.shape[0]
        slots_init = self.slot_queries.expand(b, -1, -1)
        slots = slots_init

        feat = feat + self.pos_proj(self._grid(feat.shape[1], feat.device))
        feat = self.norm_feat(self.feat_mlp(feat))
        k, v = self.to_k(feat), self.to_v(feat)

        attn = None
        for it in range(self.iters):
            # BO-QSA: earlier iterations are not back-propagated; the last one
            # passes its gradient straight through to the initial queries.
            slots = slots.detach()
            if it == self.iters - 1:
                slots = slots + slots_init - slots_init.detach()
            slots_prev = slots
            dots = torch.einsum("bid,bjd->bij", self.to_q(slots), k) * self.scale
            # Row 0: the empty option, a fixed logit of 0.
            dots = torch.cat([torch.zeros_like(dots[:, :1]), dots], dim=1)
            attn = dots.softmax(dim=1) + self.eps
            attn_w = attn[:, 1:] / attn[:, 1:].sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn_w)  # (b, K-1, D)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, -1, self.slot_dim)
            slots = slots + self.to_res(slots)

        return slots, attn
