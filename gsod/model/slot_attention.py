"""Slot attention with a background slot and K object slots.

The background has its own slot, as in uORF (Yu et al., ICLR 2022): it
competes with the object slots for the tokens like any slot, but has its own
parameters (initial query, query projection, GRU, residual MLP), so which
slot is the background is fixed by construction:

    attention row 0     background slot
    attention rows 1..K object slots

The softmax over rows 0..K makes every token choose between "background" and
one of the objects. The background slot conditions the colours of
GSODModel's BackgroundShell; the object slots are decoded into Gaussians.

Slot initialisation: BO-QSA (Jia et al., ICLR 2023, "Improving Object-centric
Learning with Query Optimization"). Every slot, the background included, has
its own learned initial query. The refinement iterations run on detached
slots, and the gradient of the last iteration is passed straight through to
the queries.
Learned queries trained without this scheme (plain QSA) are worse than random
initialisation in the paper's ablation; the detach + straight-through pair is
what makes them work.

Original Slot Attention initialisation (replaced, kept for reference): all
object slots sampled from one shared learnable Gaussian, with gradients
through every iteration. Slots then start nearly identical and tend to split
one object equally between them (fg_slot_cos_mean 0.45 vs 0.33, attention
FG-ARI 0.71 vs 0.85 with BO-QSA in our runs).

    # __init__
    #     self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
    #     self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
    #     init.xavier_uniform_(self.slots_logsigma)
    # forward
    #     mu = self.slots_mu.expand(b, self.num_slots, -1)
    #     sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
    #     slots = mu + sigma * torch.randn_like(mu)
    #     for _ in range(self.iters):     # no detach, no straight-through
    #         ...
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init


class SlotAttention(nn.Module):
    """BxNxC tokens -> (BxKxD object slots, Bx1xD background slot,
    Bx(1+K)xN attention)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        """num_slots: number of object slots K (the background is extra)."""
        super().__init__()
        assert num_slots >= 1, "need at least one object slot"
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim

        # BO-QSA: one learned initial query per object slot.
        self.slot_queries = nn.Parameter(torch.empty(1, num_slots, slot_dim))
        init.xavier_uniform_(self.slot_queries)
        # Background slot: its own initial query.
        self.bg_query = nn.Parameter(torch.empty(1, 1, slot_dim))
        init.xavier_uniform_(self.bg_query)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False)
        )
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        hidden_dim = max(slot_dim, hidden_dim)
        self.to_res = self._residual_mlp(slot_dim, hidden_dim)
        # The background slot's own query projection, GRU and residual MLP.
        self.to_q_bg = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False)
        )
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.to_res_bg = self._residual_mlp(slot_dim, hidden_dim)
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

    def project_inputs(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BxNxC tokens -> (keys, values), each BxNxD.

        The keys are the space the attention compares slots in: a token can
        only be told apart from another if their keys differ.
        """
        feat = feat + self.pos_proj(self._grid(feat.shape[1], feat.device))
        feat = self.norm_feat(self.feat_mlp(feat))
        return self.to_k(feat), self.to_v(feat)

    def forward(self, feat: torch.Tensor):
        b = feat.shape[0]
        slots_init = self.slot_queries.expand(b, -1, -1)
        bg_init = self.bg_query.expand(b, -1, -1)
        slots, bg = slots_init, bg_init

        k, v = self.project_inputs(feat)

        attn = None
        for it in range(self.iters):
            # BO-QSA: earlier iterations are not back-propagated; the last one
            # passes its gradient straight through to the initial queries.
            slots, bg = slots.detach(), bg.detach()
            if it == self.iters - 1:
                slots = slots + slots_init - slots_init.detach()
                bg = bg + bg_init - bg_init.detach()
            slots_prev, bg_prev = slots, bg
            q = torch.cat([self.to_q_bg(bg), self.to_q(slots)], dim=1)
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn_w = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn_w)  # (b, 1+K, D)

            slots = self.gru(
                updates[:, 1:].reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, -1, self.slot_dim)
            slots = slots + self.to_res(slots)
            bg = self.gru_bg(updates[:, 0], bg_prev[:, 0])[:, None]
            bg = bg + self.to_res_bg(bg)

        return slots, bg, attn
