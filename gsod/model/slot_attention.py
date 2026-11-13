"""Slot attention (BO-QSA, Jia et al., ICLR 2023); slot 0 is the background."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init

# Query noise schedule.
SIGMA_START = 1.0
SIGMA_FINAL = 0.0
SIGMA_STEPS = 30000


class SlotAttention(nn.Module):
    """(b, N, C) tokens -> background slot, object slots, attention (b, K, N)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1.0,
        hidden_dim: int = 128,
    ):
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.eps = eps
        self.slot_dim = slot_dim
        self.sigma = SIGMA_START

        self.slots_init = nn.Embedding(num_slots, slot_dim)
        init.xavier_uniform_(self.slots_init.weight)
        self.pos_proj = nn.Linear(4, in_dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, slot_dim),
        )
        self.norm_feature = nn.LayerNorm(slot_dim)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] of a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1).reshape(1, n_tokens, 4)

    def anneal(self, step: int) -> None:
        """Cosine-anneal the noise sigma."""
        progress = min(step / SIGMA_STEPS, 1.0)
        a, b = 0.5 * (SIGMA_START - SIGMA_FINAL), 0.5 * (SIGMA_START + SIGMA_FINAL)
        self.sigma = a * math.cos(math.pi * progress) + b

    def forward(self, feat: torch.Tensor):
        b, D = feat.shape[0], self.slot_dim
        feat = self.feat_mlp(feat + self.pos_proj(self._grid(feat.shape[1], feat.device)))
        feat = self.norm_feature(feat)
        k, v = self.project_k(feat), self.project_v(feat)

        mu = self.slots_init.weight.expand(b, -1, -1)
        slots_init = mu
        if self.training and self.sigma > 0:
            slots_init = mu + torch.randn_like(mu) * self.sigma * mu.detach()
        slots = slots_init

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slots = slots.detach() + slots_init - slots_init.detach()
            prev = slots
            q = self.project_q(self.norm_slots(slots))
            attn = torch.einsum("bkd,bnd->bkn", q, k).mul(D**-0.5).softmax(dim=1)
            attn_wm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bkn,bnd->bkd", attn_wm, v)
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(b, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots[:, 0], slots[:, 1:], attn, v
