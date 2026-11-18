"""Slot attention as in uORF (Yu et al., ICLR 2022): the background slot has
its own initial distribution, query, GRU and residual MLP; slot 0 is the
background."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init


class SlotAttention(nn.Module):
    """(b, N, C) tokens -> background slot, object slots, attention (b, K, N)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)

        self.norm_feat = nn.LayerNorm(in_dim)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        hidden_dim = max(slot_dim, hidden_dim)
        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )

    def _update(self, gru: nn.GRUCell, res: nn.Module, updates: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        b, k, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, k, d)
        return slots + res(slots)

    def forward(self, feat: torch.Tensor):
        b = feat.shape[0]
        k_fg = self.num_slots - 1
        mu = self.slots_mu.expand(b, k_fg, -1)
        slot_fg = mu + self.slots_logsigma.exp().expand(b, k_fg, -1) * torch.randn_like(mu)
        mu_bg = self.slots_mu_bg.expand(b, 1, -1)
        slot_bg = mu_bg + self.slots_logsigma_bg.exp().expand(b, 1, -1) * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k, v = self.to_k(feat), self.to_v(feat)

        attn = None
        for _ in range(self.iters):
            dots_bg = torch.einsum("bid,bjd->bij", self.to_q_bg(slot_bg), k) * self.scale
            dots_fg = torch.einsum("bid,bjd->bij", self.to_q(slot_fg), k) * self.scale
            attn = torch.cat([dots_bg, dots_fg], dim=1).softmax(dim=1) + self.eps  # (b, K, N)
            attn_bg, attn_fg = attn[:, :1], attn[:, 1:]
            updates_bg = torch.einsum("bjd,bij->bid", v, attn_bg / attn_bg.sum(-1, keepdim=True))
            updates_fg = torch.einsum("bjd,bij->bid", v, attn_fg / attn_fg.sum(-1, keepdim=True))
            slot_bg = self._update(self.gru_bg, self.to_res_bg, updates_bg, slot_bg)
            slot_fg = self._update(self.gru, self.to_res, updates_fg, slot_fg)

        return slot_bg[:, 0], slot_fg, attn
