"""Slot attention: background as in uORF (own query, GRU, residual MLP),
initialisation and optimisation as in BO-QSA (learned per-slot queries with
annealed noise, bi-level: only the last iteration is differentiated, straight
through to the queries). Slot 0 is the background."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from gsod.utils.schedule import cosine

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
        self.sigma = SIGMA_START

        self.slots_init = nn.Embedding(num_slots, slot_dim)
        init.xavier_uniform_(self.slots_init.weight)

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

    def anneal(self, step: int) -> None:
        """Cosine-anneal the noise sigma."""
        self.sigma = cosine(step / SIGMA_STEPS, SIGMA_START, SIGMA_FINAL)

    def forward(self, feat: torch.Tensor):
        b = feat.shape[0]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_fg = start[:, :1], start[:, 1:]

        feat = self.norm_feat(feat)
        k, v = self.to_k(feat), self.to_v(feat)

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_fg = slot_fg.detach() + start[:, 1:] - start[:, 1:].detach()
            dots_bg = torch.einsum("bid,bjd->bij", self.to_q_bg(slot_bg), k) * self.scale
            dots_fg = torch.einsum("bid,bjd->bij", self.to_q(slot_fg), k) * self.scale
            attn = torch.cat([dots_bg, dots_fg], dim=1).softmax(dim=1) + self.eps  # (b, K, N)
            attn_bg, attn_fg = attn[:, :1], attn[:, 1:]
            updates_bg = torch.einsum("bjd,bij->bid", v, attn_bg / attn_bg.sum(-1, keepdim=True))
            updates_fg = torch.einsum("bjd,bij->bid", v, attn_fg / attn_fg.sum(-1, keepdim=True))
            slot_bg = self._update(self.gru_bg, self.to_res_bg, updates_bg, slot_bg)
            slot_fg = self._update(self.gru, self.to_res, updates_fg, slot_fg)

        return slot_bg[:, 0], slot_fg, attn
