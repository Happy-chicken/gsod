"""Slot attention: a background slot + K-1 object slots (uORF).

Logits: learned temperature * cos(query, key), softmax over slots. As
BO-QSA: every slot starts from its own learned query, refined with detached
iterations and a straight-through last step to the queries. The background
has its own query, GRU and MLP (uORF).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

_TAU_INIT, _TAU_MIN, _TAU_MAX = 5.0, 1.0, 50.0


class SlotAttention(nn.Module):
    """BxNxC tokens -> (BxD background slot, Bx(K-1)xD object slots, BxKxN
    attention, softmax over the background (row 0) and the object slots)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        """num_slots: K, the background slot included (K-1 object slots)."""
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.eps = eps
        self.slot_dim = slot_dim
        self.log_tau = nn.Parameter(torch.tensor(math.log(_TAU_INIT)))

        self.n_objects = num_slots - 1
        self.bg_query = nn.Parameter(torch.empty(1, 1, slot_dim))
        self.fg_query = nn.Parameter(torch.empty(1, self.n_objects, slot_dim))
        init.xavier_uniform_(self.bg_query)
        init.xavier_uniform_(self.fg_query)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        hidden_dim = max(slot_dim, hidden_dim)
        self.to_q = self._query(slot_dim)
        self.to_q_bg = self._query(slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.to_res = self._residual_mlp(slot_dim, hidden_dim)
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
    def _query(dim: int) -> nn.Sequential:
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))

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
        return torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1).reshape(1, n_tokens, 4)

    def temperature(self) -> torch.Tensor:
        """The shared logit temperature, in [_TAU_MIN, _TAU_MAX]."""
        return self.log_tau.exp().clamp(_TAU_MIN, _TAU_MAX)

    def _update(self, gru, res, updates, prev):
        """GRU then residual MLP; updates, prev: (b, n, D)."""
        b, n, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, n, d)
        return slots + res(slots)

    def forward(self, feat: torch.Tensor):
        """feat (b, N, C) -> background slot (b, D), object slots (b, K-1, D),
        attention (b, K, N), the tokens' values v (b, N, D)."""
        b = feat.shape[0]
        grid = self._grid(feat.shape[1], feat.device)
        feat = feat + self.pos_proj(grid)
        feat = self.norm_feat(self.feat_mlp(feat))
        k, v = F.normalize(self.to_k(feat), dim=-1), self.to_v(feat)
        tau = self.temperature()

        bg_init = self.bg_query.expand(b, -1, -1)
        fg_init = self.fg_query.expand(b, -1, -1)
        bg, fg = bg_init, fg_init

        attn = None
        for it in range(self.iters):
            # BO-QSA: straight-through gradient to the initial slots.
            bg, fg = bg.detach(), fg.detach()
            if it == self.iters - 1:
                bg = bg + bg_init - bg_init.detach()
                fg = fg + fg_init - fg_init.detach()
            q = torch.cat([self.to_q_bg(bg), self.to_q(fg)], dim=1)  # (b, K, D)
            dots = tau * torch.einsum("bid,bjd->bij", F.normalize(q, dim=-1), k)
            attn = dots.softmax(dim=1) + self.eps
            attn_w = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn_w)  # (b, K, D)
            bg = self._update(self.gru_bg, self.to_res_bg, updates[:, :1], bg)
            fg = self._update(self.gru, self.to_res, updates[:, 1:], fg)

        return bg[:, 0], fg, attn, v
