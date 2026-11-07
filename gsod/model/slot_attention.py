"""Slot attention: a background slot + K-1 object slots (uORF).

Logits: learned temperature * cos(query, key), softmax over slots. The
background has its own query, GRU and MLP; object slots are seeded from
tokens unlike the background query (farthest point sampling). Both are
refined BO-QSA style (detached iterations, straight-through last step).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

_TAU_INIT, _TAU_MIN, _TAU_MAX = 5.0, 1.0, 50.0


@torch.no_grad()
def seed_tokens(objectness: torch.Tensor, positions: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest point sampling weighted by objectness: objectness (b, N),
    token positions (N, 2) -> (b, k) token indices. The first is the most
    object-like token; each next maximises objectness (scaled to [0, 1] per
    image) times the distance to the nearest seed so far."""
    lo = objectness.amin(-1, keepdim=True)
    hi = objectness.amax(-1, keepdim=True)
    o = (objectness - lo) / (hi - lo).clamp_min(1e-8)
    idx = [o.argmax(-1)]
    nearest = torch.cdist(positions[idx[0]], positions)  # (b, N)
    for _ in range(k - 1):
        nxt = (o * nearest).argmax(-1)
        idx.append(nxt)
        nearest = torch.minimum(nearest, torch.cdist(positions[nxt], positions))
    return torch.stack(idx, -1)


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
        self.seed_proj = nn.Linear(in_dim, slot_dim)
        self.bg_query = nn.Parameter(torch.empty(1, 1, slot_dim))
        init.xavier_uniform_(self.bg_query)

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

    def forward(self, feat: torch.Tensor, fixed_attn: torch.Tensor | None = None):
        """fixed_attn (b, K, N): used instead of the computed attention
        (diagnostic, gsod.oracle_attention)."""
        b = feat.shape[0]
        grid = self._grid(feat.shape[1], feat.device)
        feat = feat + self.pos_proj(grid)
        feat = self.norm_feat(self.feat_mlp(feat))
        k, v = F.normalize(self.to_k(feat), dim=-1), self.to_v(feat)
        tau = self.temperature()

        bg_init = self.bg_query.expand(b, -1, -1)
        q_bg = F.normalize(self.to_q_bg(bg_init), dim=-1)  # (b, 1, D)
        objectness = 1.0 - torch.einsum("bid,bjd->bj", q_bg, k.detach())
        idx = seed_tokens(objectness, grid[0, :, :2], self.n_objects)  # (b, K-1)
        picked = torch.gather(feat, 1, idx[..., None].expand(-1, -1, feat.shape[-1]))
        fg_init = self.seed_proj(picked)
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
            attn = dots.softmax(dim=1) + self.eps if fixed_attn is None else fixed_attn
            attn_w = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn_w)  # (b, K, D)
            bg = self._update(self.gru_bg, self.to_res_bg, updates[:, :1], bg)
            fg = self._update(self.gru, self.to_res, updates[:, 1:], fg)

        return bg[:, 0], fg, attn
