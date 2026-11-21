"""Slot attention: background as in uORF (own query, GRU, residual MLP); the
rest as in BO-QSA: tokens + position embedding through a feature MLP, learned
per-slot queries with annealed noise, weighted mean with eps = 1, DropPath on
the residual MLP, bi-level (only the last iteration is differentiated, straight
through to the queries). Slot 0 is the background.

refine(): one more object-slot update whose keys also see a per-token error
(what the current slots leave unexplained), so an idle slot can be drawn to a
missed object (after MONet / IODINE). Starts as a plain iteration."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from gsod.model.layers import mlp
from gsod.utils.schedule import cosine

# Query noise schedule.
SIGMA_START = 1.0
SIGMA_FINAL = 0.0
SIGMA_STEPS = 30000
DROP_PATH = 0.2
EPS = 1.0  # weighted mean: attention / (its sum + EPS)
HIDDEN = 128  # MLP width (at least the slot size)


def _drop_path(x: torch.Tensor, training: bool) -> torch.Tensor:
    """Drop the whole residual of a sample with probability DROP_PATH."""
    if not training:
        return x
    keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= DROP_PATH).to(x.dtype)
    return x * keep / (1.0 - DROP_PATH)


class SlotAttention(nn.Module):
    """Keys / values of (b, N, C) tokens -> background slot, object slots,
    attention (b, K, N)."""

    def __init__(self, num_slots: int, dim: int, iters: int):
        """dim: token and slot size."""
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.scale = dim**-0.5
        self.sigma = SIGMA_START

        self.slots_init = nn.Embedding(num_slots, dim)
        init.xavier_uniform_(self.slots_init.weight)

        hidden = max(dim, HIDDEN)
        self.to_pos = nn.Linear(4, dim)
        self.mlp_feat = mlp(dim, hidden, dim, norm=True)
        self.norm_feat = nn.LayerNorm(dim)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.gru = nn.GRUCell(dim, dim)
        self.gru_bg = nn.GRUCell(dim, dim)
        self.mlp = mlp(dim, hidden, dim, norm=True)
        self.mlp_bg = mlp(dim, hidden, dim, norm=True)
        self.to_err = nn.Linear(1, dim)
        nn.init.zeros_(self.to_err.weight)
        nn.init.zeros_(self.to_err.bias)

    def anneal(self, step: int) -> None:
        """Cosine-anneal the noise sigma."""
        self.sigma = cosine(step / SIGMA_STEPS, SIGMA_START, SIGMA_FINAL)

    def keys_values(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens (b, N, C) + position embedding -> keys, values (b, N, D)."""
        feat = self.mlp_feat(feat + self.to_pos(self._grid(feat.shape[1], feat.device)))
        feat = self.norm_feat(feat)
        return self.to_k(feat), self.to_v(feat)

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        b = k.shape[0]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_obj = start[:, :1], start[:, 1:]

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_obj = slot_obj.detach() + start[:, 1:] - start[:, 1:].detach()
            attn, updates = self._attend(slot_bg, slot_obj, k, v, k)  # (b, K, N)
            updates_bg, updates_obj = updates[:, :1], updates[:, 1:]
            slot_bg = self._update(self.gru_bg, self.mlp_bg, updates_bg, slot_bg)
            slot_obj = self._update(self.gru, self.mlp, updates_obj, slot_obj)

        return slot_bg[:, 0], slot_obj, attn

    def refine(
        self, k: torch.Tensor, v: torch.Tensor, slot_bg: torch.Tensor, slot_obj: torch.Tensor, error: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys / values, slots from forward, error (b, N) -> object slots and
        attention after one error-aware update."""
        attn, updates = self._attend(slot_bg[:, None], slot_obj, k, v, k + self.to_err(error[..., None]))
        return self._update(self.gru, self.mlp, updates[:, 1:], slot_obj), attn

    def _attend(
        self, slot_bg: torch.Tensor, slot_obj: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_obj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> attention (b, K, N) and the weighted means of v (b, K, D)."""
        dots_bg = torch.einsum("bid,bjd->bij", self.to_q_bg(slot_bg), k) * self.scale
        dots_obj = torch.einsum("bid,bjd->bij", self.to_q(slot_obj), k_obj) * self.scale
        attn = torch.cat([dots_bg, dots_obj], dim=1).softmax(dim=1)
        attn_wm = attn / (attn.sum(-1, keepdim=True) + EPS)
        return attn, torch.einsum("bij,bjd->bid", attn_wm, v)

    def _update(self, gru: nn.GRUCell, mlp: nn.Module, updates: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        b, k, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, k, d)
        return slots + _drop_path(mlp(slots), self.training)

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] of a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1).reshape(1, n_tokens, 4)
