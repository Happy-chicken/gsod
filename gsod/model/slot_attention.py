"""Slot attention: background with its own query, GRU and residual MLP; the
rest as in BO-QSA: tokens + position embedding through a feature MLP, learned
per-slot queries with annealed noise, weighted mean with eps = 1, DropPath on
the residual MLP, bi-level (only the last iteration is differentiated, straight
through to the queries). Slot 0 is the background."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from gsod.model.layers import mlp
from gsod.utils.schedule import cosine

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
    """(b, N, C) tokens -> background slot, object slots, attention (b, K, N)."""

    def __init__(self, num_slots: int, dim: int, iters: int):
        """dim: token and slot size."""
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.scale = dim**-0.5
        # A tensor, so torch.compile does not recompile as it anneals.
        self.register_buffer("sigma", torch.tensor(SIGMA_START), persistent=False)
        self.noisy = SIGMA_START > 0

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

    def anneal(self, step: int) -> None:
        """Cosine-anneal the noise sigma."""
        sigma = cosine(step / SIGMA_STEPS, SIGMA_START, SIGMA_FINAL)
        self.sigma.fill_(sigma)
        self.noisy = sigma > 0

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokens (b, N, C) -> background slot (b, D), object slots (b, K-1, D), attention (b, K, N)."""
        # Keys / values: tokens + position embedding.
        feat = self.mlp_feat(tokens + self.to_pos(self._grid(tokens.shape[1], tokens.device)))
        feat = self.norm_feat(feat)
        k, v = self.to_k(feat), self.to_v(feat)

        b = k.shape[0]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.noisy:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_obj = start[:, :1], start[:, 1:]

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_obj = slot_obj.detach() + start[:, 1:] - start[:, 1:].detach()
            attn, updates = self._attend(slot_bg, slot_obj, k, v)  # (b, K, N)
            updates_bg, updates_obj = updates[:, :1], updates[:, 1:]
            slot_bg = self._update(self.gru_bg, self.mlp_bg, updates_bg, slot_bg)
            slot_obj = self._update(self.gru, self.mlp, updates_obj, slot_obj)

        return slot_bg[:, 0], slot_obj, attn

    def _attend(
        self, slot_bg: torch.Tensor, slot_obj: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> attention (b, K, N) and the weighted means of v (b, K, D)."""
        dots_bg = torch.einsum("bid,bjd->bij", self.to_q_bg(slot_bg), k) * self.scale
        dots_obj = torch.einsum("bid,bjd->bij", self.to_q(slot_obj), k) * self.scale
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
