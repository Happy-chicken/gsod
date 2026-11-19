"""Slot attention: background as in uORF (own query, GRU, residual MLP); the
rest as in BO-QSA: tokens through a feature MLP, learned per-slot queries with
annealed noise, weighted mean with eps = 1, DropPath on the residual MLP,
bi-level (only the last iteration is differentiated, straight through to the
initial state). Slot 0 is the background.

Positions as in Invariant Slot Attention (translation only): keys and values
are computed once without positions; each object slot carries a position mu
(its attention's centre) and sees keys grid_enc(k + grid_proj(x - mu)), so it
prefers tokens near itself; the background sees absolute positions. Values
carry no position."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init

from gsod.utils.schedule import cosine

# Query noise schedule.
SIGMA_START = 1.0
SIGMA_FINAL = 0.0
SIGMA_STEPS = 30000
DROP_PATH = 0.2


def _drop_path(x: torch.Tensor, training: bool) -> torch.Tensor:
    """Drop the whole residual of a sample with probability DROP_PATH."""
    if not training:
        return x
    keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= DROP_PATH).to(x.dtype)
    return x * keep / (1.0 - DROP_PATH)


def _mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    """Pre-LayerNorm two-layer MLP: LayerNorm -> Linear -> ReLU -> Linear."""
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d_out))


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
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim
        self.sigma = SIGMA_START

        self.slots_init = nn.Embedding(num_slots, slot_dim)
        init.xavier_uniform_(self.slots_init.weight)
        # Object slots' initial positions (spread on a grid).
        n_obj = num_slots - 1
        side = math.ceil(math.sqrt(n_obj))
        lin = (torch.arange(side) + 0.5) / side
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="xy"), -1).reshape(-1, 2)[:n_obj]
        self.pos_init = nn.Parameter(grid.clone())

        hidden_dim = max(slot_dim, hidden_dim)
        self.feat_mlp = _mlp(in_dim, hidden_dim, slot_dim)
        self.norm_feat = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.grid_proj = nn.Linear(2, slot_dim)
        self.grid_enc = _mlp(slot_dim, hidden_dim, slot_dim)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.to_res = _mlp(slot_dim, hidden_dim, slot_dim)
        self.to_res_bg = _mlp(slot_dim, hidden_dim, slot_dim)

    def _update(self, gru: nn.GRUCell, res: nn.Module, updates: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        b, k, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, k, d)
        return slots + _drop_path(res(slots), self.training)

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(N, 2) [x, y] in [0, 1] of a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x, y], dim=-1).reshape(n_tokens, 2)

    def anneal(self, step: int) -> None:
        """Cosine-anneal the noise sigma."""
        self.sigma = cosine(step / SIGMA_STEPS, SIGMA_START, SIGMA_FINAL)

    def forward(self, feat: torch.Tensor):
        b, n = feat.shape[:2]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_fg = start[:, :1], start[:, 1:]
        mu0 = self.pos_init.expand(b, -1, -1)  # (b, K-1, 2)
        mu = mu0

        x = self.norm_feat(self.feat_mlp(feat))
        k, v = self.to_k(x), self.to_v(x)  # (b, N, D), no positions
        xy = self._grid(n, feat.device)  # (N, 2)
        k_bg = self.grid_enc(k + self.grid_proj(xy))  # absolute

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_fg = slot_fg.detach() + start[:, 1:] - start[:, 1:].detach()
                mu = mu.detach() + mu0 - mu0.detach()
            k_fg = self.grid_enc(k[:, None] + self.grid_proj(xy - mu[:, :, None]))  # (b, K-1, N, D)
            dots_bg = torch.einsum("bkd,bnd->bkn", self.to_q_bg(slot_bg), k_bg) * self.scale
            dots_fg = torch.einsum("bkd,bknd->bkn", self.to_q(slot_fg), k_fg) * self.scale
            attn = torch.cat([dots_bg, dots_fg], dim=1).softmax(dim=1)  # (b, K, N)
            attn_wm = attn / (attn.sum(-1, keepdim=True) + self.eps)
            updates = torch.einsum("bkn,bnd->bkd", attn_wm, v)
            updates_bg, updates_fg = updates[:, :1], updates[:, 1:]
            slot_bg = self._update(self.gru_bg, self.to_res_bg, updates_bg, slot_bg)
            slot_fg = self._update(self.gru, self.to_res, updates_fg, slot_fg)
            # Positions: the object attention's centres.
            w = attn[:, 1:] / attn[:, 1:].sum(-1, keepdim=True)
            mu = torch.einsum("bkn,nc->bkc", w, xy)

        return slot_bg[:, 0], slot_fg, attn
