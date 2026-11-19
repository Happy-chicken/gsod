"""Slot attention: background as in uORF (own query, GRU, residual MLP); the
rest as in BO-QSA: tokens + position embedding through a feature MLP, learned
per-slot queries with annealed noise, weighted mean with eps = 1, DropPath on
the residual MLP, bi-level (only the last iteration is differentiated, straight
through to the initial state). Slot 0 is the background.

Object slots carry a position mu and scale sigma (attention mean / std, as in
Invariant Slot Attention) and see token positions relative to them,
(x - mu) / sigma, so a slot prefers tokens near its own centre; the background
sees absolute positions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
        # Object slots' initial positions (spread on a grid) and scales (0.5).
        n_obj = num_slots - 1
        side = math.ceil(math.sqrt(n_obj))
        lin = (torch.arange(side) + 0.5) / side
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="xy"), -1).reshape(-1, 2)[:n_obj]
        self.pos_init = nn.Parameter(grid.clone())
        self.scale_init = nn.Parameter(torch.full((n_obj, 2), math.log(math.expm1(0.5))))

        hidden_dim = max(slot_dim, hidden_dim)
        self.pos_proj = nn.Linear(4, in_dim)
        self.pos_proj_rel = nn.Linear(2, in_dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )
        self.norm_feat = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )
        self.to_res_bg = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )

    def _update(self, gru: nn.GRUCell, res: nn.Module, updates: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        b, k, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, k, d)
        return slots + _drop_path(res(slots), self.training)

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
        self.sigma = cosine(step / SIGMA_STEPS, SIGMA_START, SIGMA_FINAL)

    def _tokens(self, feat: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens (..., N, C) with a position embedding (..., N, C) -> keys, values."""
        x = self.norm_feat(self.feat_mlp(feat + pos))
        return self.to_k(x), self.to_v(x)

    def forward(self, feat: torch.Tensor):
        b, n = feat.shape[:2]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_fg = start[:, :1], start[:, 1:]
        mu0 = self.pos_init.expand(b, -1, -1)  # (b, K-1, 2)
        sigma0 = F.softplus(self.scale_init).expand(b, -1, -1)
        mu, sigma = mu0, sigma0

        grid = self._grid(n, feat.device)
        xy = grid[0, :, :2]  # (N, 2)
        k_bg, v_bg = self._tokens(feat, self.pos_proj(grid))  # absolute
        floor = 1.0 / math.isqrt(n)  # one token

        attn = None
        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_fg = slot_fg.detach() + start[:, 1:] - start[:, 1:].detach()
                mu = mu.detach() + mu0 - mu0.detach()
                sigma = sigma.detach() + sigma0 - sigma0.detach()
            rel = (xy - mu[:, :, None]) / sigma[:, :, None]  # (b, K-1, N, 2)
            k_fg, v_fg = self._tokens(feat[:, None], self.pos_proj_rel(rel))  # (b, K-1, N, D)
            dots_bg = torch.einsum("bkd,bnd->bkn", self.to_q_bg(slot_bg), k_bg) * self.scale
            dots_fg = torch.einsum("bkd,bknd->bkn", self.to_q(slot_fg), k_fg) * self.scale
            attn = torch.cat([dots_bg, dots_fg], dim=1).softmax(dim=1)  # (b, K, N)
            attn_wm = attn / (attn.sum(-1, keepdim=True) + self.eps)
            updates_bg = torch.einsum("bkn,bnd->bkd", attn_wm[:, :1], v_bg)
            updates_fg = torch.einsum("bkn,bknd->bkd", attn_wm[:, 1:], v_fg)
            slot_bg = self._update(self.gru_bg, self.to_res_bg, updates_bg, slot_bg)
            slot_fg = self._update(self.gru, self.to_res, updates_fg, slot_fg)
            # Positions and scales: the object attention's mean and std.
            w = attn[:, 1:] / attn[:, 1:].sum(-1, keepdim=True)
            mu = torch.einsum("bkn,nc->bkc", w, xy)
            var = torch.einsum("bkn,bknc->bkc", w, (xy - mu[:, :, None]) ** 2)
            sigma = var.clamp_min(0.0).sqrt() + floor

        return slot_bg[:, 0], slot_fg, attn
