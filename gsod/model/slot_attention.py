"""Slot attention: background as in uORF (own query, GRU, residual MLP); the
rest as in BO-QSA: tokens + position embedding through a feature MLP, learned
per-slot queries with annealed noise, weighted mean with eps = 1, DropPath on
the residual MLP, bi-level (only the last iteration is differentiated, straight
through to the queries). Slot 0 is the background.

Then one more object-slot update, whose attention is returned: the anchors and
draw factor use it, so their gradient reaches the last slot update (bi-level
alone would stop it). Its object keys get a learned bias b."""

from __future__ import annotations

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

        hidden_dim = max(slot_dim, hidden_dim)
        self.pos_proj = nn.Linear(4, in_dim)
        self.feat_mlp = _mlp(in_dim, hidden_dim, slot_dim)
        self.norm_feat = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)
        self.to_res = _mlp(slot_dim, hidden_dim, slot_dim)
        self.to_res_bg = _mlp(slot_dim, hidden_dim, slot_dim)
        self.fg_bias = nn.Parameter(torch.zeros(slot_dim))

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

    def _keys_values(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens (b, N, C) + position embedding -> keys, values (b, N, D)."""
        feat = self.feat_mlp(feat + self.pos_proj(self._grid(feat.shape[1], feat.device)))
        feat = self.norm_feat(feat)
        return self.to_k(feat), self.to_v(feat)

    def _attend(self, slot_bg, slot_fg, k, v, k_fg):
        """-> attention (b, K, N) and the weighted means of v (b, K, D)."""
        dots_bg = torch.einsum("bid,bjd->bij", self.to_q_bg(slot_bg), k) * self.scale
        dots_fg = torch.einsum("bid,bjd->bij", self.to_q(slot_fg), k_fg) * self.scale
        attn = torch.cat([dots_bg, dots_fg], dim=1).softmax(dim=1)
        attn_wm = attn / (attn.sum(-1, keepdim=True) + self.eps)
        return attn, torch.einsum("bij,bjd->bid", attn_wm, v)

    def forward(self, feat: torch.Tensor):
        b = feat.shape[0]
        start = self.slots_init.weight.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            start = start + torch.randn_like(start) * self.sigma * start.detach()
        slot_bg, slot_fg = start[:, :1], start[:, 1:]

        k, v = self._keys_values(feat)

        for i in range(self.iters):
            if i == self.iters - 1:  # bi-level
                slot_bg = slot_bg.detach() + start[:, :1] - start[:, :1].detach()
                slot_fg = slot_fg.detach() + start[:, 1:] - start[:, 1:].detach()
            _, updates = self._attend(slot_bg, slot_fg, k, v, k)
            updates_bg, updates_fg = updates[:, :1], updates[:, 1:]
            slot_bg = self._update(self.gru_bg, self.to_res_bg, updates_bg, slot_bg)
            slot_fg = self._update(self.gru, self.to_res, updates_fg, slot_fg)

        # One more object-slot update (object keys + b); its attention is returned.
        attn, updates = self._attend(slot_bg, slot_fg, k, v, k + self.fg_bias)  # (b, K, N)
        slot_fg = self._update(self.gru, self.to_res, updates[:, 1:], slot_fg)
        return slot_bg[:, 0], slot_fg, attn
