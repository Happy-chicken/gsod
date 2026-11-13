"""Slot attention: a background slot (uORF) + K-1 object slots (BO-QSA).

Object slots start from their own learned queries (Jia et al., ICLR 2023);
the background has its own query, query projection, GRU and MLP (uORF).
Training adds mu + eps * sigma * mu.detach() to the initial queries, sigma
cosine-annealed SIGMA_START -> SIGMA_FINAL over SIGMA_STEPS steps (anneal;
none in evaluation). Logits q . k / sqrt(D), softmax over slots; weighted
mean with epsilon 1.0; GRU then residual MLP; the last iteration starts from
the detached slots plus a straight-through path to the initial queries.

After the first iteration a duplicate object slot (most of its attention
shared with a stronger object slot) is moved to the token least like every
slot, if that token is unlike them all (> UNCLAIMED); otherwise it is left
to lose and stay empty.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

# Initial-query noise schedule (BO-QSA defaults).
SIGMA_START = 1.0
SIGMA_FINAL = 0.0
SIGMA_STEPS = 30000
# Duplicates: shared share of the weaker slot's attention; unclaimed: 1 - max cos.
DUPLICATE = 0.5
UNCLAIMED = 0.5


def _residual_mlp(dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, dim))


class SlotAttention(nn.Module):
    """BxNxC tokens -> (BxD background slot, Bx(K-1)xD object slots, BxKxN
    attention, softmax over the slots, row 0 the background)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1.0,
        hidden_dim: int = 128,
    ):
        """num_slots: K, the background slot included (K-1 object slots)."""
        super().__init__()
        assert num_slots >= 2, "need at least two slots"
        self.iters = iters
        self.eps = eps
        self.slot_dim = slot_dim
        self.n_objects = num_slots - 1
        self.sigma = SIGMA_START  # initial-query noise, see anneal

        self.bg_query = nn.Parameter(torch.empty(1, 1, slot_dim))
        self.fg_query = nn.Parameter(torch.empty(1, self.n_objects, slot_dim))
        init.xavier_uniform_(self.bg_query)
        init.xavier_uniform_(self.fg_query)
        # Position embedding + feature MLP on the encoder tokens.
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
        # Objects and background: own query projection, GRU and MLP each.
        self.norm_slots, self.norm_slots_bg = nn.LayerNorm(slot_dim), nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_q_bg = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru, self.gru_bg = nn.GRUCell(slot_dim, slot_dim), nn.GRUCell(slot_dim, slot_dim)
        self.norm_mlp, self.norm_mlp_bg = nn.LayerNorm(slot_dim), nn.LayerNorm(slot_dim)
        self.mlp = _residual_mlp(slot_dim, hidden_dim)
        self.mlp_bg = _residual_mlp(slot_dim, hidden_dim)

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] in [0, 1] for a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1).reshape(1, n_tokens, 4)

    def anneal(self, step: int) -> None:
        """Noise sigma at a training step: cosine SIGMA_START -> SIGMA_FINAL."""
        progress = min(step / SIGMA_STEPS, 1.0)
        a, b = 0.5 * (SIGMA_START - SIGMA_FINAL), 0.5 * (SIGMA_START + SIGMA_FINAL)
        self.sigma = a * math.cos(math.pi * progress) + b

    def _queries(self, bg: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
        """(b, 1, D), (b, K-1, D) -> queries (b, K, D)."""
        return torch.cat([self.project_q_bg(self.norm_slots_bg(bg)), self.project_q(self.norm_slots(fg))], 1)

    def _update(self, gru, norm, mlp, updates, prev):
        b, n, d = prev.shape
        slots = gru(updates.reshape(-1, d), prev.reshape(-1, d)).reshape(b, n, d)
        return slots + mlp(norm(slots))

    @torch.no_grad()
    def _reseed(self, fg: torch.Tensor, attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, feat: torch.Tensor):
        """Move duplicate object slots to unclaimed tokens. fg (b, K-1, D),
        attn (b, K, N), queries (b, K, D), keys (b, N, D), features (b, N, D)
        -> new slot values (b, K-1, D), the moved mask (b, K-1)."""
        a = attn[:, 1:]  # (b, K-1, N)
        mass = a.sum(-1)  # (b, K-1)
        shared = torch.minimum(a[:, :, None], a[:, None]).sum(-1)  # (b, K-1, K-1)
        weaker = mass[:, :, None] < mass[:, None]  # [j, l]: j weaker than l
        duplicate = ((shared / mass[:, :, None].clamp_min(1e-8) > DUPLICATE) & weaker).any(-1)  # (b, K-1)
        unclaimed = 1.0 - torch.einsum("bkd,bnd->bkn", F.normalize(q, dim=-1), F.normalize(k, dim=-1)).amax(1)
        new = torch.zeros_like(fg)
        moved = torch.zeros_like(duplicate)
        for i in range(fg.shape[0]):
            u = unclaimed[i].clone()
            for j in duplicate[i].nonzero().flatten().tolist():
                n = int(u.argmax())
                if u[n] <= UNCLAIMED:
                    break
                new[i, j] = feat[i, n]
                moved[i, j] = True
                u[n] = -1.0
        return new, moved

    def forward(self, feat: torch.Tensor):
        """feat (b, N, C) -> background slot (b, D), object slots (b, K-1, D),
        attention (b, K, N)."""
        b, D = feat.shape[0], self.slot_dim
        feat = self.feat_mlp(feat + self.pos_proj(self._grid(feat.shape[1], feat.device)))
        feat = self.norm_feature(feat)
        k, v = self.project_k(feat), self.project_v(feat)

        bg_init = self.bg_query.expand(b, -1, -1)
        fg_init = self.fg_query.expand(b, -1, -1)
        if self.training and self.sigma > 0:
            bg_init = bg_init + torch.randn_like(bg_init) * self.sigma * bg_init.detach()
            fg_init = fg_init + torch.randn_like(fg_init) * self.sigma * fg_init.detach()
        bg, fg = bg_init, fg_init

        attn = None
        for i in range(self.iters):
            if i == 1:  # after the first iteration: duplicates to unclaimed tokens
                new, moved = self._reseed(fg, attn, self._queries(bg, fg), k, feat)
                fg = torch.where(moved[..., None], new, fg)
                fg_init = torch.where(moved[..., None], new, fg_init)
            if i == self.iters - 1:  # bi-level
                bg = bg.detach() + bg_init - bg_init.detach()
                fg = fg.detach() + fg_init - fg_init.detach()
            q = self._queries(bg, fg)
            attn = torch.einsum("bkd,bnd->bkn", q, k).mul(D**-0.5).softmax(dim=1)  # (b, K, N)
            attn_wm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bkn,bnd->bkd", attn_wm, v)
            bg = self._update(self.gru_bg, self.norm_mlp_bg, self.mlp_bg, updates[:, :1], bg)
            fg = self._update(self.gru, self.norm_mlp, self.mlp, updates[:, 1:], fg)

        return bg[:, 0], fg, attn
