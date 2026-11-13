"""Slot attention (BO-QSA, Jia et al., ICLR 2023); slot 0 is the background.

Every slot starts from its own learned query, so slot identity is fixed and
slot 0 (decoded by the background decoder) learns to be the background. All
modules are shared. Training adds mu + eps * sigma * mu.detach() to the
initial queries, sigma cosine-annealed SIGMA_START -> SIGMA_FINAL over SIGMA_STEPS steps (anneal;
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
        self.sigma = SIGMA_START  # initial-query noise, see anneal

        self.slots_init = nn.Embedding(num_slots, slot_dim)
        init.xavier_uniform_(self.slots_init.weight)
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
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.mlp = _residual_mlp(slot_dim, hidden_dim)

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

    @torch.no_grad()
    def _reseed(self, attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor, feat: torch.Tensor):
        """Move duplicate object slots (1..K-1) to unclaimed tokens. attn
        (b, K, N), queries (b, K, D), keys, features (b, N, D) -> new slot
        values (b, K, D), the moved mask (b, K) (slot 0 never moves)."""
        a = attn.clone()
        a[:, 0] = 0.0  # the background is never a duplicate nor a stronger slot
        mass = a.sum(-1)  # (b, K)
        shared = torch.minimum(a[:, :, None], a[:, None]).sum(-1)  # (b, K, K)
        weaker = mass[:, :, None] < mass[:, None]  # [j, l]: j weaker than l
        duplicate = ((shared / mass[:, :, None].clamp_min(1e-8) > DUPLICATE) & weaker).any(-1)  # (b, K)
        unclaimed = 1.0 - torch.einsum("bkd,bnd->bkn", F.normalize(q, dim=-1), F.normalize(k, dim=-1)).amax(1)
        new = torch.zeros_like(q)
        moved = torch.zeros_like(duplicate)
        for i in range(q.shape[0]):
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

        mu = self.slots_init.weight.expand(b, -1, -1)
        slots_init = mu
        if self.training and self.sigma > 0:
            slots_init = mu + torch.randn_like(mu) * self.sigma * mu.detach()
        slots = slots_init

        attn = None
        for i in range(self.iters):
            if i == 1:  # after the first iteration: duplicates to unclaimed tokens
                new, moved = self._reseed(attn, self.project_q(self.norm_slots(slots)), k, feat)
                slots = torch.where(moved[..., None], new, slots)
                slots_init = torch.where(moved[..., None], new, slots_init)
            if i == self.iters - 1:  # bi-level
                slots = slots.detach() + slots_init - slots_init.detach()
            prev = slots
            q = self.project_q(self.norm_slots(slots))
            attn = torch.einsum("bkd,bnd->bkn", q, k).mul(D**-0.5).softmax(dim=1)  # (b, K, N)
            attn_wm = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bkn,bnd->bkd", attn_wm, v)
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(b, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots[:, 0], slots[:, 1:], attn
