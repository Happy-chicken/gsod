"""Cross-attention Gaussian decoder for slots.

    Q: G learned Gaussian queries, shared by all slots and independent of the
       input. Query i stands for "Gaussian i" of a slot.
    K/V: n_tokens tokens per slot, linear projections of that slot's latent.
    output: one feature per (slot, query) -> head (parameter_head.py) ->
       one Gaussian.

The slot decides what is drawn (only through K/V); the query decides which
Gaussian of the slot is produced.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        head: nn.Module,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        query_std: float = 2.0,
    ):
        """Works for any number of slots; hidden_dim is the attention width
        d_model; head maps (features (K, G, d), slots (K, C)) to Gaussians."""
        super().__init__()
        d_model = int(hidden_dim)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_gaussians = int(gaussians_per_slot)
        self.M = int(n_tokens)
        self.d = int(d_model)

        # Learned Gaussian queries (G, d), shared by all slots. They are the
        # only per-Gaussian parameters, so their spread decides how far apart
        # the Gaussians of one slot start: std ~0.07 gives tanh(offset) std
        # ~0.02 (every slot collapses to a point); std 2.0 gives ~0.4, about
        # U(-0.7, 0.7) of the head's region.
        self.queries = nn.Parameter(torch.randn(self.n_gaussians, d_model) * query_std)
        self.k_proj = nn.Linear(z_dim, n_tokens * d_model)
        self.v_proj = nn.Linear(z_dim, n_tokens * d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.head = head

    def _attend(self, q, k, v):
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        return q + self.ffn(self.norm_ff(q))

    def forward(self, slots: torch.Tensor, **context) -> dict[str, torch.Tensor]:
        """slots: (K, C) -> Gaussians (K, G, ...), grouped by slot. context:
        extra per-slot inputs of the head."""
        K = slots.shape[0]
        assert K >= 1
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        q = self.queries.unsqueeze(0).expand(K, -1, -1)
        return self.head(self._attend(q, k, v), slots, **context)
