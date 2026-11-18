"""Slot attention as SlotODE; slot 0 is the background.

Input as in BO-QSA (tokens + position embedding through a feature MLP). Slots
start from the tokens aggregated along fixed orthogonal directions (SlotODE)
and follow a learned vector field (scaled spectral norm, DyT, tanh, time
embedding), integrated by explicit Euler with unit steps (one evaluation per
step, as cheap as the usual iterations; a step moves each coordinate by up to
1, like a GRU step) and differentiated through the solver. The background
has its own query and vector field (uORF).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm

# Spectral scale rho of the vector field's linear layers (SlotODE uses 3).
SPECTRAL_SCALE = 3.0


class ScaledSNLinear(nn.Module):
    """Linear layer with W = rho * W / sigma(W)."""

    def __init__(self, d_in: int, d_out: int, rho: float = SPECTRAL_SCALE):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(d_in, d_out, bias=False))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.rho = rho

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho * self.linear(x) + self.bias


class DyT(nn.Module):
    """Dynamic tanh (Zhu et al., 2025): gamma * tanh(alpha * x) + beta."""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((), 0.5))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class VectorField(nn.Module):
    """[slot; evidence; time] (3D) -> ds/dt (D), each coordinate in (-1, 1)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ScaledSNLinear(3 * dim, hidden_dim),
            DyT(hidden_dim),
            ScaledSNLinear(hidden_dim, hidden_dim),
            DyT(hidden_dim),
            ScaledSNLinear(hidden_dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sinusoidal(t: float, dim: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal embedding (dim,) of a time t in [0, 1] (scaled to [0, 1000])."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
    angles = 1000.0 * t * freqs
    return torch.cat([angles.sin(), angles.cos()])


def _orthogonal(k: int, d: int) -> torch.Tensor:
    """(k, d) fixed orthonormal rows (seeded QR)."""
    assert d >= k, "slot_dim must be at least num_slots"
    m = torch.randn(d, k, generator=torch.Generator().manual_seed(0))
    return torch.linalg.qr(m)[0].T.contiguous()


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
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim
        hidden_dim = max(slot_dim, hidden_dim)

        self.register_buffer("directions", _orthogonal(num_slots, slot_dim))
        self.pos_proj = nn.Linear(4, in_dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, slot_dim)
        )
        self.norm_feat = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.time_mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.GELU(), nn.Linear(slot_dim, slot_dim))
        self.field = VectorField(slot_dim, hidden_dim)
        self.field_bg = VectorField(slot_dim, hidden_dim)

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] of a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1).reshape(1, n_tokens, 4)

    def _velocity(self, s: torch.Tensor, phi: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ds/dt of the slots s (b, K, D) at the time embedded as phi (D,), and
        the attention there."""
        b, n_slots, d = s.shape
        q = torch.cat([self.to_q_bg(s[:, :1]), self.to_q(s[:, 1:])], 1)
        attn = torch.einsum("bkd,bnd->bkn", q, k).mul(self.scale).softmax(dim=1)
        u = torch.einsum("bkn,bnd->bkd", attn / (attn.sum(-1, keepdim=True) + self.eps), v)
        x = torch.cat([s, u, phi.expand(b, n_slots, d)], -1)
        return torch.cat([self.field_bg(x[:, :1]), self.field(x[:, 1:])], 1), attn

    def forward(self, feat: torch.Tensor):
        feat = self.feat_mlp(feat + self.pos_proj(self._grid(feat.shape[1], feat.device)))
        feat = self.norm_feat(feat)
        k, v = self.to_k(feat), self.to_v(feat)

        # Initial slots: tokens aggregated along fixed orthogonal directions.
        a = torch.einsum("kd,bnd->bkn", self.directions, feat).softmax(dim=1)
        s = torch.einsum("bkn,bnd->bkd", a, feat) / a.sum(-1, keepdim=True)

        h = 1.0  # unit steps: time runs over [0, iters]
        times = torch.stack([_sinusoidal(i / self.iters, self.slot_dim, feat.device) for i in range(self.iters)])
        phis = self.time_mlp(times)  # (iters, D)
        attn = None
        with parametrize.cached():  # spectral norms once per forward
            for phi in phis:  # explicit Euler
                ds, attn = self._velocity(s, phi, k, v)
                s = s + h * ds

        return s[:, 0], s[:, 1:], attn
