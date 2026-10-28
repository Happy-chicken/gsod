"""Slot attention over object slots, with a learned background query.

Only objects get slots. The background (a uniform colour in the data, drawn by
GSODModel's static BackgroundShell) is not a slot: it has no latent and is not
updated. It is represented in the attention by one learned query vector, so
background tokens have somewhere to go instead of being spread over the
object slots:

    attention row 0     learned background query (fixed, input-independent)
    attention rows 1..K object slots (learned initial queries, iteratively refined)

The softmax over rows 0..K makes every token choose between "background" and
one of the objects.

Discovery position prior (position_weight > 0, set by GSODModel early in
training): object slot k's logits get a bias that peaks at its own point
c_k of the image (the slots' points are spread evenly over the image):

    bias_k(token) = PRIOR_STRENGTH * (1 - |p - c_k|^2 / sigma^2),
    sigma = 0.5 / sqrt(K)

so it wins the tokens around c_k against the background row and loses the
far ones. Each slot then starts out describing its own part of the image,
and its Gaussians sit on the ray through that part (attention-anchored
frames), so what a slot sees and where it draws agree from the first step.
The bias is scaled by position_weight and fades out as training proceeds.

Slot initialisation: BO-QSA (Jia et al., ICLR 2023, "Improving Object-centric
Learning with Query Optimization"). Every object slot has its own learned
initial query. The refinement iterations run on detached slots, and the
gradient of the last iteration is passed straight through to the queries.
Learned queries trained without this scheme (plain QSA) are worse than random
initialisation in the paper's ablation; the detach + straight-through pair is
what makes them work.

Original Slot Attention initialisation (replaced, kept for reference): all
object slots sampled from one shared learnable Gaussian, with gradients
through every iteration. Slots then start nearly identical and tend to split
one object equally between them (fg_slot_cos_mean 0.45 vs 0.33, attention
FG-ARI 0.71 vs 0.85 with BO-QSA in our runs).

    # __init__
    #     self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
    #     self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
    #     init.xavier_uniform_(self.slots_logsigma)
    # forward
    #     mu = self.slots_mu.expand(b, self.num_slots, -1)
    #     sigma = self.slots_logsigma.exp().expand(b, self.num_slots, -1)
    #     slots = mu + sigma * torch.randn_like(mu)
    #     for _ in range(self.iters):     # no detach, no straight-through
    #         ...
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import init

# Logit bonus of an object slot at its own image point (fades with distance).
PRIOR_STRENGTH = 2.0


def _sunflower(n: int, radius: float) -> torch.Tensor:
    """(n, 2) points spread evenly over a disc of the given radius."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    r = radius * torch.sqrt(i / n)
    theta = math.pi * (3.0 - math.sqrt(5.0)) * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


class SlotAttention(nn.Module):
    """BxNxC tokens -> (BxKxC object slots, Bx(1+K)xN attention)."""

    def __init__(
        self,
        num_slots: int,
        in_dim: int = 64,
        slot_dim: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        """num_slots: number of object slots K (the background is extra)."""
        super().__init__()
        assert num_slots >= 1, "need at least one object slot"
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim**-0.5
        self.slot_dim = slot_dim

        # BO-QSA: one learned initial query per object slot.
        self.slot_queries = nn.Parameter(torch.empty(1, num_slots, slot_dim))
        init.xavier_uniform_(self.slot_queries)
        # Background: one learned query in key space (not a slot).
        self.bg_query = nn.Parameter(torch.randn(1, 1, slot_dim) * slot_dim**-0.5)

        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(
            nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False)
        )
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)
        self.to_res = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, slot_dim),
        )
        # Soft position embedding + feature MLP (Locatello et al. 2020).
        self.pos_proj = nn.Linear(4, in_dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
        )
        self.norm_feat = nn.LayerNorm(in_dim)
        # Discovery prior: one image point per object slot, in [0, 1]^2.
        self.register_buffer(
            "prior_points", 0.5 + _sunflower(num_slots, 0.4), persistent=False
        )
        self.prior_sigma = 0.5 / math.sqrt(num_slots)

    @staticmethod
    def _grid(n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, N, 4) [x, y, 1-x, 1-y] in [0, 1] for a square token grid."""
        side = int(round(n_tokens**0.5))
        assert side * side == n_tokens, "tokens must form a square grid"
        lin = torch.linspace(0.0, 1.0, side, device=device)
        y, x = torch.meshgrid(lin, lin, indexing="ij")
        grid = torch.stack([x, y, 1.0 - x, 1.0 - y], dim=-1)
        return grid.reshape(1, n_tokens, 4)

    def project_inputs(
        self, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BxNxC tokens -> (keys, values), each BxNxD.

        The keys are the space the attention compares slots in: a token can
        only be told apart from another if their keys differ.
        """
        feat = feat + self.pos_proj(self._grid(feat.shape[1], feat.device))
        feat = self.norm_feat(self.feat_mlp(feat))
        return self.to_k(feat), self.to_v(feat)

    def position_bias(self, n_tokens: int, device: torch.device) -> torch.Tensor:
        """(1, K, N) discovery logit bias of the object slots (see module docstring)."""
        xy = self._grid(n_tokens, device)[0, :, :2]  # (N, 2) token positions
        d2 = (xy[None] - self.prior_points[:, None].to(xy.dtype)).square().sum(-1)
        return (PRIOR_STRENGTH * (1.0 - d2 / self.prior_sigma**2))[None]

    def forward(self, feat: torch.Tensor, position_weight: float = 0.0):
        """position_weight: scale of the discovery position bias (0 = off)."""
        b = feat.shape[0]
        slots_init = self.slot_queries.expand(b, -1, -1)
        slots = slots_init

        k, v = self.project_inputs(feat)
        dots_bg = torch.einsum("bid,bjd->bij", self.bg_query.expand(b, -1, -1), k) * self.scale
        bias = (
            position_weight * self.position_bias(feat.shape[1], feat.device)
            if position_weight > 0
            else 0.0
        )

        attn = None
        for it in range(self.iters):
            # BO-QSA: earlier iterations are not back-propagated; the last one
            # passes its gradient straight through to the initial queries.
            slots = slots.detach()
            if it == self.iters - 1:
                slots = slots + slots_init - slots_init.detach()
            slots_prev = slots
            q = self.to_q(slots)
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale + bias
            attn = torch.cat([dots_bg, dots], dim=1).softmax(dim=1) + self.eps
            attn_obj = attn[:, 1:, :]
            attn_w = attn_obj / attn_obj.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn_w)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, -1, self.slot_dim)
            slots = slots + self.to_res(slots)

        return slots, attn
