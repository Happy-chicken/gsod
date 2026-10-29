"""MLP decoders for slots.

Two minimal variants:

* ``PerSlotMLPDecoder``: one MLP per slot (not permutation-equivariant;
  kept for comparison).
* ``SharedMLPDecoder``: one MLP shared by all slots.

The first three raw values are offsets inside each slot's ``SlotFrame``.
"""

from __future__ import annotations

import torch
from torch import nn

from gsod.model.decoders.parameter_head import (
    PARAMS_PER_GAUSSIAN,
    GaussianParameterHead,
    SlotFrame,
    init_raw_bias,
)


def _make_mlp(z_dim: int, hidden_dim: int, n_gaussians: int) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.ReLU(True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(True),
        nn.Linear(hidden_dim, n_gaussians * PARAMS_PER_GAUSSIAN),
    )
    with torch.no_grad():
        last = mlp[-1]
        last.weight.mul_(0.1)
        # Per-Gaussian bias rows: random spread inside the slot frame.
        init_raw_bias(last.bias.view(n_gaussians, PARAMS_PER_GAUSSIAN))
    return mlp


class Decoder(nn.Module):
    def __init__(
        self,
        n_slots: int,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        frame: SlotFrame,
    ):
        """Common decoder signature. n_slots: number of slots."""
        super().__init__()
        assert n_slots >= 1, "n_slots must be at least 1"
        self.n_slots = int(n_slots)
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_gaussians = int(gaussians_per_slot)
        self.head = GaussianParameterHead(frame)

    def _pack(self, raw: torch.Tensor, slots: torch.Tensor):
        ids = torch.arange(self.n_slots, device=slots.device)
        return self.head(raw, slots, ids.repeat_interleave(self.n_gaussians))

    def _check(self, slots: torch.Tensor) -> None:
        assert slots.shape[0] == self.n_slots, (
            f"decoder built for {self.n_slots} slots, got {slots.shape[0]}"
        )


class PerSlotMLPDecoder(Decoder):
    """One independent MLP per slot."""

    def __init__(self, n_slots, z_dim, gaussians_per_slot, hidden_dim, frame):
        super().__init__(n_slots, z_dim, gaussians_per_slot, hidden_dim, frame)
        self.mlps = nn.ModuleList(
            [
                _make_mlp(z_dim, hidden_dim, self.n_gaussians)
                for _ in range(self.n_slots)
            ]
        )

    def forward(self, slots: torch.Tensor):
        self._check(slots)
        raw = torch.cat(
            [
                mlp(slots[k]).view(self.n_gaussians, PARAMS_PER_GAUSSIAN)
                for k, mlp in enumerate(self.mlps)
            ],
            dim=0,
        )
        return self._pack(raw, slots)


class SharedMLPDecoder(Decoder):
    """One MLP shared by all slots."""

    def __init__(self, n_slots, z_dim, gaussians_per_slot, hidden_dim, frame):
        super().__init__(n_slots, z_dim, gaussians_per_slot, hidden_dim, frame)
        self.fg_mlp = _make_mlp(z_dim, hidden_dim, self.n_gaussians)

    def forward(self, slots: torch.Tensor):
        self._check(slots)
        raw = self.fg_mlp(slots).view(-1, PARAMS_PER_GAUSSIAN)
        return self._pack(raw, slots)
