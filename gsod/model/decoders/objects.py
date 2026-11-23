"""Object decoder: slot -> box at the attention anchor -> G / M queries, each a
sub-box in the box holding M Gaussians (G in all).
The anchor: where the slot's view-0 tokens meet the floor, attention-weighted.
Box sizes, shifts and Gaussian scales unbounded; BOX_INIT only sets where they start.
Opacity = per-slot share of its brightest Gaussian x presence (attention area /
AREA_FULL, at most 1): the attention decides whether a slot shows, never 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.background import BACKGROUND_THICKNESS, FLOOR_HALF
from gsod.model.layers import mlp, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions

# Attention area (fraction of tokens) at which a slot is fully present.
AREA_FULL = 0.004
# Initial box half side (units of R; an initial value, not a limit).
BOX_INIT = 0.25
# Initial Gaussian scale, as a fraction of the initial box (not the current one).
SCALE_FRACTION = 0.1
# Spread of the learned queries (how far apart Gaussians start).
QUERY_STD = 2.0
# Transformer: heads, slot tokens, layers; RegionHead MLP width.
N_HEADS, N_TOKENS, N_LAYERS = 4, 8, 2
REGION_HIDDEN = 64
# ObjectHead outputs per query (sub-centre, log sub-size) and per Gaussian.
_QUERY_OUT, _CHILD_OUT = 6, 23


# Anchors weight tokens by attn ** this (lean to the stronger core).
_ANCHOR_SHARPNESS = 2.0


def _token_rays(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(..., side*side, 3) scene-frame directions of the tokens' view-0 rays."""
    h, w = image_size
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return camera_directions(uu, vv, image_size, focal_ratio).reshape(-1, 3) @ frame.rot.mT


def _at_height(frame: SceneFrame, directions: torch.Tensor, z: float) -> torch.Tensor:
    """Where the rays (..., N, 3) reach height z (..., N, 2); rays not going down stay at the camera."""
    t = ((z - frame.origin[..., 2:]) / directions[..., 2].clamp(max=-1e-6)).clamp(min=0.0)
    return frame.origin[..., None, :2] + t[..., None] * directions[..., :2]


def token_points(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(..., side*side, 3): where each token's view-0 ray meets the floor (within it)."""
    xy = _at_height(frame, _token_rays(frame, side, image_size, focal_ratio), 0.0)
    xy = xy.clamp(-FLOOR_HALF, FLOOR_HALF)
    return torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)


def attention_anchors(
    attn: torch.Tensor, points: torch.Tensor, colours: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (..., K, N), token points and colours (..., N, 3) -> where each
    slot's attention points and the colour there (..., K, 3)."""
    p = (_ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(dim=-1)
    return p @ points, p @ colours


def _child_pattern(m: int) -> torch.Tensor:
    """(m, 3) initial child offsets in [-1, 1]^3, spread through the unit ball
    (Fibonacci sphere, radius growing as cbrt); one child sits at the centre."""
    if m == 1:
        return torch.zeros(1, 3)
    i = torch.arange(m, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * i / m
    r_xy = (1.0 - z * z).clamp_min(0.0).sqrt()
    angle = i * math.pi * (3.0 - math.sqrt(5.0))
    unit = torch.stack([r_xy * angle.cos(), r_xy * angle.sin(), z], -1)
    return unit * (i / m).pow(1.0 / 3.0)[:, None]


class RegionHead(nn.Module):
    """Slot latent -> box shift (along / across view 0's ray) and half sizes,
    unbounded, in units of BOX_INIT; the box stands on the floor. Starts as a
    box of half side BOX_INIT at the anchor.

    One hidden layer, two zero-initialised output layers (shift, log size).
    Everything is built with whole-row (K, 3) arithmetic, no slice + cat: a
    concatenated column of a Linear output made Inductor emit a Triton load at
    a constant index (tl.broadcast_to(4 + (0), ...)) that does not compile."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(z_dim, REGION_HIDDEN), nn.ReLU())
        self.to_shift = nn.Linear(REGION_HIDDEN, 2)
        self.to_size = nn.Linear(REGION_HIDDEN, 3)
        for out in (self.to_shift, self.to_size):
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.register_buffer("flat", torch.tensor([1.0, 1.0, 0.0]), persistent=False)
        self.register_buffer("up", torch.tensor([0.0, 0.0, 1.0]), persistent=False)
        # (x, y, z) @ turn.mT = (-y, x, 0): a quarter turn about +z.
        turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.register_buffer("turn", turn, persistent=False)

    def forward(
        self, slots: torch.Tensor, anchor: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> centre, half sizes (K, 3)."""
        h = self.hidden(slots)
        shift = BOX_INIT * self.to_shift(h)
        extent = BOX_INIT * torch.exp(self.to_size(h))
        along = F.normalize(ray * self.flat, dim=-1)
        across = along @ self.turn.mT
        # On the floor at the shifted anchor, raised by its half height.
        centre = anchor * self.flat + shift[:, :1] * along + shift[:, 1:] * across + extent * self.up
        return centre, extent


class ObjectHead(nn.Module):
    """Query feature -> a sub-box inside the slot's box and M Gaussians in it.

    Per query: sub-centre (3, tanh over the box) and log sub-size (3, starting
    at box / Q^(1/3) so the Q sub-boxes tile the box); per child: offset in the
    sub-box (3, tanh), log scale (3), quaternion (4), opacity, rgb residual (3),
    sh1 (9)."""

    def __init__(self, d_model: int, n_queries: int, m: int):
        super().__init__()
        self.m = int(m)
        self.shrink = float(n_queries) ** (1.0 / 3.0)
        self.mlp = mlp(d_model, d_model, _QUERY_OUT + self.m * _CHILD_OUT)
        with torch.no_grad():
            out = self.mlp[-1]
            out.bias.zero_()
            out.weight[3:_QUERY_OUT].mul_(0.1)  # sub-size starts near box / Q^(1/3)
            w = out.weight[_QUERY_OUT:].view(self.m, _CHILD_OUT, -1)
            b = out.bias[_QUERY_OUT:].view(self.m, _CHILD_OUT)
            w.mul_(0.1)  # children start at their pattern, scale, identity rotation
            w[:, 14:].zero_()  # view-independent at first
            b[:, 0:3] = torch.atanh(0.8 * _child_pattern(self.m))
            b[:, 6] = 1.0  # identity quaternion (wxyz)
        # Floor for positions: only z is bounded (above the floor's Gaussians).
        self.register_buffer(
            "floor", torch.tensor([-float("inf"), -float("inf"), BACKGROUND_THICKNESS]), persistent=False
        )

    def forward(
        self,
        feat: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, Q, d), box centre / half sizes / colour (K, 3) -> Gaussians (K, Q*M, ...)."""
        query, child = self.mlp(feat).split([_QUERY_OUT, self.m * _CHILD_OUT], -1)
        child = child.unflatten(-1, (self.m, _CHILD_OUT))  # (K, Q, M, 23)

        # The box, its bottom above the floor's Gaussians.
        lo = torch.maximum(centre - extent, self.floor)[:, None]
        hi = (centre + extent)[:, None]
        sub_centre = lo + (hi - lo) * 0.5 * (torch.tanh(query[..., 0:3]) + 1.0)  # (K, Q, 3)
        sub_extent = 0.5 * (hi - lo) / self.shrink * torch.exp(query[..., 3:6])
        mu = sub_centre[:, :, None] + sub_extent[:, :, None] * torch.tanh(child[..., 0:3])
        mu = torch.maximum(mu, self.floor)

        gaussians = {
            "mu": mu,
            "scale": SCALE_FRACTION * BOX_INIT * torch.exp(child[..., 3:6]),
            "quat": F.normalize(child[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(child[..., 10]),
            "rgb": shift_colour(colour[:, None, None], child[..., 11:14]),
            "sh1": child[..., 14:23].unflatten(-1, (3, 3)),
        }
        return {name: value.flatten(1, 2) for name, value in gaussians.items()}


class DecoderLayer(nn.Module):
    """Self-attention among the queries, cross-attention to the slot tokens,
    FFN; pre-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = mlp(d_model, 2 * d_model, d_model, act=nn.GELU)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q (K, Q, d), slot tokens k, v (K, T, d) -> (K, Q, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross_attn(self.norm_cross(q), k, v, need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians, scene frame."""

    def __init__(
        self, z_dim: int, gaussians_per_slot: int, gaussians_per_query: int, hidden_dim: int,
        image_size: tuple[int, int], focal_ratio: tuple[float, float],
    ):
        super().__init__()
        assert gaussians_per_slot % gaussians_per_query == 0, "G must be a multiple of M"
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        d = int(hidden_dim)
        self.d = d
        n_queries = int(gaussians_per_slot) // int(gaussians_per_query)
        self.queries = nn.Parameter(torch.randn(n_queries, d) * QUERY_STD)
        self.to_k = nn.Linear(z_dim, N_TOKENS * d)
        self.to_v = nn.Linear(z_dim, N_TOKENS * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, N_HEADS) for _ in range(N_LAYERS))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d, n_queries, gaussians_per_query)

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Object slots (B, K, C), their attention (B, K, N), token colours
        (B, N, 3), view 0's SceneFrames -> Gaussians (B, K, G, ...), half sizes
        (B, K, 3); scene frame."""
        b, n_slots = slots.shape[:2]
        side = int(round(attn.shape[-1] ** 0.5))
        points = token_points(frame, side, self.image_size, self.focal_ratio)
        anchor, colour = attention_anchors(attn, points, colours)
        ray = F.normalize(anchor - frame.origin[:, None], dim=-1)
        # Explicit flattens: a generator expression here breaks the compiled graph.
        slots = slots.flatten(0, 1)
        anchor = anchor.flatten(0, 1)
        ray = ray.flatten(0, 1)
        colour = colour.flatten(0, 1)
        K = slots.shape[0]
        centre, extent = self.region(slots, anchor, ray)
        q = self.queries.expand(K, -1, -1)
        k = self.norm_kv(self.to_k(slots).view(K, N_TOKENS, self.d))
        v = self.norm_kv(self.to_v(slots).view(K, N_TOKENS, self.d))
        for layer in self.layers:
            q = layer(q, k, v)
        gaussians = self.head(q, centre, extent, colour)
        presence = (attn.mean(-1).flatten() / AREA_FULL).clamp(max=1.0)
        opacity = gaussians["opacity"]
        gaussians["opacity"] = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6) * presence[:, None]
        unflat = {name: value.unflatten(0, (b, n_slots)) for name, value in gaussians.items()}
        return unflat, extent.unflatten(0, (b, n_slots))
