"""Object decoder: slot -> box on its attention ray -> G Gaussians in it.
Box sizes, shifts and Gaussian scales unbounded; BOX_INIT only sets where they start.
Opacity = per-slot share of its brightest Gaussian x presence (attention area /
AREA_FULL, at most 1): the attention decides whether a slot shows, never 0.
"""

from __future__ import annotations

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


# Attention rays weight tokens by attn ** this (lean to the stronger core).
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


def attention_rays(
    attn: torch.Tensor, rays: torch.Tensor, colours: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (..., K, N), token rays / colours (..., N, 3) -> unit ray, colour (..., K, 3)."""
    p = (_ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(dim=-1)
    return F.normalize(p @ rays, dim=-1), p @ colours


class RegionHead(nn.Module):
    """Slot + attention ray -> box on the floor around a point on the ray."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.mlp = mlp(z_dim + 2, REGION_HIDDEN, 5)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(
        self, slots: torch.Tensor, origin: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), camera origin, unit ray (K, 3) -> box centre, half sizes (K, 3)."""
        # + camera height, ray slope
        raw = self.mlp(torch.cat([slots, origin[:, 2:], ray[:, 2:]], -1))
        extent = BOX_INIT * torch.exp(raw[:, 2:])
        # Start where the ray reaches the box's half height, then a learned depth step.
        height = extent[:, 2:].detach()
        depth = ((height - origin[:, 2:]) / ray[:, 2:].clamp(max=-1e-3)).clamp_min(0.0)
        depth = depth + BOX_INIT * raw[:, :1]
        ray_xy = F.normalize(ray[:, :2], dim=-1)
        across = torch.stack([-ray_xy[:, 1], ray_xy[:, 0]], -1)
        xy = origin[:, :2] + depth * ray[:, :2] + BOX_INIT * raw[:, 1:2] * across  # + occlusion shift
        xy = xy.clamp(-FLOOR_HALF, FLOOR_HALF)
        # Box on the floor.
        return torch.cat([xy, extent[:, 2:]], -1), extent


class ObjectHead(nn.Module):
    """Gaussian feature -> one Gaussian."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = mlp(d_model, d_model, 23)
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight[3:].mul_(0.1)
            out.weight[14:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[6] = 1.0  # identity quaternion (wxyz)

    def forward(
        self,
        feat: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), box centre / half sizes / colour (K, 3) -> Gaussians (K, G, ...)."""
        raw = self.mlp(feat)
        t = torch.tanh(raw[..., 0:3])
        xy = centre[:, None, :2] + extent[:, None, :2] * t[..., :2]
        # Height: over [centre - extent, centre + extent], above the floor's Gaussians.
        lo = (centre[:, 2] - extent[:, 2]).clamp_min(BACKGROUND_THICKNESS)[:, None, None]
        hi = (centre[:, 2] + extent[:, 2])[:, None, None]
        mu = torch.cat([xy, lo + (hi - lo) * 0.5 * (t[..., 2:] + 1.0)], -1)

        return {
            "mu": mu,
            "scale": SCALE_FRACTION * BOX_INIT * torch.exp(raw[..., 3:6]),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": torch.sigmoid(raw[..., 10]),
            "rgb": shift_colour(colour[:, None], raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class DecoderLayer(nn.Module):
    """Self-attention among the Gaussians, cross-attention to the slot tokens,
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
        """q (K, G, d), slot tokens k, v (K, M, d) -> (K, G, d)."""
        x = self.norm_self(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        q = q + self.cross_attn(self.norm_cross(q), k, v, need_weights=False)[0]
        return q + self.mlp(self.norm_mlp(q))


class ObjectDecoder(nn.Module):
    """Object slots -> Gaussians, scene frame."""

    def __init__(
        self, z_dim: int, gaussians_per_slot: int, hidden_dim: int,
        image_size: tuple[int, int], focal_ratio: tuple[float, float],
    ):
        super().__init__()
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        d = int(hidden_dim)
        self.d = d
        self.queries = nn.Parameter(torch.randn(int(gaussians_per_slot), d) * QUERY_STD)
        self.to_k = nn.Linear(z_dim, N_TOKENS * d)
        self.to_v = nn.Linear(z_dim, N_TOKENS * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, N_HEADS) for _ in range(N_LAYERS))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d)

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Object slots (B, K, C), their attention (B, K, N), token colours
        (B, N, 3), view 0's SceneFrames -> Gaussians (B, K, G, ...), half sizes
        (B, K, 3); scene frame."""
        b, n_slots = slots.shape[:2]
        side = int(round(attn.shape[-1] ** 0.5))
        rays = _token_rays(frame, side, self.image_size, self.focal_ratio)
        ray, colour = attention_rays(attn, rays, colours)
        origin = frame.origin[:, None].expand(-1, n_slots, -1)
        slots, origin, ray, colour = (x.flatten(0, 1) for x in (slots, origin, ray, colour))
        K = slots.shape[0]
        centre, extent = self.region(slots, origin, ray)
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
