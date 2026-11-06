"""Object decoder: each object slot -> its region and Gaussians in it.

ObjectHead sets the region from the slot latent around the attention's
anchor: shifted far along view 0's ray (depth), a little across it (half
occlusion, height), so a slot draws where its attention points; half sizes
from the slot latent (attention holds only an object's core, not its size),
largest at first. G learned queries (shared by all slots, one per Gaussian)
attend to tokens projected from the slot latent, and the head turns each
(slot, query) feature into a Gaussian in the region: mu, scale, quat,
opacity, rgb (from the attention's mean colour) and sh1 (degree-1 SH,
starting at 0). Opacities are normalised so the most opaque is 1: how far a
slot is drawn is the draw factor alone (gsod.draw_factor). Everything is in
the scene frame (utils/geometry.py): lengths in units of R.

pixel_features: the queries also attend to view 0's tokens, biased by
log(slot attention), so a slot reads only its own pixels.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Object radii r_min, r_max in units of R: one guess for every dataset
# (largest objects: 0.25 R on Room-Chair, 0.14 on CLEVR-567, 0.18 on
# Primordia); each region's size is predicted within the range.
OBJECT_RADIUS = (0.05, 0.2)
# Where objects can be: within R of c, plus a turned cube's reach.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest region half width and half height in units of r_max (a cube
# turned about z reaches sqrt(2) r).
OBJECT_WIDTH = 1.5
OBJECT_HEIGHT = 1.2
# Regions start at this share of their size range.
EXTENT_INIT = 0.9
# Gaussian scales in units of r_max: range and initial value.
SCALE_RANGE = (1.0 / 40.0, 1.0)
SCALE_INIT = 1.0 / 8.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def _projection(d_model: int, n_params: int) -> nn.Sequential:
    """Feature -> raw parameters; all but the position rows start small."""
    mlp = nn.Sequential(
        nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, n_params)
    )
    with torch.no_grad():
        mlp[-1].weight[3:].mul_(0.1)
        mlp[-1].bias.zero_()
    return mlp


class ObjectHead(nn.Module):
    """Raw parameters: position 3, scale 3, quat 4, opacity 1, colour residual
    3, degree-1 SH 9."""

    def __init__(self, z_dim: int, d_model: int):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.scale_range = (SCALE_RANGE[0] * r_max, SCALE_RANGE[1] * r_max)
        self.project = _projection(d_model, 23)
        # Region from the slot latent: shift along (ray, across, up) and half
        # sizes per axis within [lo, hi].
        self.shift = nn.Linear(z_dim, 3)
        self.size = nn.Linear(z_dim, 3)
        self.register_buffer("shift_bound", torch.tensor(
            [0.5 * OBJECT_REGION, 0.5 * r_max, 0.5 * r_max]), persistent=False)
        self.register_buffer("size_lo", torch.tensor([r_min, r_min, r_min]), persistent=False)
        self.register_buffer("size_hi", torch.tensor(
            [OBJECT_WIDTH * r_max, OBJECT_WIDTH * r_max, OBJECT_HEIGHT * r_max]), persistent=False)
        lo, hi = (math.log(s) for s in self.scale_range)
        with torch.no_grad():
            bias = self.project[-1].bias
            bias[3:6] = _logit((math.log(SCALE_INIT * r_max) - lo) / (hi - lo))
            bias[6] = 1.0  # identity quaternion (w first, gsplat convention)
            self.project[-1].weight[14:].zero_()  # view-independent at first
            self.shift.weight.zero_()
            self.shift.bias.zero_()
            self.size.weight.zero_()
            self.size.bias.fill_(_logit(EXTENT_INIT))

    def region(
        self, slots: torch.Tensor, centre: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), attention centre and unit view-0 ray (K, 3) -> the
        region's centre and half sizes (K, 3)."""
        up = torch.zeros_like(ray)
        up[:, 2] = 1.0
        across = F.normalize(torch.cross(up, ray, dim=-1), dim=-1)
        basis = torch.stack([ray, across, torch.cross(ray, across, dim=-1)], 1)  # (K, 3, 3)
        t = self.shift_bound * torch.tanh(self.shift(slots))
        centre = centre + torch.einsum("ki,kij->kj", t, basis)
        extent = self.size_lo + (self.size_hi - self.size_lo) * torch.sigmoid(self.size(slots))
        return centre, extent

    def forward(
        self,
        feat: torch.Tensor,
        slots: torch.Tensor,
        centre: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d), slots (K, C), region centre / half sizes / colour
        (K, 3) -> Gaussians (K, G, ...)."""
        raw = self.project(feat)
        region = torch.logit(colour.clamp(0.02, 0.98))
        opacity = torch.sigmoid(raw[..., 10])
        opacity = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return {
            "mu": centre[:, None] + extent[:, None] * torch.tanh(raw[..., 0:3]),
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "opacity": opacity,
            "rgb": torch.sigmoid(region[:, None] + raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }


class MaskedCrossAttention(nn.Module):
    """Cross-attention with a per-slot logit bias; output starts at zero."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.h = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, x: torch.Tensor, tokens: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """x (K, G, d), tokens (K, N, d), bias (K, N) -> (K, G, d)."""
        K, G, d = x.shape
        N, h = tokens.shape[1], self.h
        q = self.q(x).view(K, G, h, d // h).transpose(1, 2)
        k, v = self.kv(tokens).view(K, N, 2, h, d // h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias[:, None, None, :].to(q.dtype))
        return self.out(o.transpose(1, 2).reshape(K, G, d))


class ObjectDecoder(nn.Module):
    """Object slots -> their regions and Gaussians, scene frame."""

    def __init__(
        self,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        query_std: float = 2.0,
        pixel_features: bool = False,
    ):
        """hidden_dim: the attention width."""
        super().__init__()
        d_model = int(hidden_dim)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_gaussians = int(gaussians_per_slot)
        self.M = int(n_tokens)
        self.d = int(d_model)

        # Learned Gaussian queries (G, d): their spread sets how far apart a
        # slot's Gaussians start (std 2.0: ~U(-0.7, 0.7) of the region; a
        # small std collapses every slot to a point).
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
        self.head = ObjectHead(z_dim, d_model)
        self.pixel_features = bool(pixel_features)
        if self.pixel_features:
            self.token_in = nn.Linear(z_dim + 6, d_model)
            self.norm_tok = nn.LayerNorm(d_model)
            self.norm_pix = nn.LayerNorm(d_model)
            self.pixel_cross = MaskedCrossAttention(d_model, n_heads)

    @property
    def extent_range(self) -> tuple[float, float]:
        """The regions' xy half width range (units of R)."""
        return float(self.head.size_lo[0]), float(self.head.size_hi[0])

    def _pixel_tokens(self, pixels: dict, centre: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
        """(K, N, d): token feature, colour, position in the slot's region."""
        K = centre.shape[0]
        rel = ((pixels["points"][None] - centre[:, None]) / extent[:, None]).clamp(-4.0, 4.0)
        shared = torch.cat([pixels["features"], pixels["colours"]], -1)
        tokens = self.token_in(torch.cat([shared[None].expand(K, -1, -1), rel], -1))
        return self.norm_tok(tokens)

    def forward(
        self,
        slots: torch.Tensor,
        anchor: torch.Tensor,
        ray: torch.Tensor,
        colour: torch.Tensor,
        pixels: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """slots (K, C); anchor, ray, colour (K, 3); pixels: features (N, C),
        colours, points (N, 3), attn (K, N) -> Gaussians (K, G, ...), region
        centre and half sizes (K, 3)."""
        K = slots.shape[0]
        centre, extent = self.head.region(slots, anchor, ray)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        q = self.queries.unsqueeze(0).expand(K, -1, -1)
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        if self.pixel_features:
            tokens = self._pixel_tokens(pixels, centre, extent)
            bias = pixels["attn"].clamp_min(1e-6).log()
            q = q + self.pixel_cross(self.norm_pix(q), tokens, bias)
        q = q + self.ffn(self.norm_ff(q))
        return self.head(q, slots, centre, extent, colour), centre, extent
