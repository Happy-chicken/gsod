"""Object decoder: object slot -> region + G Gaussians in it.

The region sits around the attention anchor (shifted along view 0's ray),
sized from the slot latent. Two kinds of queries attend to tokens from the
slot latent and share one head:
    free    G learned queries -> Gaussians anywhere in the region (the whole
            object, occluded parts included); opacities max-normalised
    pixel   one per view-0 token whose ray crosses the region above the
            floor -> a Gaussian on that ray inside the region (visible
            detail); opacity free, 0 where the ray misses
Scene frame, lengths in units of R.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Object radius range (units of R), shared by every dataset.
OBJECT_RADIUS = (0.05, 0.2)
# Half side of the area objects can be in.
OBJECT_REGION = 1.0 + 1.5 * OBJECT_RADIUS[1]
# Largest region half width / height (units of r_max).
OBJECT_WIDTH = 1.5
OBJECT_HEIGHT = 1.2
# Initial region size (share of its range).
EXTENT_INIT = 0.9
# Pixel Gaussians stay this high above the floor (units of R), so a slot
# cannot copy floor pixels consistently across views.
OBJECT_FLOOR = 0.02
# Pixel Gaussians start nearly transparent.
PIXEL_OPACITY_BIAS = -4.0
# Gaussian scale range and initial value (units of r_max).
SCALE_RANGE = (1.0 / 40.0, 1.0)
SCALE_INIT = 1.0 / 8.0


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _log_uniform(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.sigmoid(raw))


def ray_box(
    origin: torch.Tensor, dirs: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rays origin (3,) + t * dirs (N, 3) through boxes [lo, hi] (K, 3) ->
    entry and exit t (K, N); a miss has t_out <= t_in."""
    safe = torch.where(dirs.abs() < 1e-9, torch.full_like(dirs, 1e-9), dirs)
    t0 = (lo[:, None] - origin) / safe
    t1 = (hi[:, None] - origin) / safe
    return torch.minimum(t0, t1).amax(-1).clamp_min(0.0), torch.maximum(t0, t1).amin(-1)


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
    """Raw parameters: mu 3, scale 3, quat 4, opacity 1, rgb 3, sh1 9."""

    def __init__(self, z_dim: int, d_model: int):
        super().__init__()
        r_min, r_max = OBJECT_RADIUS
        self.scale_range = (SCALE_RANGE[0] * r_max, SCALE_RANGE[1] * r_max)
        self.project = _projection(d_model, 23)
        # Region shift along (ray, across, up) and half sizes.
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
            bias[6] = 1.0  # identity quaternion (wxyz)
            self.project[-1].weight[14:].zero_()  # view-independent at first
            self.shift.weight.zero_()
            self.shift.bias.zero_()
            self.size.weight.zero_()
            self.size.bias.fill_(_logit(EXTENT_INIT))

    def region(
        self, slots: torch.Tensor, centre: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor, ray (K, 3) -> region centre, half sizes (K, 3)."""
        up = torch.zeros_like(ray)
        up[:, 2] = 1.0
        across = F.normalize(torch.cross(up, ray, dim=-1), dim=-1)
        basis = torch.stack([ray, across, torch.cross(ray, across, dim=-1)], 1)  # (K, 3, 3)
        t = self.shift_bound * torch.tanh(self.shift(slots))
        centre = centre + torch.einsum("ki,kij->kj", t, basis)
        extent = self.size_lo + (self.size_hi - self.size_lo) * torch.sigmoid(self.size(slots))
        return centre, extent

    def _shared(self, raw: torch.Tensor, base: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scale, rotation and colour (residual on base) of either kind."""
        return {
            "scale": _log_uniform(raw[..., 3:6], *self.scale_range),
            "quat": F.normalize(raw[..., 6:10], dim=-1),
            "rgb": torch.sigmoid(torch.logit(base.clamp(0.02, 0.98)) + raw[..., 11:14]),
            "sh1": raw[..., 14:23].unflatten(-1, (3, 3)),
        }

    def forward(
        self, feat: torch.Tensor, centre: torch.Tensor, extent: torch.Tensor, colour: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Free queries: feat (K, G, d) -> Gaussians (K, G, ...) in the region."""
        raw = self.project(feat)
        opacity = torch.sigmoid(raw[..., 10])
        return {
            "mu": centre[:, None] + extent[:, None] * torch.tanh(raw[..., 0:3]),
            "opacity": opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6),
            **self._shared(raw, colour[:, None]),
        }

    def pixels(
        self,
        feat: torch.Tensor,
        origin: torch.Tensor,
        dirs: torch.Tensor,
        t_in: torch.Tensor,
        t_out: torch.Tensor,
        hit: torch.Tensor,
        colours: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pixel queries: feat (K, N, d), rays, their span in the region
        (K, N) -> Gaussians (K, N, ...) on the rays."""
        raw = self.project(feat)
        t = t_in + (t_out - t_in) * torch.sigmoid(raw[..., 0])
        return {
            "mu": origin + t[..., None] * dirs,
            "opacity": torch.sigmoid(raw[..., 10] + PIXEL_OPACITY_BIAS) * hit,
            **self._shared(raw, colours[None]),
        }


class ObjectDecoder(nn.Module):
    """Object slots -> regions and Gaussians, scene frame."""

    def __init__(
        self,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        *,
        n_heads: int = 4,
        n_tokens: int = 8,
        query_std: float = 2.0,
    ):
        """hidden_dim: the attention width."""
        super().__init__()
        d_model = int(hidden_dim)
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_gaussians = int(gaussians_per_slot)
        self.M = int(n_tokens)
        self.d = int(d_model)

        # Query spread sets the initial Gaussian spread (small: collapse).
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
        # Pixel queries: token feature, colour, ray entry and exit in the region.
        self.pixel_in = nn.Linear(z_dim + 9, d_model)

    @property
    def extent_range(self) -> tuple[float, float]:
        """xy half width range (units of R)."""
        return float(self.head.size_lo[0]), float(self.head.size_hi[0])

    def _pixel_queries(self, pixels: dict, centre: torch.Tensor, extent: torch.Tensor):
        """Pixel queries (K, N, d) and their rays' span in each region."""
        origin, dirs = pixels["origin"], pixels["dirs"]
        lo = centre - extent
        lo = torch.cat([lo[:, :2], lo[:, 2:].clamp_min(OBJECT_FLOOR)], -1)
        t_in, t_out = ray_box(origin, dirs, lo, centre + extent)
        hit = t_out > t_in
        t_in = torch.where(hit, t_in, torch.ones_like(t_in))
        t_out = torch.where(hit, t_out, torch.ones_like(t_out))

        def rel(t):
            return (origin + t[..., None] * dirs - centre[:, None]) / extent[:, None]

        K = centre.shape[0]
        shared = torch.cat([pixels["features"], pixels["colours"]], -1)
        q = self.pixel_in(torch.cat([shared[None].expand(K, -1, -1), rel(t_in), rel(t_out)], -1))
        return q, (origin, dirs, t_in, t_out, hit.to(q.dtype), pixels["colours"])

    def _attend(self, q, k, v):
        q = q + self.cross(self.norm_q(q), k, v, need_weights=False)[0]
        return q + self.ffn(self.norm_ff(q))

    def forward(
        self,
        slots: torch.Tensor,
        anchor: torch.Tensor,
        ray: torch.Tensor,
        colour: torch.Tensor,
        pixels: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """slots (K, C); anchor, ray, colour (K, 3); pixels (None: free
        queries only): features (N, C), colours (N, 3), view-0 rays origin
        (3,) and dirs (N, 3) -> Gaussians (K, G [+ N], ...), region centre and
        half sizes (K, 3), pixel opacities (K, N) or None."""
        K = slots.shape[0]
        centre, extent = self.head.region(slots, anchor, ray)
        k = self.norm_kv(self.k_proj(slots).view(K, self.M, self.d))
        v = self.norm_kv(self.v_proj(slots).view(K, self.M, self.d))
        q = self.queries.unsqueeze(0).expand(K, -1, -1)
        if pixels is None:
            return self.head(self._attend(q, k, v), centre, extent, colour), centre, extent, None
        pixel_q, rays = self._pixel_queries(pixels, centre, extent)
        q = self._attend(torch.cat([q, pixel_q], 1), k, v)
        free = self.head(q[:, : self.n_gaussians], centre, extent, colour)
        pix = self.head.pixels(q[:, self.n_gaussians :], *rays)
        gaussians = {name: torch.cat([free[name], pix[name]], 1) for name in free}
        return gaussians, centre, extent, pix["opacity"]
