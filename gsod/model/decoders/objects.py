"""Object decoder: slot -> a box standing on the floor -> G Gaussians in it.

1. anchor_init  each view-0 token's ray (pixel back-projected with the
                camera) meets the floor z = 0; their attention-weighted mean
                = anchor_init, the starting point on the floor.
2. box          RegionHead: anchor = anchor_init + learned shift along /
                across the ray (on the floor); half sizes (ex, ey, ez). The
                box spans x, y: anchor +- (ex, ey), z: [0, 2 ez], so its
                centre is at height ez.
3. shape      N_PATCHES surface patches per slot (AtlasNet): patch queries read the slot
              tokens -> a latent per patch; a point (u, v) of a patch lands at
              MLP(latent, u, v) in the box, from a sphere (a cube's faces pushed onto it).
              Nothing but (latent, u, v) places a point, so each patch is a smooth surface;
              (u, v) jittered within its cell in training.
4. Gaussians  one per (u, v), bound to the surface: a flat disc along its tangents (in-plane
              sizes from the neighbours' spacing), moved off its point by at most a cell
              along the surface and NORMAL_OFFSET x the box across it.
5. refine     what each Gaussian sees in view 0 (the lifter) updates its patch's latent
              (the shape) and its own offsets and appearance; it stays bound to the surface.

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
from gsod.utils.geometry import SceneFrame, camera_directions, matrix_to_quat

# Attention area (fraction of tokens) at which a slot is fully present.
AREA_FULL = 0.004
# Initial box half side (units of R; an initial value, not a limit).
BOX_INIT = 0.25
# Surface patches per slot (a cube's faces), the sphere they start on (box units).
N_PATCHES, TEMPLATE_RADIUS = 6, 0.7
# Disc sizes: in-plane this x the neighbours' spacing, across the surface this x the in-plane size.
COVER, THIN = 0.6, 0.1
# A Gaussian may leave its surface point by this x the box's mean half size (across the surface).
NORMAL_OFFSET = 0.05
# Transformer: heads, slot tokens, layers; RegionHead MLP width.
N_HEADS, N_TOKENS, N_LAYERS = 4, 8, 2
REGION_HIDDEN = 64


# anchor_init weights tokens by attn ** this (lean to the stronger core).
_ANCHOR_SHARPNESS = 2.0


def init_attention_anchor(
    attn: torch.Tensor,
    colours: torch.Tensor,
    frame: SceneFrame,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (B, K, N), token colours (B, N, 3), view 0's SceneFrames ->
    anchor_init on the floor, colour there (B, K, 3); scene frame."""
    # Pixel centre of each token (a side x side grid over the image).
    h, w = image_size
    side = int(round(attn.shape[-1] ** 0.5))
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    # Back-project with the camera: each token's view-0 ray (B, N, 3).
    rays = camera_directions(uu, vv, image_size, focal_ratio).reshape(-1, 3) @ frame.rot.mT
    # Where each ray meets the floor z = 0 (rays not going down stay at the camera), kept on the floor.
    t = (-frame.origin[..., 2:] / rays[..., 2].clamp(max=-1e-6)).clamp(min=0.0)
    xy = (frame.origin[..., None, :2] + t[..., None] * rays[..., :2]).clamp(-FLOOR_HALF, FLOOR_HALF)
    points = torch.cat([xy, torch.zeros_like(xy[..., :1])], -1)
    # Attention-weighted mean, weights attn ** _ANCHOR_SHARPNESS normalised.
    p = (_ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(dim=-1)
    return p @ points, p @ colours


def box_corners(anchor: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    """Anchor on the floor (..., 2), half sizes (..., 3) -> 8 corners (..., 8, 3);
    0-3 bottom, 4-7 top, both counter-clockwise."""
    sx = anchor.new_tensor([-1.0, 1.0, 1.0, -1.0] * 2)
    sy = anchor.new_tensor([-1.0, -1.0, 1.0, 1.0] * 2)
    top = anchor.new_tensor([0.0] * 4 + [1.0] * 4)
    x = anchor[..., None, 0] + sx * extent[..., None, 0]
    y = anchor[..., None, 1] + sy * extent[..., None, 1]
    z = 2.0 * top * extent[..., None, 2]
    return torch.stack([x, y, z], -1)


class RegionHead(nn.Module):
    """Slot -> the box: anchor = anchor_init + shift along / across view 0's
    ray, half sizes; in units of BOX_INIT, zero at first (a box of half side
    BOX_INIT at anchor_init)."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.mlp = mlp(z_dim, REGION_HIDDEN, 5)  # shift along, shift across, log (ex, ey, ez)
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(
        self, slots: torch.Tensor, anchor_init: torch.Tensor, ray: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (K, C), anchor_init, ray (K, 3) -> anchor on the floor (K, 2), half sizes (K, 3)."""
        raw = self.mlp(slots)
        along = F.normalize(ray[:, :2], dim=-1)  # the ray, seen from above
        across = torch.stack([-along[:, 1], along[:, 0]], -1)
        shift = BOX_INIT * raw[:, :2]
        anchor = anchor_init[:, :2] + shift[:, :1] * along + shift[:, 1:] * across
        extent = BOX_INIT * torch.exp(raw[:, 2:])
        return anchor, extent


def patch_uv(side: int, device, dtype, jitter: bool) -> torch.Tensor:
    """(u, v) of every patch point (N_PATCHES, side, side, 2) in [-1, 1]: cell centres, or a random
    point of each cell (jitter)."""
    g = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side * 2 - 1
    v, u = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack([u, v], -1).expand(N_PATCHES, -1, -1, -1)
    if jitter:
        uv = uv + (torch.rand_like(uv) - 0.5) * (2 / side)
    return uv


def sphere_template(uv: torch.Tensor) -> torch.Tensor:
    """(u, v) (N_PATCHES, ..., 2) -> the cube's face points pushed onto the unit sphere (N_PATCHES, ..., 3)."""
    faces = []
    for face, (axis, sign) in enumerate((a, s) for a in range(3) for s in (-1.0, 1.0)):
        u, v = uv[face].unbind(-1)
        others = [i for i in range(3) if i != axis]
        coords = [None, None, None]
        coords[axis], coords[others[0]], coords[others[1]] = torch.full_like(u, sign), sign * u, v  # outward
        faces.append(torch.stack(coords, -1))
    return F.normalize(torch.stack(faces), dim=-1)


def _grid_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Differences between neighbours along a grid dimension (central inside, one-sided at the edges)."""
    n = x.shape[dim]
    first = x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1)
    inner = (x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) / 2
    last = x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2, 1)
    return torch.cat([first, inner, last], dim)


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    """Grids (..., S, S, C) -> each inner point minus its 4 neighbours' mean (..., S-2, S-2, C)."""
    mean = (x[..., :-2, 1:-1, :] + x[..., 2:, 1:-1, :] + x[..., 1:-1, :-2, :] + x[..., 1:-1, 2:, :]) / 4
    return x[..., 1:-1, 1:-1, :] - mean


class SurfaceHead(nn.Module):
    """Patch latent + (u, v) -> the surface point (box units, before tanh, as a residual on the
    sphere) and a point feature."""

    def __init__(self, d_model: int):
        super().__init__()
        d = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d + 2, d), nn.ReLU(True), nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d), nn.ReLU(True)
        )
        self.to_point = nn.Linear(d, 3)
        nn.init.zeros_(self.to_point.weight)  # the sphere at first
        nn.init.zeros_(self.to_point.bias)

    def forward(self, latent: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """latent (K, N_PATCHES, d), uv (N_PATCHES, S, S, 2) -> point residual (K, N_PATCHES, S, S, 3),
        feature (K, N_PATCHES, S, S, d)."""
        k = latent.shape[0]
        x = torch.cat([latent[:, :, None, None].expand(-1, -1, *uv.shape[1:3], -1), uv.expand(k, -1, -1, -1, -1)], -1)
        feat = self.mlp(x)
        return self.to_point(feat), feat


class GaussianHead(nn.Module):
    """Point feature -> a disc bound to the surface: offsets (normal, 2 tangents), in-plane log
    size residual (2), opacity, rgb residual (3), sh1 (9)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = mlp(d_model, d_model, 18)
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight[:3].zero_()  # on the surface at first
            out.weight[9:].zero_()  # view-independent at first
            out.bias.zero_()

    def forward(
        self,
        point: torch.Tensor,
        feat: torch.Tensor,
        anchor: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
        uv: torch.Tensor,
        view_dependent: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Point residual (K, N_PATCHES, S, S, 3), feature (K, N_PATCHES, S, S, d), box anchor (K, 2),
        half sizes / slot colour (K, 3), uv -> Gaussians (K, G, ...) and the surface's roughness
        (K,): the mean squared Laplacian of its deformation from the sphere, per cell."""
        side = point.shape[2]
        t = torch.tanh(torch.atanh(TEMPLATE_RADIUS * sphere_template(uv)) + point)  # the surface in the box, [-1, 1]^3
        roughness = (side * _laplacian(point)).pow(2).sum(-1).flatten(1).mean(-1)
        xy = anchor[:, None, None, None] + extent[:, None, None, None, :2] * t[..., :2]
        # Height: the box's [0, 2 ez], kept above the floor's Gaussians.
        lo, hi = BACKGROUND_THICKNESS, (2.0 * extent[:, 2])[:, None, None, None, None]
        surface = torch.cat([xy, lo + (hi - lo) * 0.5 * (t[..., 2:] + 1.0)], -1)  # (K, N_PATCHES, S, S, 3)

        # The surface's frame at each point (no gradient: quaternions of near-axis frames blow up).
        du, dv = _grid_diff(surface, 3), _grid_diff(surface, 2)
        with torch.no_grad():
            normal = F.normalize(torch.linalg.cross(du, dv), dim=-1)
            t1 = F.normalize(du, dim=-1)
            t2 = torch.linalg.cross(normal, t1)
            quat = matrix_to_quat(torch.stack([t1, t2, normal], -1))
        spacing = torch.stack([du.norm(dim=-1), dv.norm(dim=-1)], -1)  # (K, N_PATCHES, S, S, 2)

        raw = self.mlp(feat)
        off = torch.tanh(raw[..., 0:3])
        across = NORMAL_OFFSET * extent.mean(-1)[:, None, None, None, None]
        mu = surface + off[..., :1] * across * normal + off[..., 1:2] * spacing[..., :1] * t1
        mu = mu + off[..., 2:3] * spacing[..., 1:] * t2
        in_plane = COVER * spacing * torch.exp(raw[..., 3:5])
        scale = torch.cat([in_plane, THIN * in_plane.mean(-1, keepdim=True)], -1).clamp_min(1e-5)
        sh1 = raw[..., 9:18].unflatten(-1, (3, 3))
        gaussians = {
            "mu": mu,
            "scale": scale,
            "quat": F.normalize(quat, dim=-1),
            "opacity": torch.sigmoid(raw[..., 5]),
            "rgb": shift_colour(colour[:, None, None, None], raw[..., 6:9]),  # slot colour + a residual
            "sh1": sh1 if view_dependent else torch.zeros_like(sh1),
        }
        return {name: value.flatten(1, 3) for name, value in gaussians.items()}, roughness


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
    """Object slots -> Gaussians on each slot's surface patches, scene frame."""

    def __init__(
        self,
        z_dim: int,
        gaussians_per_slot: int,
        hidden_dim: int,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float],
    ):
        super().__init__()
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        d = int(hidden_dim)
        self.d = d
        self.side = int((gaussians_per_slot // N_PATCHES) ** 0.5)  # G = N_PATCHES * side^2
        self.view_dependent = True  # sh1 on; the trainer turns it off early on
        self.patch_queries = nn.Parameter(torch.randn(N_PATCHES, d))
        self.to_k = nn.Linear(z_dim, N_TOKENS * d)
        self.to_v = nn.Linear(z_dim, N_TOKENS * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, N_HEADS) for _ in range(N_LAYERS))
        self.region = RegionHead(z_dim)
        self.surface = SurfaceHead(d)
        self.gaussian = GaussianHead(d)
        self.seen_to_latent = nn.Linear(d, d)  # refine: what a patch's Gaussians see -> its latent
        nn.init.zeros_(self.seen_to_latent.weight)
        nn.init.zeros_(self.seen_to_latent.bias)

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        """Object slots (B, K, C), their attention (B, K, N), token colours
        (B, N, 3), view 0's SceneFrames -> sketch Gaussians (B, K, G, ...), the box
        {"anchor_init" (B, K, 3), "anchor" (B, K, 2), "extent" (B, K, 3), "roughness" (B, K)}
        and the state for refine(); scene frame."""
        b, n_slots = slots.shape[:2]
        anchor_init, colour = init_attention_anchor(attn, colours, frame, self.image_size, self.focal_ratio)
        ray = F.normalize(anchor_init - frame.origin[:, None], dim=-1)  # camera -> anchor_init
        box = {"anchor_init": anchor_init}
        slots, anchor_init, ray, colour = (x.flatten(0, 1) for x in (slots, anchor_init, ray, colour))
        K = slots.shape[0]
        anchor, extent = self.region(slots, anchor_init, ray)
        k, v = (self.norm_kv(f(slots).view(K, N_TOKENS, self.d)) for f in (self.to_k, self.to_v))  # slot tokens
        latent = self.patch_queries.expand(K, -1, -1)
        for layer in self.layers:
            latent = layer(latent, k, v)  # (B*K, N_PATCHES, d)
        state = {
            "anchor": anchor,
            "extent": extent,
            "colour": colour,
            "presence": (attn.mean(-1).flatten() / AREA_FULL).clamp(max=1.0),
            "shape": (b, n_slots),
            "uv": patch_uv(self.side, slots.device, slots.dtype, self.training),
            "latent": latent,
        }
        box["anchor"] = anchor.unflatten(0, (b, n_slots))
        box["extent"] = extent.unflatten(0, (b, n_slots))
        gaussians, feat, box["roughness"] = self.decode(latent, None, state)
        state["feat"], state["mu"] = feat, gaussians["mu"]  # where the lifter samples
        return gaussians, box, state

    def refine(self, seen: torch.Tensor, state: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """What each sketch Gaussian sees in view 0 (B*K, G, d) -> refined Gaussians (B, K, G, ...)
        and the surface's roughness (B, K): the patches' latents move by their Gaussians' mean;
        each Gaussian adds its own to its feature."""
        k = seen.shape[0]
        latent = state["latent"] + self.seen_to_latent(seen.view(k, N_PATCHES, -1, self.d).mean(2))
        gaussians, _, roughness = self.decode(latent, seen, state)
        return gaussians, roughness

    def decode(
        self, latent: torch.Tensor, seen: torch.Tensor | None, state: dict
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Patch latents (B*K, N_PATCHES, d), what each Gaussian sees (B*K, G, d) or None ->
        Gaussians (B, K, G, ...), point features (B*K, G, d), roughness (B, K)."""
        point, feat = self.surface(latent, state["uv"])
        feat = feat.flatten(1, 3)  # (B*K, G, d)
        if seen is not None:
            feat = feat + seen
        gaussians, roughness = self.gaussian(
            point,
            feat.view(*point.shape[:4], -1),
            state["anchor"],
            state["extent"],
            state["colour"],
            state["uv"],
            self.view_dependent,
        )
        # Each slot's brightest Gaussian at its presence.
        opacity = gaussians["opacity"]
        gaussians["opacity"] = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6) * state["presence"][:, None]
        gaussians = {name: value.unflatten(0, state["shape"]) for name, value in gaussians.items()}
        return gaussians, feat, roughness.unflatten(0, state["shape"])
