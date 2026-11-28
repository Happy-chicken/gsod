"""Object decoder: slot -> a box standing on the floor -> G Gaussians in it.

1. anchor_init  each view-0 token's ray (pixel back-projected with the
                camera) meets the floor z = 0; their attention-weighted mean
                = anchor_init, the starting point on the floor.
2. box          RegionHead: anchor = anchor_init + learned shift along /
                across the ray (on the floor); half sizes (ex, ey, ez). The
                box spans x, y: anchor +- (ex, ey), z: [0, 2 ez], so its
                centre is at height ez.
3. Gaussians  on N_PATCHES surface patches per slot (AtlasNet-like): each query is a point
              (u, v) of a patch; the decoder folds the patches into the object's surface in
              the box, starting from a sphere (a cube's faces pushed onto it). Gaussians are
              flat discs on the surface: rotation from the surface's tangents, in-plane sizes
              from the neighbours' spacing; so they cover a surface, never fill a volume.

Box sizes, shifts and Gaussian scales unbounded; BOX_INIT only sets where they start.
Opacity = per-slot share of its brightest Gaussian x presence (attention area /
AREA_FULL, at most 1): the attention decides whether a slot shows, never 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.background import BACKGROUND_THICKNESS, FLOOR_HALF
from gsod.model.layers import mlp, positional_encoding, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions, matrix_to_quat

# Attention area (fraction of tokens) at which a slot is fully present.
AREA_FULL = 0.004
# Initial box half side (units of R; an initial value, not a limit).
BOX_INIT = 0.25
# Surface patches per slot (a cube's faces), the sphere they start on (box units), (u, v) PE bands.
N_PATCHES, TEMPLATE_RADIUS, UV_PE_BANDS = 6, 0.7, 4
# Disc sizes: in-plane this x the neighbours' spacing, across the surface this x the in-plane size.
COVER, THIN = 0.6, 0.1
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


def patch_grid(side: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (u, v) of every patch point (N_PATCHES, side, side, 2) in [-1, 1] and where it starts:
    the cube's face point pushed onto the unit sphere (N_PATCHES, side, side, 3)."""
    g = (torch.arange(side, device=device, dtype=dtype) + 0.5) / side * 2 - 1
    v, u = torch.meshgrid(g, g, indexing="ij")
    uv = torch.stack([u, v], -1)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            point = torch.empty(side, side, 3, device=device, dtype=dtype)
            others = [i for i in range(3) if i != axis]
            point[..., axis] = sign
            point[..., others[0]], point[..., others[1]] = sign * u, v  # outward-facing (u, v)
            faces.append(point)
    return uv.expand(N_PATCHES, -1, -1, -1), F.normalize(torch.stack(faces), dim=-1)


def _grid_diff(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Differences between neighbours along a grid dimension (central inside, one-sided at the edges)."""
    n = x.shape[dim]
    first = x.narrow(dim, 1, 1) - x.narrow(dim, 0, 1)
    inner = (x.narrow(dim, 2, n - 2) - x.narrow(dim, 0, n - 2)) / 2
    last = x.narrow(dim, n - 1, 1) - x.narrow(dim, n - 2, 1)
    return torch.cat([first, inner, last], dim)


class ObjectHead(nn.Module):
    """Gaussian feature on a patch -> one disc of the surface."""

    def __init__(self, d_model: int):
        super().__init__()
        # Position residual (3), in-plane log size residual (2), opacity, rgb residual (3), sh1 (9).
        self.mlp = mlp(d_model, d_model, 18)
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight[9:].zero_()  # view-independent at first
            out.bias.zero_()

    def forward(
        self,
        feat: torch.Tensor,
        anchor: torch.Tensor,
        extent: torch.Tensor,
        colour: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """feat (K, G, d) on the patches' grids, box anchor (K, 2), half sizes / slot colour (K, 3)
        -> Gaussians (K, G, ...)."""
        raw = self.mlp(feat)
        k, g = raw.shape[:2]
        side = int(round((g // N_PATCHES) ** 0.5))
        _, start = patch_grid(side, raw.device, raw.dtype)
        start = torch.atanh(TEMPLATE_RADIUS * start).reshape(-1, 3)  # the sphere, before tanh
        t = torch.tanh(start + raw[..., 0:3])  # position in the box, [-1, 1]^3
        xy = anchor[:, None] + extent[:, None, :2] * t[..., :2]
        # Height: the box's [0, 2 ez], kept above the floor's Gaussians.
        lo = BACKGROUND_THICKNESS
        hi = (2.0 * extent[:, 2])[:, None, None]
        mu = torch.cat([xy, lo + (hi - lo) * 0.5 * (t[..., 2:] + 1.0)], -1)

        # Discs on the surface: tangents from the neighbours on each patch's grid.
        grid = mu.view(k, N_PATCHES, side, side, 3)
        du, dv = _grid_diff(grid, 3), _grid_diff(grid, 2)
        normal = F.normalize(torch.linalg.cross(du, dv), dim=-1)
        t1 = F.normalize(du, dim=-1)
        rot = torch.stack([t1, torch.linalg.cross(normal, t1), normal], -1)  # columns: tangents, normal
        in_plane = COVER * torch.stack([du.norm(dim=-1), dv.norm(dim=-1)], -1).view(k, g, 2)
        in_plane = in_plane * torch.exp(raw[..., 3:5])
        scale = torch.cat([in_plane, THIN * in_plane.mean(-1, keepdim=True)], -1).clamp_min(1e-5)

        return {
            "mu": mu,
            "scale": scale,
            "quat": F.normalize(matrix_to_quat(rot).view(k, g, 4), dim=-1),
            "opacity": torch.sigmoid(raw[..., 5]),
            "rgb": shift_colour(colour[:, None], raw[..., 6:9]),  # slot colour + a residual
            "sh1": raw[..., 9:18].unflatten(-1, (3, 3)),
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
        self.patch_embed = nn.Parameter(torch.randn(N_PATCHES, d))  # which patch
        self.uv_to_q = nn.Linear(2 + 4 * UV_PE_BANDS, d)  # where on it
        self.to_k = nn.Linear(z_dim, N_TOKENS * d)
        self.to_v = nn.Linear(z_dim, N_TOKENS * d)
        self.norm_kv = nn.LayerNorm(d)
        self.layers = nn.ModuleList(DecoderLayer(d, N_HEADS) for _ in range(N_LAYERS))
        self.region = RegionHead(z_dim)
        self.head = ObjectHead(d)

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        """Object slots (B, K, C), their attention (B, K, N), token colours
        (B, N, 3), view 0's SceneFrames -> Gaussians (B, K, G, ...), the box
        {"anchor_init" (B, K, 3), "anchor" (B, K, 2), "extent" (B, K, 3)} and the
        state for decode(); scene frame."""
        b, n_slots = slots.shape[:2]
        anchor_init, colour = init_attention_anchor(attn, colours, frame, self.image_size, self.focal_ratio)
        ray = F.normalize(anchor_init - frame.origin[:, None], dim=-1)  # camera -> anchor_init
        box = {"anchor_init": anchor_init}
        slots, anchor_init, ray, colour = (x.flatten(0, 1) for x in (slots, anchor_init, ray, colour))
        K = slots.shape[0]
        anchor, extent = self.region(slots, anchor_init, ray)
        state = {
            "k": self.norm_kv(self.to_k(slots).view(K, N_TOKENS, self.d)),  # slot tokens
            "v": self.norm_kv(self.to_v(slots).view(K, N_TOKENS, self.d)),
            "anchor": anchor,
            "extent": extent,
            "colour": colour,
            "presence": (attn.mean(-1).flatten() / AREA_FULL).clamp(max=1.0),
            "shape": (b, n_slots),
        }
        box["anchor"] = anchor.unflatten(0, (b, n_slots))
        box["extent"] = extent.unflatten(0, (b, n_slots))
        uv, _ = patch_grid(self.side, slots.device, slots.dtype)
        queries = self.patch_embed[:, None, None] + self.uv_to_q(
            positional_encoding(uv.reshape(-1, 2), UV_PE_BANDS)
        ).view(N_PATCHES, self.side, self.side, -1)
        gaussians, state["q"] = self.decode(queries.reshape(-1, self.d).expand(K, -1, -1), state)
        state["mu"] = gaussians["mu"]  # where the lifter samples
        return gaussians, box, state

    def decode(self, q: torch.Tensor, state: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """q (B*K, G, d) through the decoder layers (cross-attention to the slot tokens) ->
        Gaussians (B, K, G, ...) in the box, and q after the layers."""
        for layer in self.layers:
            q = layer(q, state["k"], state["v"])
        gaussians = self.head(q, state["anchor"], state["extent"], state["colour"])
        # Each slot's brightest Gaussian at its presence.
        opacity = gaussians["opacity"]
        gaussians["opacity"] = opacity / opacity.amax(dim=1, keepdim=True).clamp_min(1e-6) * state["presence"][:, None]
        return {name: value.unflatten(0, state["shape"]) for name, value in gaussians.items()}, q
