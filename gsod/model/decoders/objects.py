"""Object decoder: per object slot, a box on the floor and a Splatter-Image-style sketch inside it,
from the slot alone (no image, so the grouping matters); the lifter (lifter.py) refines it.

1. box       anchor_init: the attention-weighted mean of where the view-0 token rays meet the
             floor. RegionHead(slot): anchor = anchor_init + a shift along / across the ray;
             half sizes (ex, ey, ez); the box spans anchor +- (ex, ey), z in [0, 2 ez]; its
             Gaussians from BACKGROUND_THICKNESS up, above the floor's
2. window    the box's view-0 projection (a square around it), WINDOW x WINDOW pixels; no
             gradient (only where to look)
3. q         per window pixel: MLP(slot, PE(the pixel in the window)), no image
4. decode    q -> one Gaussian per window pixel, as in Splatter Image:
               depth    sigmoid over the ray's stretch inside the box [t_in, t_out] (Splatter
                        Image: [znear, zfar]); rays missing the box draw nothing
               offset   camera frame;  rotation  unit quaternion in the camera frame
               scale    exp x the pixel's footprint
               colour   the slot's colour + a residual; SH degree 1 in the camera frame
               opacity  sigmoid, per slot a share of its brightest x presence (attention area /
                        AREA_FULL, at most 1)
             The lifter adds what each Gaussian sees in view 0 to its q; decode again.
Box sizes and shifts unbounded; BOX_INIT only sets where they start. Scene frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.background import BACKGROUND_THICKNESS, FLOOR_HALF
from gsod.model.layers import mlp, positional_encoding, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions, matrix_to_quat, quat_mul, rotate_sh1

BOX_INIT = 0.25  # initial box half side (scene units)
REGION_HIDDEN = 64
ANCHOR_SHARPNESS = 2.0  # anchor_init weights: attention ** this
AREA_FULL = 0.004  # attention area (fraction of tokens) at which a slot is fully present
WINDOW = 48  # window pixels per side
WINDOW_MARGIN, MIN_HALF = 1.1, 0.04  # window half side: this x the box's, at least this x the image
NEAR = 1e-3  # box stretches start at least this far along the ray
PE_BANDS = 8
SCALE_POINT = 0.5  # initial size: this fraction of a window pixel's footprint
# Per Gaussian: depth, offset (3), opacity, log scale (3), quat (4), rgb residual (3), sh1 (9).
N_OUT = 24
# Output layer init per slice: (xavier gain, bias). Splatter Image's chairs config, but colour
# (zero residual), opacity (0.5) and scale (pixel-sized) start visible.
OUT_INIT = {
    "depth": (slice(0, 1), 1.0, 0.0),
    "offset": (slice(1, 4), 1e-3, 0.0),
    "opacity": (slice(4, 5), 1e-3, 0.0),
    "scale": (slice(5, 8), 1e-3, 0.0),
    "quat": (slice(8, 12), 1.0, 0.0),
    "rgb": (slice(12, 15), 0.0, 0.0),
    "sh1": (slice(15, 24), 0.0, 0.0),
}


def init_attention_anchor(
    attn: torch.Tensor, colours: torch.Tensor, rays: torch.Tensor, frame: SceneFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (B, K, N), token colours (B, N, 3), the token rays (B, N, 3) -> anchor_init on the
    floor, the slot's colour (B, K, 3)."""
    t = (-frame.origin[:, None, 2:] / rays[..., 2:].clamp(max=-1e-6)).clamp(min=0.0)  # (B, N, 1)
    xy = (frame.origin[:, None, :2] + t * rays[..., :2]).clamp(-FLOOR_HALF, FLOOR_HALF)
    p = (ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(-1)
    return p @ F.pad(xy, (0, 1)), p @ colours


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
    """Slot -> the box: anchor = anchor_init + shift along / across view 0's ray, half sizes;
    in units of BOX_INIT, zero at first."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.mlp = mlp(z_dim, REGION_HIDDEN, 5)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, slots: torch.Tensor, anchor_init: torch.Tensor, frame: SceneFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """slots (B, K, C), anchor_init (B, K, 3) -> anchor on the floor (B, K, 2), half sizes (B, K, 3)."""
        raw = self.mlp(slots)
        along = F.normalize((anchor_init - frame.origin[:, None])[..., :2], dim=-1)  # the ray, seen from above
        across = torch.stack([-along[..., 1], along[..., 0]], -1)
        shift = BOX_INIT * raw[..., :2]
        anchor = anchor_init[..., :2] + shift[..., :1] * along + shift[..., 1:] * across
        return anchor, BOX_INIT * torch.exp(raw[..., 2:])


class ObjectDecoder(nn.Module):
    """Object slots + their attention -> sketch Gaussians (B, K, WINDOW*WINDOW, ...), boxes and the
    state for decode() (a refined q)."""

    def __init__(self, z_dim: int, hidden_dim: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]):
        super().__init__()
        d = hidden_dim
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        self.region = RegionHead(z_dim)
        self.from_slot = nn.Linear(z_dim, d)
        self.from_point = nn.Linear(2 + 4 * PE_BANDS, d)
        self.hidden = nn.Sequential(nn.ReLU(True), nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        self.head = nn.Sequential(nn.ReLU(True), nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, N_OUT))
        with torch.no_grad():
            out = self.head[-1]
            for part, gain, value in OUT_INIT.values():
                if gain > 0:
                    nn.init.xavier_uniform_(out.weight[part], gain)
                else:
                    out.weight[part].zero_()
                out.bias[part] = value

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        """Object slots (B, K, C), their attention (B, K, s*s), token colours (B, s*s, 3), view 0's
        SceneFrames -> sketch Gaussians (B, K, WINDOW*WINDOW, ...), the box {"anchor_init" (B, K, 3),
        "anchor" (B, K, 2), "extent" (B, K, 3)} and the state for decode(); scene frame."""
        h, w = self.image_size
        k = slots.shape[1]
        size = slots.new_tensor([w, h])
        f = slots.new_tensor([self.focal_ratio[0] * w, self.focal_ratio[1] * h])
        side = int(round(attn.shape[-1] ** 0.5))

        # 1. The box.
        idx = torch.arange(side, device=slots.device, dtype=slots.dtype) + 0.5
        tv, tu = torch.meshgrid(idx * h / side - 0.5, idx * w / side - 0.5, indexing="ij")
        token_rays = self.rays(torch.stack([tu, tv], -1).reshape(-1, 2), frame)
        anchor_init, colour = init_attention_anchor(attn, colours, token_rays, frame)
        anchor, extent = self.region(slots, anchor_init, frame)

        # 2. The window: a square around the box's view-0 projection (no gradient), its pixels' rays.
        g = (torch.arange(WINDOW, device=slots.device, dtype=slots.dtype) + 0.5) / WINDOW * 2 - 1
        gv, gu = torch.meshgrid(g, g, indexing="ij")
        local = torch.stack([gu, gv], -1).reshape(-1, 2)  # (P, 2) in [-1, 1]
        with torch.no_grad():
            corners = box_corners(anchor, extent)  # (B, K, 8, 3)
            cam = (corners - frame.origin[:, None, None]) @ frame.rot[:, None]  # camera frame
            uv = f * cam[..., :2] / cam[..., 2:].clamp_min(1e-2) + (size - 1) / 2
            lo, hi = uv.amin(-2), uv.amax(-2)  # (B, K, 2)
            half = (WINDOW_MARGIN * (hi - lo).amax(-1) / 2).clamp_min(MIN_HALF * max(h, w))  # (B, K)
            points = ((lo + hi) / 2)[:, :, None] + half[..., None, None] * local  # (B, K, P, 2) pixels
        n_points = points.shape[2]
        rays = self.rays(points.flatten(1, 2), frame).unflatten(1, (k, n_points))  # (B, K, P, 3)
        t_in, t_out, hit = self.box_stretch(rays, anchor, extent, frame)  # (B, K, P)

        # 3. q per window pixel, from the slot alone.
        q = self.hidden(self.from_slot(slots)[:, :, None] + self.from_point(positional_encoding(local, PE_BANDS)))
        state = {
            "rays": rays,
            "t_in": t_in,
            "t_out": t_out,
            "hit": hit,
            "spacing": 2 * half / WINDOW,  # (B, K) window pixel width, image pixels
            "colour": colour,
            "presence": (attn.mean(-1) / AREA_FULL).clamp(max=1.0),  # (B, K)
            "q": q.flatten(0, 1),  # (B*K, P, d)
            "frame": frame,
        }
        gaussians = self.decode(state["q"], state)
        state["mu"] = gaussians["mu"]  # where the lifter samples
        return gaussians, {"anchor_init": anchor_init, "anchor": anchor, "extent": extent}, state

    def decode(self, q: torch.Tensor, state: dict) -> dict[str, torch.Tensor]:
        """q (B*K, P, d), e.g. refined by the lifter -> Gaussians (B, K, P, ...), one per window pixel."""
        frame = state["frame"]
        b, k = state["presence"].shape
        raw = self.head(q).unflatten(0, (b, k))  # (B, K, P, N_OUT)
        t_in, t_out = state["t_in"], state["t_out"]
        depth = t_in + torch.sigmoid(raw[..., 0]) * (t_out - t_in)  # (B, K, P)
        rot = frame.rot[:, None, None]  # camera -> scene frame
        footprint = depth * state["spacing"][..., None] / (self.image_size[1] * self.focal_ratio[0])
        cam_quat = matrix_to_quat(frame.rot)[:, None, None].expand(*raw.shape[:3], 4)
        opacity = torch.sigmoid(raw[..., 4]) * state["hit"]
        opacity = opacity / opacity.amax(-1, keepdim=True).clamp_min(1e-6) * state["presence"][..., None]
        sh1 = rotate_sh1(raw[..., 15:].unflatten(-1, (3, 3)).flatten(1, 2), frame.rot).unflatten(1, (k, -1))
        return {
            "mu": frame.origin[:, None, None] + depth[..., None] * state["rays"] + (rot @ raw[..., 1:4, None])[..., 0],
            "scale": SCALE_POINT * footprint[..., None] * torch.exp(raw[..., 5:8]),
            "quat": quat_mul(cam_quat, F.normalize(raw[..., 8:12], dim=-1)),
            "opacity": opacity,
            "rgb": shift_colour(state["colour"][:, :, None], raw[..., 12:15]),
            "sh1": sh1,
        }

    def rays(self, uv: torch.Tensor, frame: SceneFrame) -> torch.Tensor:
        """View-0 pixels (..., N, 2) -> rays (B, N, 3), scene frame, camera z = 1."""
        return camera_directions(uv[..., 0], uv[..., 1], self.image_size, self.focal_ratio) @ frame.rot.mT

    @staticmethod
    def box_stretch(
        rays: torch.Tensor, anchor: torch.Tensor, extent: torch.Tensor, frame: SceneFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rays (B, K, P, 3), each slot's box -> where each ray enters and leaves its box (camera
        depth, B, K, P) and whether it meets it at all (0 / 1)."""
        bottom = torch.full_like(extent[..., :1], BACKGROUND_THICKNESS)  # above the floor's Gaussians
        lo = torch.cat([anchor - extent[..., :2], bottom], -1)[:, :, None]
        hi = torch.cat([anchor + extent[..., :2], 2.0 * extent[..., 2:]], -1)[:, :, None]
        safe = torch.where(rays.abs() < 1e-6, torch.full_like(rays, 1e-6), rays)
        origin = frame.origin[:, None, None]
        t0, t1 = (lo - origin) / safe, (hi - origin) / safe
        t_in = torch.minimum(t0, t1).amax(-1).clamp_min(NEAR)
        t_out = torch.maximum(t0, t1).amin(-1)
        hit = (t_out > t_in).to(rays.dtype)
        return t_in, torch.maximum(t_out, t_in), hit
