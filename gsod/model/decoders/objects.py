"""Object decoder: per object slot, a box on the floor and a Splatter Image of view 0 inside it.

1. box       anchor_init: the attention-weighted mean of where the view-0 token rays meet the
             floor. RegionHead(slot): anchor = anchor_init + a shift along / across the ray;
             half sizes (ex, ey, ez); the box spans anchor +- (ex, ey), z in [0, 2 ez]; its
             Gaussians from BACKGROUND_THICKNESS up, above the floor's
2. window    the box's view-0 projection (a square around it), resampled to WINDOW x WINDOW;
             no gradient (only where to look)
3. Gaussians a U-Net on the window [rgb, the slot's attention], FiLM-conditioned on the slot,
             predicts M Gaussians per window pixel as in Splatter Image:
               depth    sigmoid over the ray's stretch inside the box [t_in, t_out] (Splatter
                        Image: [znear, zfar]); rays missing the box draw nothing
               offset   raw, camera frame
               opacity  sigmoid;  scale  exp;  rotation  unit quaternion in the camera frame
               colour   SH: DC raw, degree 1 in the camera frame
             so every Gaussian centre (before the offset) lies in the box.
Box sizes and shifts unbounded; BOX_INIT only sets where they start. Scene frame.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders.background import BACKGROUND_THICKNESS, FLOOR_HALF
from gsod.model.layers import mlp
from gsod.utils.geometry import SceneFrame, camera_directions, matrix_to_quat, quat_mul, rotate_sh1
from gsod.utils.splatter import SH_C0

BOX_INIT = 0.25  # initial box half side (scene units)
REGION_HIDDEN = 64
ANCHOR_SHARPNESS = 2.0  # anchor_init weights: attention ** this
WINDOW = 48  # window pixels per side
WINDOW_MARGIN, MIN_HALF = 1.1, 0.04  # window half side: this x the box's, at least this x the image
NEAR = 1e-3  # box stretches start at least this far along the ray
# U-Net (SongUNet-like, Splatter Image's backbone): channels per level, blocks per level, attention level.
UNET_CHANNELS, UNET_BLOCKS, UNET_ATTENTION, DROPOUT = (64, 128, 128, 128), 2, 2, 0.1
# Per Gaussian, as in Splatter Image: depth, offset (3), opacity, log scale (3), quat (4), SH DC (3), sh1 (9).
N_OUT = 24
# Output layer init per slice: (xavier gain, bias). Splatter Image's chairs config.
SCALE_INIT = 0.01  # scene units
OUT_INIT = {
    "depth": (slice(0, 1), 1.0, 0.0),
    "offset": (slice(1, 4), 1e-3, 0.0),
    "opacity": (slice(4, 5), 1e-3, -2.0),
    "scale": (slice(5, 8), 1e-3, math.log(SCALE_INIT)),
    "quat": (slice(8, 12), 1.0, 0.0),
    "dc": (slice(12, 15), 5.0, 0.0),
    "sh1": (slice(15, 24), 0.0, 0.0),
}


def init_attention_anchor(attn: torch.Tensor, rays: torch.Tensor, frame: SceneFrame) -> torch.Tensor:
    """attn (B, K, N), the token rays (B, N, 3) -> anchor_init on the floor (B, K, 3)."""
    t = (-frame.origin[:, None, 2:] / rays[..., 2:].clamp(max=-1e-6)).clamp(min=0.0)  # (B, N, 1)
    xy = (frame.origin[:, None, :2] + t * rays[..., :2]).clamp(-FLOOR_HALF, FLOOR_HALF)
    p = (ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(-1)
    return p @ F.pad(xy, (0, 1))


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


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, c // 4), c)


class Block(nn.Module):
    """Residual block: norm, SiLU, conv; FiLM from the slot; norm, SiLU, dropout, conv; optional
    self-attention."""

    def __init__(self, c_in: int, c_out: int, d_emb: int, attention: bool):
        super().__init__()
        self.norm0, self.conv0 = _norm(c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.film = nn.Linear(d_emb, 2 * c_out)
        self.norm1, self.conv1 = _norm(c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.attention = nn.MultiheadAttention(c_out, 1, batch_first=True) if attention else None
        self.norm2 = _norm(c_out) if attention else None

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv0(F.silu(self.norm0(x)))
        scale, shift = self.film(emb)[..., None, None].chunk(2, 1)
        h = F.silu(self.norm1(h) * (scale + 1) + shift)
        h = self.conv1(F.dropout(h, DROPOUT, self.training))
        x = (h + self.skip(x)) / math.sqrt(2)
        if self.attention is not None:
            t = self.norm2(x).flatten(2).transpose(1, 2)  # (N, HW, C)
            x = (x + self.attention(t, t, t, need_weights=False)[0].transpose(1, 2).reshape_as(x)) / math.sqrt(2)
        return x


class UNet(nn.Module):
    """Image (N, c_in, S, S) + embedding (N, d_emb) -> per-pixel outputs (N, c_out, S, S)."""

    def __init__(self, c_in: int, c_out: int, d_emb: int):
        super().__init__()
        c = UNET_CHANNELS
        self.stem = nn.Conv2d(c_in, c[0], 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        c_prev, skips = c[0], []
        for level, ch in enumerate(c):
            for _ in range(UNET_BLOCKS):
                self.down.append(Block(c_prev, ch, d_emb, level == UNET_ATTENTION))
                c_prev = ch
                skips.append(ch)
        for level, ch in reversed(list(enumerate(c))):
            for _ in range(UNET_BLOCKS):
                self.up.append(Block(c_prev + skips.pop(), ch, d_emb, level == UNET_ATTENTION))
                c_prev = ch
        self.out = nn.Sequential(_norm(c_prev), nn.SiLU(), nn.Conv2d(c_prev, c_out, 1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x, skips, i = self.stem(x), [], 0
        for level in range(len(UNET_CHANNELS)):
            if level > 0:
                x = F.avg_pool2d(x, 2)
            for _ in range(UNET_BLOCKS):
                x = self.down[i](x, emb)
                skips.append(x)
                i += 1
        i = 0
        for level in reversed(range(len(UNET_CHANNELS))):
            for _ in range(UNET_BLOCKS):
                x = self.up[i](torch.cat([x, skips.pop()], 1), emb)
                i += 1
            if level > 0:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.out(x)


class ObjectDecoder(nn.Module):
    """Object slots + their attention + view 0 -> Gaussians (B, K, WINDOW*WINDOW*M, ...) and boxes."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        gaussians_per_point: int,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float],
    ):
        super().__init__()
        self.m = m = gaussians_per_point
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        self.region = RegionHead(z_dim)
        self.embed = nn.Sequential(nn.Linear(z_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.unet = UNet(4, m * N_OUT, hidden_dim)  # window [rgb, the slot's attention]
        with torch.no_grad():  # Splatter Image's init, per output slice
            out = self.unet.out[-1]
            weight, bias = out.weight.view(m, N_OUT, -1), out.bias.view(m, N_OUT)
            for part, gain, value in OUT_INIT.values():
                for j in range(m):
                    if gain > 0:
                        nn.init.xavier_uniform_(weight[j, part], gain)
                    else:
                        weight[j, part].zero_()
                    bias[j, part] = value

    def forward(
        self, slots: torch.Tensor, attn: torch.Tensor, image: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Object slots (B, K, C), their attention (B, K, s*s), view-0 image (B, 3, H, W), view 0's
        SceneFrames -> Gaussians (B, K, WINDOW*WINDOW*M, ...) and the box {"anchor_init" (B, K, 3),
        "anchor" (B, K, 2), "extent" (B, K, 3)}; scene frame."""
        h, w = self.image_size
        b, k = slots.shape[:2]
        size = image.new_tensor([w, h])
        f = image.new_tensor([self.focal_ratio[0] * w, self.focal_ratio[1] * h])
        side = int(round(attn.shape[-1] ** 0.5))

        # 1. The box.
        idx = torch.arange(side, device=image.device, dtype=image.dtype) + 0.5
        tv, tu = torch.meshgrid(idx * h / side - 0.5, idx * w / side - 0.5, indexing="ij")
        anchor_init = init_attention_anchor(attn, self.rays(torch.stack([tu, tv], -1).reshape(-1, 2), frame), frame)
        anchor, extent = self.region(slots, anchor_init, frame)

        # 2. The window: a square around the box's view-0 projection (no gradient).
        with torch.no_grad():
            corners = box_corners(anchor, extent)  # (B, K, 8, 3)
            cam = (corners - frame.origin[:, None, None]) @ frame.rot[:, None]  # camera frame
            uv = f * cam[..., :2] / cam[..., 2:].clamp_min(1e-2) + (size - 1) / 2
            lo, hi = uv.amin(-2), uv.amax(-2)  # (B, K, 2)
            centre = (lo + hi) / 2
            half = (WINDOW_MARGIN * (hi - lo).amax(-1) / 2).clamp_min(MIN_HALF * max(h, w))  # (B, K)
            g = (torch.arange(WINDOW, device=image.device, dtype=image.dtype) + 0.5) / WINDOW * 2 - 1
            gv, gu = torch.meshgrid(g, g, indexing="ij")
            points = centre[:, :, None] + half[..., None, None] * torch.stack([gu, gv], -1).reshape(
                -1, 2
            )  # (B, K, P, 2)
        n_points = points.shape[2]
        grid = ((points + 0.5) / size * 2 - 1).flatten(1, 2)[:, :, None]  # (B, K*P, 1, 2)

        def crop(x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B, K, C, WINDOW, WINDOW)
            out = F.grid_sample(x, grid, padding_mode="border", align_corners=False)[..., 0]  # (B, C, K*P)
            return out.unflatten(2, (k, WINDOW, WINDOW)).transpose(1, 2)

        maps = crop(attn.unflatten(2, (side, side)))  # (B, K windows, K maps, WINDOW, WINDOW)
        own = maps[:, torch.arange(k), torch.arange(k)][:, :, None]  # each slot's map in its window
        window = torch.cat([crop(image), own], 2).flatten(0, 1)  # (B*K, 4, WINDOW, WINDOW)

        # 3. Splatter Image on the window, conditioned on the slot.
        raw = self.unet(window, self.embed(slots).flatten(0, 1))  # (B*K, M*N_OUT, WINDOW, WINDOW)
        raw = (
            raw.unflatten(1, (self.m, N_OUT)).flatten(3).permute(0, 3, 1, 2).unflatten(0, (b, k))
        )  # (B, K, P, M, N_OUT)

        rays = self.rays(points.flatten(1, 2), frame).unflatten(1, (k, n_points))  # (B, K, P, 3)
        t_in, t_out, hit = self.box_stretch(rays, anchor, extent, frame)  # (B, K, P)
        depth = t_in[..., None] + torch.sigmoid(raw[..., 0]) * (t_out - t_in)[..., None]  # (B, K, P, M)
        rot = frame.rot[:, None, None, None]  # camera -> scene frame
        offset = (rot @ raw[..., 1:4, None])[..., 0]
        cam_quat = matrix_to_quat(frame.rot)[:, None, None, None].expand(*raw.shape[:4], 4)
        gaussians = {
            "mu": frame.origin[:, None, None, None] + depth[..., None] * rays[..., None, :] + offset,
            "scale": torch.exp(raw[..., 5:8]),
            "quat": quat_mul(cam_quat, F.normalize(raw[..., 8:12], dim=-1)),
            "opacity": torch.sigmoid(raw[..., 4]) * hit[..., None],
            "rgb": 0.5 + SH_C0 * raw[..., 12:15],  # DC
            "sh1": rotate_sh1(raw[..., 15:].unflatten(-1, (3, 3)).flatten(1, 3), frame.rot),
        }
        gaussians["sh1"] = gaussians["sh1"].unflatten(1, raw.shape[1:4])
        box = {"anchor_init": anchor_init, "anchor": anchor, "extent": extent}
        return {name: value.flatten(2, 3) for name, value in gaussians.items()}, box

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
