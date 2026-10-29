"""Slot model: encoder -> slot attention -> pixel Gaussians -> splatting.

views[0] -> Encoder -> features -> SlotAttention -> K slots (0 = background)
    features -> PixelGeometry -> Gaussians on the rays of view 0 (geometry)
    Gaussian positions + slots -> SlotMixer -> colour, opacity and slot
        probabilities of every Gaussian (uORF's density mixture)
    background slot -> background Gaussians outside view 0
    all Gaussians (view-0 azimuth frame -> world) -> splatting into every view

Where the slots' competition comes from: every slot is asked at every
Gaussian and the slot probabilities sum to one there (SlotMixer), as uORF
does at every point of its radiance field, so the reconstruction of every
view rewards the slot that explains a point best. Geometry comes from the
pixels and knows nothing about slots, so no surface is lost because no slot
claimed it.

Structural prior, as in uORF (no masks or foreground targets): slot 0 is the
background, with its own slot-attention parameters and its own field in the
mixer; object densities are zero outside the box objects can occupy. The
background also owns a set of Gaussians for what view 0 does not see (out
of its frustum); they are switched off where view 0 looks, so they cannot
cover the objects there.

Slot indices in every output (masks, slot_alpha, attention rows):
    0       background
    1..K-1  objects

Only views[0] is encoded, in training as at inference.

Training objective: reconstruction only (losses.py).

Frames
------
Only views[0] is decoded into 3D and the camera azimuth is uniform over 360
deg, so the image alone does not say where things are in *world* xy.
Gaussians are therefore built in the azimuth frame of view 0 (yaw-rotated so
view 0 looks along +x; uORF's nss2cam0), where the mixer sees their
positions, and rotated into the world before splatting. Cameras are never
modified.

Everything that is rendered comes from Gaussians: there is no constant colour
fill; a pixel no Gaussian covers renders black.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from gsod.data_generator.generator import SceneGenConfig
from gsod.model.decoders import (
    AttentionDecoder,
    PixelGeometry,
    SceneBounds,
    SlotFrame,
    SlotMixer,
)
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention

Splatter = Callable[..., dict[str, torch.Tensor]]


def view_yaw(cam2world: torch.Tensor) -> torch.Tensor:
    """Azimuth (rad) of one camera's viewing direction about world +z.

    Same as gsod.utils.camera.camera_azimuth.
    """
    forward = cam2world[:3, 2]
    return torch.atan2(forward[1], forward[0])


def _yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [
            torch.stack([c, -s, zero]),
            torch.stack([s, c, zero]),
            torch.stack([zero, zero, one]),
        ]
    )


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def yaw_to_world(
    gaussians: dict[str, torch.Tensor], yaw: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Rotate Gaussians from the view-0 azimuth frame into the world frame."""
    rot = _yaw_matrix(yaw).to(gaussians["mu"].dtype)
    half = 0.5 * yaw
    q_yaw = torch.stack(
        [
            torch.cos(half),
            torch.zeros_like(half),
            torch.zeros_like(half),
            torch.sin(half),
        ]
    ).to(gaussians["quat"].dtype)
    return {
        **gaussians,
        "mu": gaussians["mu"] @ rot.T,
        "quat": _quat_mul(q_yaw.expand_as(gaussians["quat"]), gaussians["quat"]),
    }


def view0_pose(cam2world0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """View 0's camera in its own azimuth frame: (origin (3,), rotation
    (3, 3), columns = camera axes). A rotation about +z, so heights are world
    heights."""
    c2w = cam2world0
    to_azimuth = _yaw_matrix(-view_yaw(c2w)).to(c2w.dtype)
    return to_azimuth @ c2w[:3, 3], to_azimuth @ c2w[:3, :3]


def pixel_rays(
    rotation: torch.Tensor,
    side: int,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> torch.Tensor:
    """Directions (side*side, 3), row-major, of the rays through the centres
    of a side x side grid over the image (the encoder's feature cells),
    rotated by the camera rotation. Camera z = 1, so the ray parameter is
    the camera depth."""
    h, w = image_size
    idx = torch.arange(side, device=rotation.device, dtype=rotation.dtype)
    # Cell centres in image pixel coordinates (pixel centres at integers,
    # matching the dataset and the bilinear resize into the encoder).
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    fx, fy = focal_ratio[0] * w, focal_ratio[1] * h
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    d_cam = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1)
    return d_cam.reshape(-1, 3) @ rotation.T


def outside_view(
    mu: torch.Tensor,
    origin: torch.Tensor,
    rotation: torch.Tensor,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
    near: float,
) -> torch.Tensor:
    """(M,) True where a Gaussian centre is not in view of the camera."""
    p = (mu - origin) @ rotation  # camera coordinates
    h, w = image_size
    z = p[:, 2].clamp_min(near)
    u = focal_ratio[0] * w * p[:, 0] / z + (w - 1.0) / 2.0
    v = focal_ratio[1] * h * p[:, 1] / z + (h - 1.0) / 2.0
    seen = (
        (p[:, 2] > near)
        & (u >= -0.5)
        & (u <= w - 0.5)
        & (v >= -0.5)
        & (v <= h - 0.5)
    )
    return ~seen


class GSODModel(nn.Module):
    def __init__(
        self,
        splat: Splatter,
        image_size: tuple[int, int] = (128, 128),
        input_size: int = 64,
        z_dim: int = 64,
        num_slots: int = 6,
        attn_iter: int = 3,
        gaussians_per_pixel: int = 2,
        bg_gaussians: int = 1024,
        decoder_hidden_dim: int = 128,
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        scene_config: dict | None = None,
    ):
        """num_slots: K, the background slot included.
        gaussians_per_pixel: Gaussians on the ray of each encoder feature
        cell (input_size x input_size cells).
        bg_gaussians: Gaussians of the background outside view 0.
        scene_config: the SceneGenConfig dict from meta.json (default: the
        SceneGenConfig defaults); sets the depth range, the object box and
        the background frame.
        Slots are initialised BO-QSA style (see slot_attention.py)."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.splat = splat
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.num_slots = num_slots
        self.focal_ratio = tuple(focal_ratio)
        self.near = near

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, in_dim=z_dim, slot_dim=z_dim, iters=attn_iter
        )
        scene_config = scene_config or asdict(SceneGenConfig())
        bounds = SceneBounds.from_config(scene_config)
        self.geometry = PixelGeometry(
            z_dim,
            decoder_hidden_dim,
            gaussians_per_pixel,
            bounds,
            near,
            # Feature cell width at depth 1: image width / input_size pixels
            # over the focal length (geometric mean of x and y).
            cell=1.0 / (input_size * math.sqrt(focal_ratio[0] * focal_ratio[1])),
        )
        self.mixer = SlotMixer(z_dim, decoder_hidden_dim, bounds)
        self.bg_decoder = AttentionDecoder(
            z_dim,
            bg_gaussians,
            decoder_hidden_dim,
            SlotFrame.from_scene_config(z_dim, scene_config),
        )

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "Geometry": self.geometry,
            "Mixer": self.mixer,
            "BgDecoder": self.bg_decoder,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N,3,H,W) -> (N, s*s, C) encoder tokens, s = input_size."""
        feat = self.encoder(
            F.interpolate(
                images, size=self.input_size, mode="bilinear", align_corners=False
            )
        )
        return feat.flatten(start_dim=2).permute(0, 2, 1)

    def _decode(
        self, tokens: torch.Tensor, slots: torch.Tensor, cam2world0: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """tokens: (s*s, C) view-0 features; slots: (K, C), 0 = background.

        Returns (world-frame Gaussians, pixel Gaussians first (layer-major),
        then the background's; (M, K) slot probabilities; (G, s*s) depths of
        the pixel Gaussians)."""
        side = self.input_size
        origin, rotation = view0_pose(cam2world0)
        directions = pixel_rays(rotation, side, self.image_size, self.focal_ratio)
        feat = tokens.T.reshape(-1, side, side)
        geometry, depth = self.geometry(feat, origin, directions)
        rgb, opacity, probs = self.mixer(geometry["mu"], slots)

        bg, bg_probs = self.bg_decoder(slots[:1])
        with torch.no_grad():
            keep = outside_view(
                bg["mu"], origin, rotation, self.image_size, self.focal_ratio, self.near
            )
        bg = {**bg, "opacity": bg["opacity"] * keep.to(bg["opacity"].dtype)}

        pixel = {**geometry, "rgb": rgb, "opacity": opacity}
        gaussians = {name: torch.cat([pixel[name], bg[name]]) for name in pixel}
        slot_probs = torch.cat([probs, F.pad(bg_probs, (0, self.num_slots - 1))])
        return yaw_to_world(gaussians, view_yaw(cam2world0)), slot_probs, depth

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """images: (N,3,H,W) in [0,1]; cam2world: (N,4,4) world frame.
        Only views[0] is encoded."""
        tokens = self.encode(images[0:1])
        slots, attn = self.slot_attention(tokens)
        slots, attn = slots[0], attn[0]  # (K, C), (K, s*s)
        gaussians, slot_probs, depth = self._decode(tokens[0], slots, cam2world[0])
        rendered = self.splat(
            gaussians,
            slot_probs,
            cam2world,
            self.image_size,
            self.focal_ratio,
            self.near,
        )
        # Per-slot weights sum to the pixel's Gaussian coverage; uncovered
        # pixels render black and take the argmax of zeros (slot 0, the
        # background) in masks.
        alpha = rendered["slot_alpha"]
        return {
            "rgb": rendered["rgb"],
            "slot_alpha": alpha,  # (N, K, H, W)
            "mask_idx": alpha.argmax(dim=1),  # (N, H, W)
            "attn": attn,  # (K, s*s) view-0 slot attention; row 0 = background
            "slots": slots,  # (K, C) view-0 slots; 0 = background
            "slot_probs": slot_probs,  # (M, K) slot probabilities per Gaussian
            "gaussians": gaussians,  # world frame, pixel Gaussians first
            "depth": depth,  # (G, s*s) camera depths of the pixel Gaussians
        }
