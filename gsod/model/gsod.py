"""Slot model: encoder -> slot attention -> Gaussian decoder -> splatting.

views[0] -> Encoder -> tokens -> SlotAttention -> K slots (0 = background)
    all slots       -> 2D spatial broadcast decoder -> view 0 (training signal)
    background slot -> background Gaussian decoder --+
    object slots    -> object Gaussian decoder ------+-> Gaussians (view-0
                       azimuth frame -> world) -> splatting into every view

Structural prior, as in uORF (no masks or foreground targets): slot 0 is the
background, with its own slot-attention parameters and its own decoder,
whose frame spans the scene out beyond the cameras; object frames are
object-sized and inside the placement area (SlotFrame). Which pixels are
background is still learned from reconstruction alone.

Coarse to fine: the 2D decoder's slot masks are a coarse decomposition of
view 0. They guide the 3D model in two ways, without gradient back into the
2D branch (the 2D masks are learned from 2D reconstruction alone):
  * ray anchoring: each object slot's frame centre is pulled onto the ray of
    view 0 through the centroid of its 2D mask (weight ``anchor_weight``,
    raised by train.py once the 2D decomposition has formed);
  * the consistency loss (losses.py): the rendered slot split of view 0
    follows the 2D masks (faded out late in training, so the 3D
    reconstruction refines what 2D cannot resolve).

Slot indices in every output (masks, slot_alpha, attention rows):
    0       background
    1..K-1  objects

Only views[0] is encoded, in training as at inference.

Training objective: reconstruction only (losses.py). Every gradient reaches
every module; the 2D and the 3D reconstruction share the encoder and the
slot attention.

Frames
------
Only views[0] is decoded into 3D and the camera azimuth is uniform over 360
deg, so the image alone does not say where things are in *world* xy.
Gaussians are therefore predicted in the azimuth frame of view 0 (yaw-rotated
so view 0 looks along +x; uORF's nss2cam0), each slot's frame from its latent
alone (SlotFrame), and rotated into the world before splatting. Cameras are
never modified.

Everything that is rendered comes from Gaussians: there is no constant colour
fill; a pixel no Gaussian covers renders black.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from gsod.data_generator.generator import SceneGenConfig
from gsod.model.aux_decoder import SpatialBroadcastDecoder
from gsod.model.decoders import AttentionDecoder, SlotFrame
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


def mask_rays(
    masks: torch.Tensor,
    cam2world0: torch.Tensor,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray of view 0 through the centroid of each slot mask.

    masks: (K, a, a) slot masks of view 0 (e.g. the 2D decoder's alphas).
    Returns (origin (3,), directions (K, 3)) in the view-0 azimuth frame. The
    azimuth frame is a rotation about +z, so heights are world heights;
    SlotFrame intersects each ray with the height it predicts for that slot.
    """
    side = masks.shape[-1]
    h, w = image_size
    weights = masks.flatten(1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    idx = torch.arange(side, device=masks.device, dtype=masks.dtype)
    # Mask cell centres in image pixel coordinates (pixel centres at integers,
    # matching the dataset and the area-downsampled 2D target).
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    u_bar = (weights * uu.reshape(-1)).sum(-1)
    v_bar = (weights * vv.reshape(-1)).sum(-1)

    fx, fy = focal_ratio[0] * w, focal_ratio[1] * h
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    d_cam = torch.stack(
        [(u_bar - cx) / fx, (v_bar - cy) / fy, torch.ones_like(u_bar)], -1
    )
    c2w = cam2world0.to(masks.dtype)
    to_azimuth = _yaw_matrix(-view_yaw(cam2world0)).to(masks.dtype)
    # World ray o + t * R d_cam, rotated into the azimuth frame.
    origin = to_azimuth @ c2w[:3, 3]
    directions = d_cam @ (to_azimuth @ c2w[:3, :3]).T
    return origin, directions


class GSODModel(nn.Module):
    def __init__(
        self,
        splat: Splatter,
        image_size: tuple[int, int] = (128, 128),
        input_size: int = 64,
        z_dim: int = 64,
        num_slots: int = 6,
        attn_iter: int = 3,
        gaussians_per_slot: int = 512,
        bg_gaussians_per_slot: int = 1024,
        decoder_hidden_dim: int = 128,
        aux2d_size: int = 64,
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        scene_config: dict | None = None,
        canonical_azimuth: bool = True,
        decoder_cls: type[nn.Module] = AttentionDecoder,
    ):
        """num_slots: K, the background slot included.
        bg_gaussians_per_slot: Gaussians of the background slot.
        aux2d_size: side of the 2D reconstruction of view 0.
        scene_config: the SceneGenConfig dict from meta.json (default: the
        SceneGenConfig defaults); sets the frame bounds.
        decoder_cls: Gaussian decoder class (AttentionDecoder, SharedMLPDecoder,
        PerSlotMLPDecoder), built with the common decoder signature.
        Slots are initialised BO-QSA style (see slot_attention.py)."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.splat = splat
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.num_slots = num_slots
        self.focal_ratio = tuple(focal_ratio)
        self.near = near
        self.canonical_azimuth = bool(canonical_azimuth)

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, in_dim=z_dim, slot_dim=z_dim, iters=attn_iter
        )
        self.aux_decoder = SpatialBroadcastDecoder(z_dim, size=aux2d_size)
        scene_config = scene_config or asdict(SceneGenConfig())
        self.decoder = decoder_cls(
            n_slots=num_slots - 1,
            z_dim=z_dim,
            gaussians_per_slot=gaussians_per_slot,
            hidden_dim=decoder_hidden_dim,
            frame=SlotFrame.from_scene_config(z_dim, scene_config),
        )
        # Weight of the ray-anchored object frame centres (SlotFrame); set
        # per step by train.py, 1 = the trained model.
        self.anchor_weight = 1.0
        self.bg_decoder = decoder_cls(
            n_slots=1,
            z_dim=z_dim,
            gaussians_per_slot=bg_gaussians_per_slot,
            hidden_dim=decoder_hidden_dim,
            frame=SlotFrame.from_scene_config(z_dim, scene_config, background=True),
        )

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "AuxDecoder": self.aux_decoder,
            "Decoder": self.decoder,
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

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """images: (N,3,H,W) in [0,1]; cam2world: (N,4,4) world frame.
        Only views[0] is encoded."""
        slots, attn = self.slot_attention(self.encode(images[0:1]))
        aux_rgb, aux_alpha = self.aux_decoder(slots)
        slots, attn = slots[0], attn[0]  # (K, C), (K, s*s)

        # Gaussians (view-0 azimuth frame), background first. Object frames
        # are anchored to the rays through their 2D masks' centroids.
        frames = [
            getattr(getattr(d, "head", None), "frame", None)
            for d in (self.bg_decoder, self.decoder)
        ]
        obj_frame = frames[1]
        if obj_frame is not None and self.canonical_azimuth:
            obj_frame.ray = mask_rays(
                aux_alpha[0, 1:].detach(),
                cam2world[0],
                self.image_size,
                self.focal_ratio,
            )
            obj_frame.anchor_weight = self.anchor_weight
        frame_centre = None
        try:
            bg_gaussians, _ = self.bg_decoder(slots[:1])
            obj_gaussians, obj_probs = self.decoder(slots[1:])
            if None not in frames:
                with torch.no_grad():  # frames actually used, for diagnostics
                    frame_centre = torch.cat(
                        [frames[0](slots[:1])[0], frames[1](slots[1:])[0]]
                    )
        finally:
            if obj_frame is not None:
                obj_frame.ray = None
        gaussians = {
            name: torch.cat([bg_gaussians[name], obj_gaussians[name]])
            for name in bg_gaussians
        }
        n_bg = bg_gaussians["mu"].shape[0]
        slot_ids = torch.cat(
            [obj_probs.new_zeros(n_bg, dtype=torch.long), obj_probs.argmax(-1) + 1]
        )
        slot_probs = F.one_hot(slot_ids, self.num_slots).to(obj_probs.dtype)
        if self.canonical_azimuth:
            gaussians = yaw_to_world(gaussians, view_yaw(cam2world[0]))

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
            "slot_ids": slot_ids,  # (M,) slot of each Gaussian
            "gaussians": gaussians,  # world frame, background first
            "frame_centre": frame_centre,  # (K, 3) frames, view-0 azimuth frame
            "aux_rgb": aux_rgb,  # (1, 3, a, a) 2D reconstruction of view 0
            "aux_alpha": aux_alpha,  # (1, K, a, a)
        }
