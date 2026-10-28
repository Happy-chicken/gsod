"""Slot model: encoder -> slot attention -> Gaussian decoder -> splatting.

views[0] -> Encoder -> tokens -> SlotAttention -> K slots
    slots -> 2D spatial broadcast decoder -> view 0 in 2D (training signal)
    slots -> Gaussian decoder -> Gaussians (view-0 azimuth frame -> world)
          -> splatting into every view

All K slots are interchangeable: there is no background slot, background
model or foreground target. Which slots draw the background (one or several)
is decided by the slot competition, driven by reconstruction alone (2D and
multi-view 3D). FG-ARI ignores the GT background, so it does not matter how
the background is split.

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

import torch
import torch.nn.functional as F
from torch import nn

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
        decoder_hidden_dim: int = 128,
        aux2d_size: int = 64,
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        scene_config: dict | None = None,
        canonical_azimuth: bool = True,
        decoder_cls: type[nn.Module] = AttentionDecoder,
    ):
        """num_slots: K, all interchangeable.
        aux2d_size: side of the 2D reconstruction of view 0.
        scene_config: the SceneGenConfig dict from meta.json; sets the frame
        priors (scene size and reach).
        decoder_cls: Gaussian decoder class (AttentionDecoder, SharedMLPDecoder,
        PerSlotMLPDecoder), built with the common decoder signature.
        Slots are initialised BO-QSA style (see slot_attention.py)."""
        super().__init__()
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
        frame = (
            SlotFrame.from_scene_config(z_dim, scene_config)
            if scene_config is not None
            else SlotFrame(z_dim)
        )
        self.decoder = decoder_cls(
            n_slots=num_slots,
            z_dim=z_dim,
            gaussians_per_slot=gaussians_per_slot,
            hidden_dim=decoder_hidden_dim,
            frame=frame,
        )

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "AuxDecoder": self.aux_decoder,
            "Decoder": self.decoder,
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

        # Gaussians (view-0 azimuth frame).
        gaussians, slot_probs = self.decoder(slots)
        frame = getattr(getattr(self.decoder, "head", None), "frame", None)
        frame_centre = None
        if frame is not None:
            with torch.no_grad():  # for diagnostics
                frame_centre = frame(slots)[0]
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
        # pixels render black and take the argmax of zeros (slot 0) in masks.
        alpha = rendered["slot_alpha"]

        return {
            "rgb": rendered["rgb"],
            "slot_alpha": alpha,  # (N, K, H, W)
            "mask_idx": alpha.argmax(dim=1),  # (N, H, W)
            "attn": attn,  # (K, s*s) view-0 slot attention
            "slots": slots,  # (K, C) view-0 slots
            "slot_ids": slot_probs.argmax(-1),  # (M,) slot of each Gaussian
            "gaussians": gaussians,  # world frame
            "frame_centre": frame_centre,  # (K, 3) frames, view-0 azimuth frame
            "aux_rgb": aux_rgb,  # (1, 3, a, a) 2D reconstruction of view 0
            "aux_alpha": aux_alpha,  # (1, K, a, a)
        }
