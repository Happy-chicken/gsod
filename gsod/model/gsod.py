"""Slot model: encoder -> slot attention -> object decoder + static background -> splatting.

views[0] -> Encoder -> tokens -> SlotAttention -> K-1 object slots
object slots -> Gaussian decoder -> object Gaussians (view-0 azimuth frame -> world)
BackgroundShell (learnable, scene-independent) -> background Gaussians (world)
all Gaussians -> splatting

Slot indices used by every output (masks, slot_alpha, attention rows):
    0       background (BackgroundShell; attention row 0 = learned bg query)
    1..K-1  object slots
``num_slots`` = K counts the background, so configs keep their meaning.

Frames
------
Only views[0] is encoded and the camera azimuth is uniform over 360 deg, so the
image alone does not say where objects are in *world* xy. Object Gaussians are
therefore predicted in the azimuth frame of view 0 (yaw-rotated so view 0 looks
along +x; uORF's nss2cam0) and rotated into the world before splatting. The
background shell is static and lives in the world frame. Cameras are never
modified.

Discovery
---------
No foreground target is given. Early in training (``anchor_weight`` < 1, set
by train.py) each object slot's attention is biased towards its own part of
the image (SlotAttention's position prior, weight 1 - anchor_weight), its
frame on the ray through that part is widened (SlotFrame), and object
Gaussians get an opacity floor (``opacity_floor``). A slot therefore sees
the content of its part of the image and draws it there; the multi-view
reconstruction pulls its Gaussians onto the objects, and the attention
learns from the rendered slot partition (losses.py) while the bias fades.
``anchor_weight`` = 1 and no floor is the trained model.

Everything that is rendered comes from Gaussians: there is no constant colour
fill; a pixel no Gaussian covers renders black.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.background import BackgroundShell
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


def attention_rays(
    attn_obj: torch.Tensor,
    cam2world0: torch.Tensor,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray of view 0 through each object slot's attention centroid.

    attn_obj: (K-1, s*s) object rows of the view-0 slot attention.
    Returns (origin (3,), directions (K-1, 3)) in the view-0 azimuth frame,
    differentiable in attn_obj. The azimuth frame is a rotation about +z, so
    heights are world heights; SlotFrame intersects each ray with the height
    it predicts for that slot.
    """
    side = int(round(attn_obj.shape[1] ** 0.5))
    h, w = image_size
    weights = attn_obj / attn_obj.sum(dim=-1, keepdim=True)
    idx = torch.arange(side, device=attn_obj.device, dtype=attn_obj.dtype)
    # Token centres in image pixel coordinates (pixel centres at integers,
    # matching the dataset; F.interpolate with align_corners=False).
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
    c2w = cam2world0.to(attn_obj.dtype)
    to_azimuth = _yaw_matrix(-view_yaw(cam2world0)).to(attn_obj.dtype)
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
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        scene_config: dict | None = None,
        canonical_azimuth: bool = True,
        anchor_frames: bool = True,
        decoder_cls: type[nn.Module] = AttentionDecoder,
    ):
        """num_slots: K, background included (K-1 object slots).
        bg_gaussians_per_slot: number of background-shell Gaussians.
        scene_config: the SceneGenConfig dict from meta.json; sets the
        object-frame priors and the background-shell radius.
        decoder_cls: object decoder class (AttentionDecoder, SharedMLPDecoder,
        PerSlotMLPDecoder), built with the common decoder signature.
        Object slots are initialised BO-QSA style (see slot_attention.py)."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.splat = splat
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.num_slots = num_slots
        self.focal_ratio = tuple(focal_ratio)
        self.near = near
        self.canonical_azimuth = bool(canonical_azimuth)
        # Object frame centres follow the slot-attention centroids
        # (requires canonical_azimuth, as the rays are in that frame).
        self.anchor_frames = bool(anchor_frames) and self.canonical_azimuth

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots - 1,
            in_dim=z_dim,
            slot_dim=z_dim,
            iters=attn_iter,
        )
        frame = (
            SlotFrame.from_scene_config(z_dim, scene_config, n_slots=num_slots - 1)
            if scene_config is not None
            else SlotFrame(z_dim, n_slots=num_slots - 1)
        )
        self.decoder = decoder_cls(
            n_slots=num_slots - 1,
            z_dim=z_dim,
            gaussians_per_slot=gaussians_per_slot,
            hidden_dim=decoder_hidden_dim,
            frame=frame,
        )
        # Background shell well outside every camera.
        radius = 20.0
        if scene_config is not None:
            cam_max = float(scene_config["distance_range"][1]) + float(
                scene_config.get("target_height", 0.4)
            )
            radius = 3.0 * cam_max
        self.background = BackgroundShell(
            n_gaussians=bg_gaussians_per_slot, radius=radius
        )
        # Discovery schedule, set per step by train.py (see the module
        # docstring); these defaults are the trained model.
        self.anchor_weight = 1.0
        self.opacity_floor = 0.0

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "Decoder": self.decoder,
            "Background": self.background,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N,3,H,W) -> (1, s*s, C) encoder tokens of views[0], s = input_size."""
        feat = self.encoder(
            F.interpolate(
                images[0:1], size=self.input_size, mode="bilinear", align_corners=False
            )
        )
        return feat.flatten(start_dim=2).permute(0, 2, 1)

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """images: (N,3,H,W) in [0,1]; cam2world: (N,4,4) world frame.
        Only views[0] is encoded."""
        tokens = self.encode(images)
        obj_slots, attn = self.slot_attention(
            tokens, position_weight=1.0 - self.anchor_weight
        )
        obj_slots, attn = obj_slots.squeeze(0), attn.squeeze(0)  # (K-1, C), (K, N)

        # Object Gaussians (view-0 azimuth frame), with attention-anchored frames.
        frame = getattr(getattr(self.decoder, "head", None), "frame", None)
        if frame is not None:
            frame.anchor_weight = self.anchor_weight
        if frame is not None and self.anchor_frames:
            frame.ray = attention_rays(
                attn[1:], cam2world[0], self.image_size, self.focal_ratio
            )
        frame_centre = None
        try:
            obj_gaussians, obj_probs = self.decoder(obj_slots)
            if frame is not None:
                with torch.no_grad():  # frames actually used, for diagnostics
                    frame_centre = frame(obj_slots)[0]
        finally:
            if frame is not None:
                frame.ray = None
        if self.canonical_azimuth:
            obj_gaussians = yaw_to_world(obj_gaussians, view_yaw(cam2world[0]))
        if self.training and self.opacity_floor > 0:
            floor = self.opacity_floor
            obj_gaussians = {
                **obj_gaussians,
                "opacity": floor + (1.0 - floor) * obj_gaussians["opacity"],
            }

        # Static background Gaussians (world frame) + object Gaussians.
        bg_gaussians = self.background()
        n_bg = self.background.n_gaussians
        gaussians = {
            name: torch.cat([bg_gaussians[name], obj_gaussians[name]], dim=0)
            for name in bg_gaussians
        }
        device = obj_probs.device
        slot_ids = torch.cat(
            [
                torch.zeros(n_bg, dtype=torch.long, device=device),
                obj_probs.argmax(-1) + 1,
            ]
        )
        slot_probs = F.one_hot(slot_ids, self.num_slots).to(obj_probs.dtype)

        rendered = self.splat(
            gaussians,
            slot_probs,
            cam2world,
            self.image_size,
            self.focal_ratio,
            self.near,
        )

        # Per-slot weights sum to the pixel's Gaussian coverage. Uncovered
        # pixels render black; for masks they count as background.
        alpha = rendered["slot_alpha"]
        coverage = alpha.sum(dim=1, keepdim=True)
        empty = (1.0 - coverage).clamp(min=0.0)
        alpha_full = torch.cat([alpha[:, :1] + empty, alpha[:, 1:]], dim=1)
        mask_idx = alpha_full.argmax(dim=1)

        return {
            "rgb": rendered["rgb"],
            "slot_alpha": alpha,  # Gaussian weights only
            "slot_alpha_full": alpha_full,  # uncovered share added to slot 0
            "mask_idx": mask_idx,
            "attn": attn,  # (K, s*s); row 0 = background query
            "slots": obj_slots,  # (K-1, C) object slots
            "slot_ids": slot_ids,
            "gaussians": gaussians,  # world frame, background first
            # (M, 3) object Gaussians' positions in their frames, or None
            "obj_offset": obj_gaussians.get("offset"),
            "frame_centre": frame_centre,  # (K-1, 3) object frames, view-0 azimuth frame
        }
