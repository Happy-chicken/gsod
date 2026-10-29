"""Slot model: encoder -> slot attention -> Gaussians per slot -> splatting.

views[0] -> Encoder -> tokens -> SlotAttention -> K slots (0 = background)
    background slot -> background decoder: floor layer + far shell
    object slots    -> object decoder: Gaussians anywhere in the object box
    each slot's Gaussians (view-0 azimuth frame -> world) -> splatted on its
        own into every view -> the slots alpha-composited as layers in a
        soft depth order

The slots compete in the 3D render itself, with no 2D decoder and no masks:

  * one share of each pixel: the slots' compositing weights w_k(p) sum to
    the pixel's coverage (``composite``), so a slot gains a pixel only from
    the slots it hides or that hide it, and the reconstruction gradient
    moves the pixel to the slot whose colour fits; w_k is also slot k's
    mask, in every view.
  * every slot can reach every object: object Gaussians start spread over
    the whole object box (no per-slot frame).
  * limited capacity: the background cannot enter the object box and has a
    lower bound on scale; an object slot has one base colour (per-Gaussian
    residuals are penalised) and is pulled compact (compactness term, from
    train.py's schedule), so one slot is a poor fit for two objects.

Slot indices in every output (masks, slot_alpha, attention rows):
    0       background
    1..K-1  objects

Only views[0] is encoded, in training as at inference.

Frames
------
Only views[0] is decoded into 3D and the camera azimuth is uniform over 360
deg, so the image alone does not say where things are in *world* xy.
Gaussians are therefore predicted in the azimuth frame of view 0 (yaw-rotated
so view 0 looks along +x; uORF's nss2cam0) and rotated into the world before
splatting. Cameras are never modified.

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
from gsod.model.decoders import AttentionDecoder, BackgroundHead, ObjectHead, SceneBounds
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention

Splatter = Callable[..., dict[str, torch.Tensor]]

# A slot's transparent part counts as this much of a layer at the far depth.
_ALPHA_EPS = 1e-3


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


def composite(
    layers: list[dict[str, torch.Tensor]], tau: float, far: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha compositing of the slots as layers, in a soft depth order.

    layers: per slot, the splatter output (V, C, H, W): accumulated "rgb"
    and "depth", and "alpha". Slot k sits at its expected depth d_k (pulled
    to the far depth where it is nearly transparent). Slot j is in front of
    k with probability sigmoid((d_k - d_j) / tau) and then hides alpha_j of
    it:

        T_k   = prod_{j != k} (1 - alpha_j * sigmoid((d_k - d_j) / tau))
        image = sum_k T_k * rgb_k          (rgb_k already carries alpha_k)
        w_k   = T_k * alpha_k              slot k's share of the pixel

    A faint slot hides little, however close it is, and there is no
    division by a slot's alpha and no log in the image. Returns (image
    (V, 3, H, W), weights (V, K, H, W), summing to the pixel's coverage).
    """
    rgb = torch.stack([layer["rgb"] for layer in layers], 1)  # (V, K, 3, H, W)
    alpha = torch.stack([layer["alpha"] for layer in layers], 1)  # (V, K, 1, H, W)
    acc_depth = torch.stack([layer["depth"] for layer in layers], 1)
    depth = (acc_depth + _ALPHA_EPS * far) / (alpha + _ALPHA_EPS)
    # [v, j, k] = d_k - d_j: j in front of k.
    front = torch.sigmoid((depth[:, None] - depth[:, :, None]) / tau)
    not_self = 1.0 - torch.eye(alpha.shape[1], device=alpha.device, dtype=alpha.dtype)
    front = front * not_self[None, :, :, None, None, None]
    transmittance = (1.0 - alpha[:, :, None] * front).prod(1)  # (V, K, 1, H, W)
    image = (transmittance * rgb).sum(1)
    weights = transmittance * alpha
    return image, weights[:, :, 0]


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
    ):
        """num_slots: K, the background slot included.
        bg_gaussians_per_slot: Gaussians of the background slot.
        scene_config: the SceneGenConfig dict from meta.json (default: the
        SceneGenConfig defaults); sets the object box and the background's
        floor and shell.
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
        self.decoder = AttentionDecoder(
            z_dim,
            gaussians_per_slot,
            decoder_hidden_dim,
            ObjectHead(z_dim, decoder_hidden_dim, bounds),
        )
        self.bg_decoder = AttentionDecoder(
            z_dim,
            bg_gaussians_per_slot,
            decoder_hidden_dim,
            BackgroundHead(decoder_hidden_dim, bounds, bg_gaussians_per_slot),
        )
        # Soft depth order of the slot layers: depth differences of a
        # fraction of an object decide who is in front; nearly transparent
        # parts count as this far.
        self.depth_tau = bounds.r_max / 4.0
        self.far = bounds.reach + float(scene_config["distance_range"][1])
        # Compactness: an object slot's Gaussians within this distance of
        # their centre cost nothing.
        self.object_radius = 1.5 * bounds.r_max

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
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

    def _capacity_terms(
        self, obj: dict[str, torch.Tensor], residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Object slots (K-1, G, ...): (compactness, appearance) terms, each
        weighted by the (detached) opacity of the Gaussians, so invisible
        Gaussians do not count and the terms cannot be met by fading."""
        w = obj["opacity"].detach()  # (K-1, G)
        mass = w.sum(1, keepdim=True).clamp_min(1e-8)
        centre = (w[..., None] * obj["mu"]).sum(1) / mass  # (K-1, 3)
        dist = ((obj["mu"] - centre[:, None]) ** 2).sum(-1).add(1e-12).sqrt()
        total = w.sum().clamp_min(1e-8)
        compactness = (w * F.relu(dist - self.object_radius) ** 2).sum() / total
        appearance = (w * residual.abs().mean(-1)).sum() / total
        return compactness, appearance

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """images: (N,3,H,W) in [0,1]; cam2world: (N,4,4) world frame.
        Only views[0] is encoded."""
        slots, attn = self.slot_attention(self.encode(images[0:1]))
        slots, attn = slots[0], attn[0]  # (K, C), (K, s*s)

        obj = self.decoder(slots[1:])  # (K-1, G, ...), view-0 azimuth frame
        residual = obj.pop("rgb_residual")
        compactness, appearance = self._capacity_terms(obj, residual)
        bg = self.bg_decoder(slots[:1])  # (1, G_bg, ...)
        yaw = view_yaw(cam2world[0])
        obj, bg = yaw_to_world(obj, yaw), yaw_to_world(bg, yaw)

        per_slot = [{name: bg[name][0] for name in bg}] + [
            {name: obj[name][k] for name in obj} for k in range(slots.shape[0] - 1)
        ]
        layers = [
            self.splat(g, cam2world, self.image_size, self.focal_ratio, self.near)
            for g in per_slot
        ]
        image, weights = composite(layers, self.depth_tau, self.far)

        gaussians = {name: torch.cat([g[name] for g in per_slot]) for name in bg}
        slot_ids = torch.cat(
            [
                torch.full((g["mu"].shape[0],), k, device=slots.device)
                for k, g in enumerate(per_slot)
            ]
        )
        return {
            "rgb": image,
            "slot_alpha": weights,  # (N, K, H, W), sums to the coverage
            "mask_idx": weights.argmax(dim=1),  # (N, H, W)
            "attn": attn,  # (K, s*s) view-0 slot attention; row 0 = background
            "slots": slots,  # (K, C) view-0 slots; 0 = background
            "slot_ids": slot_ids,  # (M,) slot of each Gaussian
            "gaussians": gaussians,  # world frame, background first
            "compactness": compactness,
            "appearance": appearance,
        }
