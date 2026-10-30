"""Slot model: encoder -> slot attention -> Gaussians per slot -> splatting.

views[0] -> Encoder -> tokens -> SlotAttention -> K-1 object slots
    object slots -> object decoder -> Gaussians per slot (view-0 azimuth
        frame -> world) -> each slot splatted on its own into every view ->
        alpha-composited as layers in a soft depth order, over the
        background colour

The background is empty: there is no background slot. Slot attention has
an empty option (row 0 of the attention, no parameters) that takes the
tokens no object slot wants, and whatever no object covers shows the
background colour (the dataset's ``background_color``, black if absent).
Nothing has to learn to draw it.

The slots compete in the 3D render itself, with no 2D decoder and no masks:

  * one share of each pixel: the slots' compositing weights w_k(p) sum to
    the pixel's coverage (``composite``), so a slot gains a pixel only from
    the slots it hides or that hide it, and the reconstruction gradient
    moves the pixel to the slot whose colour fits; w_k is also slot k's
    mask, in every view.
  * input-dependent from the first step: each object slot's Gaussians sit
    in the region its attention covers (attention-weighted centre and spread
    of the points where the token rays reach object height) and start with
    that region's mean colour (``attention_regions``). The reconstruction
    gradient reaches the attention through them; while the attention is
    still spread, so are the Gaussians.
  * limited capacity: an object slot has one base colour (per-Gaussian
    residuals are penalised) and is pulled compact (compactness term, from
    train.py's schedule), so one slot is a poor fit for two objects.

Slot indices in every output (masks, slot_alpha, attention rows):
    0       background (in slot_alpha: the uncovered share of the pixel)
    1..K-1  objects

Only views[0] is encoded, in training as at inference.

Frames
------
Only views[0] is decoded into 3D and the camera azimuth is uniform over 360
deg, so the image alone does not say where things are in *world* xy.
Gaussians are therefore predicted in the azimuth frame of view 0 (yaw-rotated
so view 0 looks along +x; uORF's nss2cam0) and rotated into the world before
splatting. Cameras are never modified.

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from gsod.data_generator.generator import SceneGenConfig
from gsod.model.decoders import AttentionDecoder, ObjectHead, SceneBounds
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


def token_points(
    cam2world0: torch.Tensor,
    side: int,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
    height: float,
    xy_bound: float,
) -> torch.Tensor:
    """(side*side, 3), row-major: where the view-0 ray through the centre of
    each encoder token cell reaches z = height, in the view-0 azimuth frame,
    with xy clamped to [-xy_bound, xy_bound] (rays at or above the horizon
    end on the bound)."""
    h, w = image_size
    c2w = cam2world0
    idx = torch.arange(side, device=c2w.device, dtype=c2w.dtype)
    # Cell centres in image pixel coordinates (pixel centres at integers,
    # matching the dataset and the bilinear resize into the encoder).
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    fx, fy = focal_ratio[0] * w, focal_ratio[1] * h
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    d_cam = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], -1)
    to_azimuth = _yaw_matrix(-view_yaw(c2w)).to(c2w.dtype)
    origin = to_azimuth @ c2w[:3, 3]
    directions = d_cam.reshape(-1, 3) @ (to_azimuth @ c2w[:3, :3]).T
    t = ((height - origin[2]) / directions[:, 2].clamp(max=-1e-6)).clamp(min=0.0)
    xy = (origin[:2] + t[:, None] * directions[:, :2]).clamp(-xy_bound, xy_bound)
    return torch.cat([xy, torch.full_like(xy[:, :1], height)], -1)


def attention_regions(
    attn: torch.Tensor,
    points: torch.Tensor,
    colours: torch.Tensor,
    extent_range: tuple[float, float],
    half_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each slot's attention region: attn (K, N) over N tokens, their points
    (N, 3) and colours (N, 3) -> centre (K, 3) and mean colour (K, 3), the
    attention-weighted means, and extent (K, 3): twice the weighted spread
    in xy, within extent_range, and half_height in z."""
    p = attn / attn.sum(1, keepdim=True)
    centre = p @ points
    spread = (p @ points[:, :2] ** 2 - centre[:, :2] ** 2).clamp_min(0.0).add(1e-8).sqrt()
    extent_xy = (2.0 * spread).clamp(*extent_range)
    extent = torch.cat([extent_xy, torch.full_like(extent_xy[:, :1], half_height)], -1)
    return centre, extent, p @ colours


def composite(
    layers: list[dict[str, torch.Tensor]],
    tau: float,
    far: float,
    background: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alpha compositing of the object slots as layers, in a soft depth
    order, over a background colour (3,) behind all of them.

    layers: per slot, the splatter output (V, C, H, W): accumulated "rgb"
    and "depth", and "alpha". Slot k sits at its expected depth d_k (pulled
    to the far depth where it is nearly transparent). Slot j is in front of
    k with probability sigmoid((d_k - d_j) / tau) and then hides alpha_j of
    it:

        T_k   = prod_{j != k} (1 - alpha_j * sigmoid((d_k - d_j) / tau))
        T_bg  = prod_k (1 - alpha_k)
        image = sum_k T_k * rgb_k + T_bg * background
                                           (rgb_k already carries alpha_k)
        w_k   = T_k * alpha_k              slot k's share of the pixel

    A faint slot hides little, however close it is, and there is no
    division by a slot's alpha and no log in the image. Returns (image
    (V, 3, H, W), weights (V, 1 + K, H, W): the background's share T_bg
    first, then the slots').
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
    uncovered = (1.0 - alpha).prod(1)  # (V, 1, H, W)
    image = (transmittance * rgb).sum(1) + uncovered * background[:, None, None]
    weights = torch.cat([uncovered, (transmittance * alpha)[:, :, 0]], 1)
    return image, weights


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
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        scene_config: dict | None = None,
    ):
        """num_slots: K, the empty background included (K-1 object slots).
        scene_config: the SceneGenConfig dict from meta.json (default: the
        SceneGenConfig defaults); sets the object box and the background
        colour (``background_color``, black if absent).
        Slots are initialised BO-QSA style (see slot_attention.py)."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the empty background; need >= 2"
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
        self.bounds = bounds
        self.decoder = AttentionDecoder(
            z_dim,
            gaussians_per_slot,
            decoder_hidden_dim,
            ObjectHead(z_dim, decoder_hidden_dim, bounds),
        )
        self.register_buffer(
            "background",
            torch.tensor(scene_config.get("background_color", (0.0, 0.0, 0.0))),
            persistent=False,
        )
        # Soft depth order of the slot layers: depth differences of a
        # fraction of an object decide who is in front; nearly transparent
        # parts count as this far.
        self.depth_tau = bounds.r_max / 4.0
        self.far = bounds.d_max + 2.0 * bounds.fg_xy
        # Compactness: an object slot's Gaussians within this distance of
        # their centre cost nothing.
        self.object_radius = 1.5 * bounds.r_max

    def networks(self) -> dict[str, nn.Module]:
        """Named networks for checkpoint.save_networks / load_networks."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
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
        slots, attn = slots[0], attn[0]  # (K-1, C) object slots, (K, s*s)

        # Object slots start from their attention regions: token rays at the
        # height of an average object's centre, and the token colours.
        b, side = self.bounds, self.input_size
        mid = 0.5 * (b.r_min + b.r_max)
        points = token_points(
            cam2world[0], side, self.image_size, self.focal_ratio, b.floor_z + mid, b.fg_xy
        )
        colours = F.interpolate(images[0:1], size=side, mode="area")[0].flatten(1).T
        centre, extent, colour = attention_regions(
            attn[1:], points, colours, (b.r_max, b.fg_xy), b.r_max + 0.125
        )
        # (K-1, G, ...), view-0 azimuth frame
        obj = self.decoder(slots, centre=centre, extent=extent, colour=colour)
        residual = obj.pop("rgb_residual")
        compactness, appearance = self._capacity_terms(obj, residual)
        obj = yaw_to_world(obj, view_yaw(cam2world[0]))

        per_slot = [{name: obj[name][k] for name in obj} for k in range(slots.shape[0])]
        layers = [
            self.splat(g, cam2world, self.image_size, self.focal_ratio, self.near)
            for g in per_slot
        ]
        image, weights = composite(layers, self.depth_tau, self.far, self.background)

        gaussians = {name: obj[name].flatten(0, 1) for name in obj}
        n_gauss = obj["mu"].shape[1]
        slot_ids = torch.arange(1, slots.shape[0] + 1, device=slots.device).repeat_interleave(
            n_gauss
        )
        return {
            "rgb": image,
            "slot_alpha": weights,  # (N, K, H, W); 0 = uncovered (background)
            "mask_idx": weights.argmax(dim=1),  # (N, H, W)
            "attn": attn,  # (K, s*s) view-0 slot attention; row 0 = empty
            "slots": slots,  # (K-1, C) view-0 object slots (masks 1..K-1)
            "slot_ids": slot_ids,  # (M,) slot of each Gaussian (1..K-1)
            "gaussians": gaussians,  # world frame, object slots only
            "compactness": compactness,
            "appearance": appearance,
        }
