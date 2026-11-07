"""Slot model: view 0 -> slots -> Gaussians -> splatted into every view.

    encode / slot attention   view 0 -> background slot + K-1 object slots
    decode_objects            object slot -> region + Gaussians
    decode_background         background slot -> floor discs + fill colour
    render                    per-slot layers (scaled by draw factor), soft
                              depth order, over the background

Slot roles come from what each can draw. Decoders work in view 0's scene
frame (utils/geometry.py). Slot index 0 is the background, 1..K-1 objects.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders import BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.objects import OBJECT_RADIUS, OBJECT_REGION
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import SceneFrame, camera_directions, points_to_world, scene_frame, to_world
from gsod.utils.splatter import GsplatSplatter

# Transparent parts count as a layer at the far depth with this weight.
_ALPHA_EPS = 1e-3
# Anchors weight tokens by attn ** this (lean to the stronger core).
_ANCHOR_SHARPNESS = 2.0
# draw_factor: occupancy drawn fully; token count drawn at half.
FULL_OCCUPANCY = 0.8
COUNT_HALF = 1.0


def draw_factor(attn: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """attn (K, N) -> (K,) in [0, 1], how far each object slot is drawn:

        f = min(1, occupancy / FULL_OCCUPANCY)^power * count / (count + COUNT_HALF)
        occupancy = sum a^2 / sum a,  count = sum a

    A split object costs coverage, so one owner is preferred."""
    count = attn.sum(-1)
    occupancy = attn.pow(2).sum(-1) / count.clamp_min(1e-8)
    share = (occupancy / FULL_OCCUPANCY).clamp(1e-6, 1.0)
    return share.pow(power) * count / (count + COUNT_HALF)


_ANCHOR_HEIGHT = 0.5 * (OBJECT_RADIUS[0] + OBJECT_RADIUS[1])
_CORNERS = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]


def token_points(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(side*side, 3): each token's view-0 ray at mid object height."""
    h, w = image_size
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    directions = camera_directions(uu, vv, image_size, focal_ratio).reshape(-1, 3) @ frame.rot.T
    t = ((_ANCHOR_HEIGHT - frame.origin[2]) / directions[:, 2].clamp(max=-1e-6)).clamp(min=0.0)
    xy = (frame.origin[:2] + t[:, None] * directions[:, :2]).clamp(-OBJECT_REGION, OBJECT_REGION)
    return torch.cat([xy, torch.full_like(xy[:, :1], _ANCHOR_HEIGHT)], -1)


def attention_anchors(
    attn: torch.Tensor, points: torch.Tensor, colours: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (K, N), points, colours (N, 3) -> anchor, colour (K, 3)."""
    p = (_ANCHOR_SHARPNESS * attn.log()).softmax(dim=1)
    return p @ points, p @ colours


def composite(
    layers: list[dict[str, torch.Tensor]], tau: torch.Tensor, far: torch.Tensor, background: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slot layers in soft depth order over the background:

        T_k   = prod_{j != k} (1 - alpha_j * sigmoid((d_k - d_j) / tau))
        image = sum_k T_k * rgb_k + prod_k (1 - alpha_k) * background

    Returns image (V, 3, H, W), weights (V, 1 + K, H, W), contributions
    (V, K, 3, H, W).
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
    contributions = transmittance * rgb
    image = contributions.sum(1) + uncovered * background
    weights = torch.cat([uncovered, (transmittance * alpha)[:, :, 0]], 1)
    return image, weights, contributions


class GSODModel(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        input_size: int = 64,
        z_dim: int = 64,
        num_slots: int = 6,
        attn_iter: int = 3,
        gaussians_per_slot: int = 512,
        decoder_hidden_dim: int = 128,
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
    ):
        """num_slots: K, the background slot included."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.splat = GsplatSplatter()
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.focal_ratio = tuple(focal_ratio)
        self.near = near
        self.draw_power = 1.0

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, in_dim=z_dim, slot_dim=z_dim, iters=attn_iter
        )
        self.decoder = ObjectDecoder(z_dim, gaussians_per_slot, decoder_hidden_dim)
        self.background = BackgroundDecoder(z_dim, decoder_hidden_dim)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "Decoder": self.decoder,
            "BgDecoder": self.background,
        }

    # -- stages ---------------------------------------------------------------
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (N, 3, H, W) -> encoder tokens (N, s*s, C), s = input_size."""
        small = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return self.encoder(small).flatten(start_dim=2).permute(0, 2, 1)

    def decode_objects(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Object slots (K-1, C), attention (K-1, s*s), token colours (s*s, 3)
        -> world Gaussians (K-1, G, ...), region half sizes "extent" (K-1, 3)
        and world "corners" (K-1, 8, 3)."""
        points = token_points(frame, self.input_size, self.image_size, self.focal_ratio)
        anchor, colour = attention_anchors(attn, points, colours)
        ray = F.normalize(anchor - frame.origin, dim=-1)
        gaussians, centre, extent = self.decoder(slots, anchor=anchor, ray=ray, colour=colour)
        signs = torch.tensor(_CORNERS, dtype=centre.dtype, device=centre.device)
        corners = points_to_world(centre[:, None] + extent[:, None] * signs, frame)
        return {"gaussians": to_world(gaussians, frame), "extent": extent, "corners": corners}

    def decode_background(
        self, bg_slot: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Background slot (C,), attention (s*s,), token colours (s*s, 3) ->
        world floor discs (N, ...) and "fill" (3,)."""
        side = self.input_size
        gaussians = self.background(
            bg_slot, attn.reshape(side, side), colours.T.reshape(3, side, side), frame.yaw
        )
        fill = gaussians.pop("fill")
        return {"fill": fill, "gaussians": to_world(gaussians, frame, rotate=False)}

    def render(self, scene: dict, cam2world: torch.Tensor) -> dict[str, torch.Tensor]:
        """A scene (forward(render=False)) rendered into cam2world (N, 4, 4)."""
        obj = scene["objects"]["gaussians"]
        bg = scene["background"]
        draw, slots = scene["draw"], scene["slots"]
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        layers = []
        for k in range(slots.shape[0]):
            layer = self.splat({name: obj[name][k] for name in obj}, *cams)
            layers.append({name: draw[k] * value for name, value in layer.items()})
        floor = self.splat(bg["gaussians"], *cams)
        bg_rgb = floor["rgb"] + (1.0 - floor["alpha"]) * bg["fill"][None, :, None, None]
        frame = scene["frame"]
        tau = 0.25 * OBJECT_RADIUS[1] * frame.scale
        far = (frame.origin.norm() + 2.0 * OBJECT_REGION) * frame.scale
        image, weights, slot_rgb = composite(layers, tau, far, bg_rgb)

        n_gauss = obj["mu"].shape[1]
        return {
            "rgb": image,  # (N, 3, H, W)
            "fg_rgb": slot_rgb.sum(1),
            "slot_rgb": slot_rgb,  # (N, K-1, 3, H, W)
            "bg_rgb": bg_rgb,
            "slot_alpha": weights,  # (N, K, H, W), 0 = background
            "mask_idx": weights.argmax(dim=1),  # (N, H, W)
            "attn": scene["attn"],  # (K, s*s)
            "draw": draw,  # (K-1,)
            "extent": scene["objects"]["extent"],  # (K-1, 3)
            "corners": scene["objects"]["corners"],  # (K-1, 8, 3)
            "slots": slots,  # (K-1, C)
            "slot_ids": torch.arange(1, slots.shape[0] + 1, device=slots.device).repeat_interleave(n_gauss),
            "gaussians": {name: obj[name].flatten(0, 1) for name in obj},  # objects, world
        }

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor, render: bool = True
    ) -> dict:
        """images (N, 3, H, W) in [0, 1], cam2world (N, 4, 4) -> render
        outputs, or with render=False the decoded scene."""
        bg_slot, slots, attn = (x[0] for x in self.slot_attention(self.encode(images[0:1])))
        frame = scene_frame(cam2world[0], self.image_size, self.focal_ratio)
        colours = F.interpolate(images[0:1], size=self.input_size, mode="area")[0].flatten(1).T
        scene = {
            "objects": self.decode_objects(slots, attn[1:], colours, frame),
            "background": self.decode_background(bg_slot, attn[0], colours, frame),
            "draw": draw_factor(attn[1:], self.draw_power),
            "attn": attn,
            "slots": slots,
            "frame": frame,
        }
        return self.render(scene, cam2world) if render else scene
