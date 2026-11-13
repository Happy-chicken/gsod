"""Slot model: view 0 -> slots -> Gaussians -> splatted into every view.

    encode / slot attention   view 0 -> background slot + K-1 object slots
    decode_objects            object slot -> box at its attention anchor ->
                              Gaussians in it
    decode_background         background slot -> floor discs + fill colour
    render                    all Gaussians (object opacities x the slot's
                              draw factor, floor discs) sorted and blended
                              together; slot masks from the same blending

Decoders work in view 0's scene frame (utils/geometry.py). Slot 0 is the
background.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders import BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.objects import OBJECT_RADIUS, OBJECT_REGION
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import SceneFrame, camera_directions, scene_frame, to_world
from gsod.utils.splatter import GsplatSplatter

# Anchors weight tokens by attn ** this (lean to the stronger core).
_ANCHOR_SHARPNESS = 2.0
# Draw factor ramps: occupancy, attention area (fraction of tokens).
OCC_EMPTY = 0.4
FULL_OCCUPANCY = 0.8
AREA_EMPTY = 0.001
AREA_FULL = 0.004


def draw_factor(attn: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """attn (K, N) -> (K,): occupancy ramp^power * area ramp."""
    count = attn.sum(-1)
    occupancy = attn.pow(2).sum(-1) / count.clamp_min(1e-8)
    # >= 1e-6: power 0 draws every slot.
    share = ((occupancy - OCC_EMPTY) / (FULL_OCCUPANCY - OCC_EMPTY)).clamp(1e-6, 1.0)
    presence = ((count / attn.shape[-1] - AREA_EMPTY) / (AREA_FULL - AREA_EMPTY)).clamp(0.0, 1.0)
    return share.pow(power) * presence


_ANCHOR_HEIGHT = 0.5 * (OBJECT_RADIUS[0] + OBJECT_RADIUS[1])


def _token_rays(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(side*side, 3) scene-frame directions of the tokens' view-0 rays."""
    h, w = image_size
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return camera_directions(uu, vv, image_size, focal_ratio).reshape(-1, 3) @ frame.rot.T


def _at_height(frame: SceneFrame, directions: torch.Tensor, z: float) -> torch.Tensor:
    """Where the rays reach height z (N, 2); rays not going down stay at the camera."""
    t = ((z - frame.origin[2]) / directions[:, 2].clamp(max=-1e-6)).clamp(min=0.0)
    return frame.origin[:2] + t[:, None] * directions[:, :2]


def token_points(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(side*side, 3): each token's view-0 ray at mid object height."""
    xy = _at_height(frame, _token_rays(frame, side, image_size, focal_ratio), _ANCHOR_HEIGHT)
    xy = xy.clamp(-OBJECT_REGION, OBJECT_REGION)
    return torch.cat([xy, torch.full_like(xy[:, :1], _ANCHOR_HEIGHT)], -1)


def attention_anchors(
    attn: torch.Tensor, points: torch.Tensor, colours: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """attn (K, N), token points and colours (N, 3) -> where each slot's
    attention points and the colour there (K, 3)."""
    p = (_ANCHOR_SHARPNESS * attn.clamp_min(1e-8).log()).softmax(dim=1)
    return p @ points, p @ colours


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
        self.obj_decoder = ObjectDecoder(z_dim, gaussians_per_slot, decoder_hidden_dim)
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "BgDecoder": self.bg_decoder,
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
        -> world Gaussians (K-1, G, ...) and box half sizes "extent" (K-1, 3)."""
        side = self.input_size
        points = token_points(frame, side, self.image_size, self.focal_ratio)
        anchor, colour = attention_anchors(attn, points, colours)
        ray = F.normalize(anchor - frame.origin, dim=-1)
        gaussians, _, extent = self.obj_decoder(slots, anchor=anchor, ray=ray, colour=colour)
        return {"gaussians": to_world(gaussians, frame), "extent": extent}

    def decode_background(
        self, bg_slot: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Background slot (C,), attention (s*s,), token colours (s*s, 3) ->
        world floor discs (N, ...) and "fill" (3,)."""
        side = self.input_size
        gaussians = self.bg_decoder(
            bg_slot, attn.reshape(side, side), colours.T.reshape(3, side, side), frame
        )
        fill = gaussians.pop("fill")
        return {"fill": fill, "gaussians": to_world(gaussians, frame, rotate=False)}

    def render(self, scene: dict, cam2world: torch.Tensor) -> dict[str, torch.Tensor]:
        """A scene (forward(render=False)) rendered into cam2world (N, 4, 4):
        every Gaussian, object or floor, sorted and blended together."""
        obj, bg, draw = scene["objects"]["gaussians"], scene["background"], scene["draw"]
        n_obj, n_gauss = obj["mu"].shape[:2]
        floor = bg["gaussians"]
        objects = {name: value.flatten(0, 1) for name, value in obj.items()}
        objects["opacity"] = (obj["opacity"] * draw[:, None]).flatten()
        every = {name: torch.cat([objects[name], floor[name]]) for name in objects}
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        fill = bg["fill"][None, :, None, None]
        out = self.splat(every, *cams)
        image = out["rgb"] + (1.0 - out["alpha"]) * fill

        # Decomposition from the same blending (no gradient): per-slot weights
        # (0 = floor + fill) and per-object colour contributions.
        with torch.no_grad():
            every = {name: value.detach() for name, value in every.items()}
            ids = torch.cat([
                torch.arange(1, n_obj + 1, device=image.device).repeat_interleave(n_gauss),
                torch.zeros(floor["mu"].shape[0], dtype=torch.long, device=image.device),
            ])
            onehot = F.one_hot(ids, n_obj + 1).float()
            obj_rgb = (onehot[:, 1:, None] * every["rgb"][:, None]).flatten(1)
            parts = self.splat(every, *cams, features=torch.cat([onehot, obj_rgb], 1))["features"]
            weights = parts[:, : n_obj + 1].clone()
            weights[:, 0] += (1.0 - out["alpha"][:, 0]).detach()
            slot_rgb = parts[:, n_obj + 1 :].unflatten(1, (n_obj, 3))
            floor_only = self.splat({name: value.detach() for name, value in floor.items()}, *cams)
            bg_rgb = floor_only["rgb"] + (1.0 - floor_only["alpha"]) * fill.detach()

        return {
            "rgb": image,  # (N, 3, H, W)
            "fg_rgb": slot_rgb.sum(1),
            "slot_rgb": slot_rgb,  # (N, K-1, 3, H, W)
            "bg_rgb": bg_rgb,  # the floor alone
            "slot_alpha": weights,  # (N, K, H, W), 0 = background
            "mask_idx": weights.argmax(dim=1),  # (N, H, W)
            "attn": scene["attn"],  # (K, s*s)
            "draw": draw,  # (K-1,)
            "extent": scene["objects"]["extent"],  # (K-1, 3)
        }

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor, render: bool = True) -> dict:
        """images (N, 3, H, W) in [0, 1], cam2world (N, 4, 4) -> render
        outputs, or with render=False the decoded scene."""
        tokens = self.encode(images[0:1])
        frame = scene_frame(cam2world[0], self.image_size, self.focal_ratio)
        bg_slot, slots, attn = (x[0] for x in self.slot_attention(tokens))
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
