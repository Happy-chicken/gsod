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

from gsod.model.decoders import AttnDecoder, BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.background import DETAIL_LIFT, DETAIL_TOKENS
from gsod.model.decoders.objects import OBJECT_RADIUS, OBJECT_REGION, TOKEN_QUERIES
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import SceneFrame, camera_directions, scene_frame, to_world, yaw_matrix
from gsod.utils.splatter import GsplatSplatter

# Anchors weight tokens by attn ** this (lean to the stronger core).
_ANCHOR_SHARPNESS = 2.0
# Draw factor: attention area (fraction of tokens) ramp.
AREA_EMPTY = 0.001
AREA_FULL = 0.004


def draw_factor(attn: torch.Tensor) -> torch.Tensor:
    """attn (K, N) -> (K,): 0 below AREA_EMPTY, 1 from AREA_FULL."""
    return ((attn.mean(-1) - AREA_EMPTY) / (AREA_FULL - AREA_EMPTY)).clamp(0.0, 1.0)


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


def token_uv(side: int, like: torch.Tensor) -> torch.Tensor:
    """(side*side, 2) token centres in [-1, 1]."""
    lin = (torch.arange(side, device=like.device, dtype=like.dtype) + 0.5) / side * 2.0 - 1.0
    vv, uu = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([uu.flatten(), vv.flatten()], -1)


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


@torch.no_grad()
def select_tokens(attn: torch.Tensor, n: int) -> torch.Tensor:
    """attn (K, N) -> (K-1, n): evenly spaced owned tokens, topped up by attention."""
    owner = attn.argmax(0)
    out = []
    for k in range(1, attn.shape[0]):
        a, owned = attn[k], owner == k
        count = int(owned.sum())
        if count >= n:
            step = torch.linspace(0, count - 1, n, device=a.device).long()
            picks = owned.nonzero()[:, 0][step]
        else:
            rest = torch.where(owned, torch.full_like(a, -1.0), a).topk(n - count).indices
            picks = torch.cat([owned.nonzero()[:, 0], rest])
        out.append(picks)
    return torch.stack(out)


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

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, in_dim=z_dim, slot_dim=z_dim, iters=attn_iter
        )
        self.obj_decoder = ObjectDecoder(z_dim, gaussians_per_slot, decoder_hidden_dim)
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim)
        self.attn_decoder = AttnDecoder(z_dim)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "BgDecoder": self.bg_decoder,
            "AttnDecoder": self.attn_decoder,
        }

    # -- stages ---------------------------------------------------------------
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (N, 3, H, W) -> encoder tokens (N, s*s, C), s = input_size."""
        small = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return self.encoder(small).flatten(start_dim=2).permute(0, 2, 1)

    def decode_objects(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
    ) -> dict[str, torch.Tensor]:
        """slots (K-1, C), attn (K, s*s), token colours (s*s, 3) -> Gaussians, extent."""
        side = self.input_size
        points = token_points(frame, side, self.image_size, self.focal_ratio)
        anchor, colour = attention_anchors(attn[1:], points, colours)
        ray = F.normalize(anchor - frame.origin, dim=-1)
        # Tokens each object slot owns -> token queries.
        uv = token_uv(side, attn)
        dirs = F.normalize(_token_rays(frame, side, self.image_size, self.focal_ratio), dim=-1)
        centre, extent = self.obj_decoder.region(slots, anchor, ray)
        idx = select_tokens(attn, TOKEN_QUERIES)  # (K-1, T)
        picked = torch.cat([uv[idx], dirs[idx]], -1)
        gaussians = self.obj_decoder(slots, centre, extent, colour, picked, frame.origin)
        return {"gaussians": to_world(gaussians, frame), "extent": extent}

    def decode_background(
        self, bg_slots: torch.Tensor, attn_all: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Background slots (1 + B, C), all slots' attention (K_all, s*s) (background
        first), token colours (s*s, 3) -> world floor Gaussians (N, ...), "fill"."""
        side, n_bg = self.input_size, bg_slots.shape[0]
        # Detail slots' most attended tokens (rays meeting the floor) -> floor points.
        rays = _token_rays(frame, side, self.image_size, self.focal_ratio)
        detail = attn_all[1:n_bg]
        idx = torch.where(rays[:, 2] < -1e-6, detail, -1.0).topk(DETAIL_TOKENS, -1).indices  # (B, T)
        floor = _at_height(frame, rays, DETAIL_LIFT) @ yaw_matrix(frame.yaw)[:2, :2].T.to(rays.dtype)
        gaussians = self.bg_decoder(
            bg_slots,
            attn_all[:n_bg].sum(0).reshape(side, side),
            colours.T.reshape(3, side, side),
            frame,
            floor[idx],
            token_uv(side, attn_all)[idx],
            detail.gather(1, idx),
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
            "bg_attn": scene["bg_attn"],  # (1 + B, s*s)
            "draw": draw,  # (K-1,)
            "extent": scene["objects"]["extent"],  # (K-1, 3)
        }

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor, render: bool = True) -> dict:
        """images (N, 3, H, W) in [0, 1], cam2world (N, 4, 4) -> render
        outputs, or with render=False the decoded scene."""
        tokens = self.encode(images[0:1])
        frame = scene_frame(cam2world[0], self.image_size, self.focal_ratio)
        bg_slots, slots, attn_all = (x[0] for x in self.slot_attention(tokens))
        n_bg = bg_slots.shape[0]
        attn = torch.cat([attn_all[:n_bg].sum(0, keepdim=True), attn_all[n_bg:]])  # row 0: background
        colours = F.interpolate(images[0:1], size=self.input_size, mode="area")[0].flatten(1).T
        scene = {
            "objects": self.decode_objects(slots, attn, colours, frame),
            "background": self.decode_background(bg_slots, attn_all, colours, frame),
            "draw": draw_factor(attn[1:]),
            "attn": attn,
            "bg_attn": attn_all[:n_bg],  # global, detail
            "slots": slots,
            "frame": frame,
        }
        if not render:
            return scene
        out = self.render(scene, cam2world)
        # Global 2D mixture of view 0 (competition among slots).
        bg = F.interpolate(out["bg_rgb"][0:1], size=self.input_size, mode="area")[0].flatten(1).T
        out["attn_rgb"] = self.attn_decoder(bg_slots, slots, bg, token_uv(self.input_size, attn))
        out["attn_target"] = colours
        return out
