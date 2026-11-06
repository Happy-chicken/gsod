"""Slot model: view 0 -> slots -> Gaussians -> splatting into every view.

    encode            view 0 -> tokens
    slot attention    tokens -> background slot + K-1 object slots
    decode_objects    each object slot -> its region and Gaussians in it
    decode_background background slot -> floor discs + fill colour
    render            every view: one layer per object slot (scaled by its
                      draw factor), soft depth order, over the background

The slots share every token; their roles come from what each can draw: an
object slot an object-sized region, the background a layer behind every
object, on the floor plane. The compositing weights w_k sum to the pixel's
coverage, so slots compete for pixels in the render, and w_k is slot k's
mask in every view.

Both decoders work in the scene frame, set by view 0's camera
(utils/geometry.py: lengths in units of R); their Gaussians are mapped into
the world before splatting.

Slot indices in every output: 0 background (in slot_alpha: the uncovered
share of the pixel), 1..K-1 objects. Only view 0 is encoded.

pixel_features: both decoders also read view 0's encoder tokens, each
through its own slot's attention (decoders/objects.py, background.py), for
the thin parts and textures a slot latent cannot hold.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders import BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.objects import OBJECT_RADIUS, OBJECT_REGION
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import (
    SceneFrame,
    camera_directions,
    points_to_world,
    scene_frame,
    to_world,
    yaw_matrix,
)
from gsod.utils.splatter import GsplatSplatter

# A slot's transparent part counts as this much of a layer at the far depth.
_ALPHA_EPS = 1e-3
# Anchors weight tokens by attn ** this, so a slot with two cores leans to
# the stronger one instead of the mean of two objects.
_ANCHOR_SHARPNESS = 2.0
# draw_factor: occupancy at which a slot is drawn fully (below 1: an
# object's rim tokens are shared, so an owner reaches ~0.83).
FULL_OCCUPANCY = 0.8
# draw_factor: token count drawn at half (one token is not an object; a
# small CLEVR object, ~11 tokens, keeps 0.92).
COUNT_HALF = 1.0


def draw_factor(attn: torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """How far each object slot is drawn: attn (K, N) -> (K,) in [0, 1].

        f = min(1, occupancy / FULL_OCCUPANCY)^power * count / (count + COUNT_HALF)
        occupancy = sum a^2 / sum a     mean share of its tokens
        count     = sum a               tokens held

    Linear in the share, so a split costs coverage: two slots at 0.47 / 0.47
    on one object get 0.58 each and cover 0.82 of it, and the reconstruction
    prefers one owner; a slot spread at 0.15 over the floor gets 0.19.
    power < 1 (0 early in training, train.py) spares slots still finding
    an object; at 1 a split costs most. GSODModel scales the slot's whole
    layer by f."""
    count = attn.sum(-1)
    occupancy = attn.pow(2).sum(-1) / count.clamp_min(1e-8)
    share = (occupancy / FULL_OCCUPANCY).clamp(1e-6, 1.0)
    return share.pow(power) * count / (count + COUNT_HALF)


# Anchors sit at mid object height (units of R).
_ANCHOR_HEIGHT = 0.5 * (OBJECT_RADIUS[0] + OBJECT_RADIUS[1])
# Box corners as signs of the half sizes.
_CORNERS = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]


def token_points(
    frame: SceneFrame, side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(side*side, 3): where view 0's ray through each token cell's centre
    reaches mid object height (scene frame), xy clamped to the object region."""
    h, w = image_size
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    # Cell centres in pixels (pixel centres at integers, as in the dataset).
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
    """Where each slot's attention points and the colour there: attn (K, N),
    token points and colours (N, 3) -> (K, 3) each, means weighted by the
    sharpened attention."""
    p = (_ANCHOR_SHARPNESS * attn.log()).softmax(dim=1)
    return p @ points, p @ colours


def composite(
    layers: list[dict[str, torch.Tensor]], tau: torch.Tensor, far: torch.Tensor, background: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Object-slot layers in a soft depth order over the background image.

    layers: per slot, the splatter output (V, C, H, W) ("rgb", "depth"
    premultiplied by "alpha"); d_k is slot k's expected depth.

        T_k   = prod_{j != k} (1 - alpha_j * sigmoid((d_k - d_j) / tau))
        T_bg  = prod_k (1 - alpha_k)
        image = sum_k T_k * rgb_k + T_bg * background
        w_k   = T_k * alpha_k              slot k's share of the pixel

    Returns image (V, 3, H, W), weights (V, 1 + K, H, W) (T_bg first) and
    contributions T_k * rgb_k (V, K, 3, H, W).
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
        pixel_features: bool = False,
    ):
        """num_slots: K, the background slot included."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.splat = GsplatSplatter()
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.focal_ratio = tuple(focal_ratio)
        self.near = near
        # Exponent of the draw factor's share term (train.py ramps it to 1).
        self.draw_power = 1.0

        self.encoder = Encoder(3, z_dim=z_dim)
        self.slot_attention = SlotAttention(
            num_slots=num_slots, in_dim=z_dim, slot_dim=z_dim, iters=attn_iter
        )
        self.pixel_features = bool(pixel_features)
        self.decoder = ObjectDecoder(
            z_dim, gaussians_per_slot, decoder_hidden_dim, pixel_features=self.pixel_features
        )
        self.background = BackgroundDecoder(
            z_dim, decoder_hidden_dim, pixel_features=self.pixel_features
        )

    def networks(self) -> dict[str, nn.Module]:
        """Named networks"""
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
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
        tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Object slots (K-1, C), their attention (K-1, s*s), token colours
        (s*s, 3), encoder tokens (s*s, C; read with pixel_features) ->
        Gaussians (K-1, G, ...) in the world, with each region's half sizes
        "extent" (K-1, 3, units of R) and world "corners" (K-1, 8, 3)."""
        points = token_points(frame, self.input_size, self.image_size, self.focal_ratio)
        anchor, colour = attention_anchors(attn, points, colours)
        ray = F.normalize(anchor - frame.origin, dim=-1)  # view 0's ray through each anchor
        pixels = None
        if self.pixel_features:
            pixels = {"features": tokens, "colours": colours, "points": points, "attn": attn}
        gaussians, centre, extent = self.decoder(
            slots, anchor=anchor, ray=ray, colour=colour, pixels=pixels
        )
        signs = torch.tensor(_CORNERS, dtype=centre.dtype, device=centre.device)
        corners = points_to_world(centre[:, None] + extent[:, None] * signs, frame)
        return {"gaussians": to_world(gaussians, frame), "extent": extent, "corners": corners}

    def floor_pixels(
        self, maps: torch.Tensor, attn: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Where each floor disc projects in view 0: maps (C + 3, s, s)
        (encoder features, colours), the background's attention (s, s) ->
        per disc "features" (N, C), "colours" (N, 3) sampled there (bilinear)
        and "w" (N,) = attention there (detached) * in view."""
        mu = self.background.floor_mu  # world-aligned scene frame
        to_frame = yaw_matrix(-frame.yaw).to(mu.dtype)
        cam = (mu @ to_frame.T - frame.origin) @ frame.rot  # view 0's camera coordinates
        z = cam[:, 2]
        front = z > 1e-3
        z = torch.where(front, z, torch.ones_like(z))
        # grid_sample coordinates (align_corners=False): pixel centres at
        # integers, the principal point at the image centre.
        grid = torch.stack(
            [2.0 * self.focal_ratio[0] * cam[:, 0] / z, 2.0 * self.focal_ratio[1] * cam[:, 1] / z], -1
        )
        inside = front & (grid.abs() <= 1.0).all(-1)
        grid = torch.where(inside[:, None], grid, torch.zeros_like(grid))
        stacked = torch.cat([maps, attn.detach()[None]], 0)
        sampled = F.grid_sample(
            stacked[None], grid[None, None], mode="bilinear", align_corners=False
        )[0, :, 0].T  # (N, C + 4)
        c = maps.shape[0] - 3
        return {
            "features": sampled[:, :c],
            "colours": sampled[:, c : c + 3],
            "w": sampled[:, -1] * inside.to(sampled.dtype),
        }

    def decode_background(
        self,
        bg_slot: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
        tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Background slot (C,), its attention (s*s,), token colours (s*s, 3),
        encoder tokens (s*s, C; read with pixel_features) -> floor discs
        (N, ...) in the world and the "fill" colour (3,)."""
        side = self.input_size
        attn_map = attn.reshape(side, side)
        colour_map = colours.T.reshape(3, side, side)
        pixels = None
        if self.pixel_features:
            maps = torch.cat([tokens.T.reshape(-1, side, side), colour_map], 0)
            pixels = self.floor_pixels(maps, attn_map, frame)
        gaussians = self.background(bg_slot, attn_map, colour_map, frame.yaw, pixels=pixels)
        fill = gaussians.pop("fill")
        return {"fill": fill, "gaussians": to_world(gaussians, frame, rotate=False)}

    def render(self, scene: dict, cam2world: torch.Tensor) -> dict[str, torch.Tensor]:
        """A scene (forward with render=False) rendered into every view of
        cam2world (N, 4, 4)."""
        obj = scene["objects"]["gaussians"]
        bg = scene["background"]
        draw, slots = scene["draw"], scene["slots"]
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        # Each slot's layer scaled as a whole (rgb and depth are premultiplied,
        # so its expected depth is unchanged): its alpha is at most draw[k].
        layers = []
        for k in range(slots.shape[0]):
            layer = self.splat({name: obj[name][k] for name in obj}, *cams)
            layers.append({name: draw[k] * value for name, value in layer.items()})
        # The floor, over the fill colour where it does not cover the view.
        floor = self.splat(bg["gaussians"], *cams)
        bg_rgb = floor["rgb"] + (1.0 - floor["alpha"]) * bg["fill"][None, :, None, None]
        # Soft depth order: a fraction of an object decides who is in front;
        # nearly transparent parts count as past every object.
        frame = scene["frame"]
        tau = 0.25 * OBJECT_RADIUS[1] * frame.scale
        far = (frame.origin.norm() + 2.0 * OBJECT_REGION) * frame.scale
        image, weights, slot_rgb = composite(layers, tau, far, bg_rgb)

        n_gauss = obj["mu"].shape[1]
        return {
            "rgb": image,  # (N, 3, H, W)
            "fg_rgb": slot_rgb.sum(1),  # the object slots alone, over black
            "slot_rgb": slot_rgb,  # (N, K-1, 3, H, W) each slot's contribution
            "bg_rgb": bg_rgb,  # the background alone
            "slot_alpha": weights,  # (N, K, H, W); 0 = uncovered (background)
            "mask_idx": weights.argmax(dim=1),  # (N, H, W)
            "attn": scene["attn"],  # (K, s*s) view-0 slot attention; row 0 = background
            "draw": draw,  # (K-1,) the object slots' draw factors
            "extent": scene["objects"]["extent"],  # (K-1, 3) region half sizes, units of R
            "corners": scene["objects"]["corners"],  # (K-1, 8, 3) region corners, world
            "slots": slots,  # (K-1, C) object slots (masks 1..K-1)
            "slot_ids": torch.arange(1, slots.shape[0] + 1, device=slots.device).repeat_interleave(n_gauss),
            "gaussians": {name: obj[name].flatten(0, 1) for name in obj},  # world, objects only
        }

    def forward(
        self, images: torch.Tensor, cam2world: torch.Tensor, render: bool = True
    ) -> dict:
        """images (N, 3, H, W) in [0, 1], cam2world (N, 4, 4): the scene from
        view 0 rendered into every view, or with render=False the scene
        itself: objects and background (decode_objects / decode_background),
        draw (K-1,) the object slots' draw factors, attn (K, s*s), slots
        (K-1, C), frame (view 0's SceneFrame)."""
        tokens = self.encode(images[0:1])
        bg_slot, slots, attn = (x[0] for x in self.slot_attention(tokens))
        tokens = tokens[0]  # (s*s, C)
        frame = scene_frame(cam2world[0], self.image_size, self.focal_ratio)
        colours = F.interpolate(images[0:1], size=self.input_size, mode="area")[0].flatten(1).T
        scene = {
            "objects": self.decode_objects(slots, attn[1:], colours, frame, tokens),
            "background": self.decode_background(bg_slot, attn[0], colours, frame, tokens),
            "draw": draw_factor(attn[1:], self.draw_power),
            "attn": attn,
            "slots": slots,
            "frame": frame,
        }
        return self.render(scene, cam2world) if render else scene
