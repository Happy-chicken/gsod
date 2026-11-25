"""Slot model: view 0 -> slots -> Gaussians -> splatted into every view; a
batch of scenes at once.

    encode / slot attention   view 0 -> background slot + K-1 object slots
    decode_objects            object slot -> box on the floor at its anchor ->
                              Gaussians in it, fainter as its attention shrinks
    decode_background         background slot + depth from the encoder -> Gaussians on view 0's rays
    refine                    objects decoded on from each Gaussian's view-0 pixel, where its slot owns it
    render                    all Gaussians blended together; masks from the
                              same blending

Decoders work in view 0's scene frame (utils/geometry.py). Slot 0 is the
background.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.decoders import BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.background import DepthDecoder
from gsod.model.decoders.objects import box_corners
from gsod.model.encoder import Encoder
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import SceneFrame, points_to_world, scene_frame, to_world
from gsod.utils.splatter import splat


class GSODModel(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        input_size: int,
        z_dim: int,
        num_slots: int,
        attn_iter: int,
        gaussians_per_slot: int,
        decoder_hidden_dim: int,
        focal_ratio: tuple[float, float],
        near: float,
        refine_steps: int = 1,
    ):
        """num_slots: K, the background slot included."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.focal_ratio = tuple(focal_ratio)
        self.near = near

        self.encoder = Encoder(z_dim)
        self.slot_attention = SlotAttention(num_slots, z_dim, attn_iter)
        self.obj_decoder = ObjectDecoder(z_dim, gaussians_per_slot, decoder_hidden_dim, image_size, focal_ratio)
        self.depth_decoder = DepthDecoder(z_dim)
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim, image_size, focal_ratio)
        self.refine_steps = refine_steps

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "DepthDecoder": self.depth_decoder,
            "BgDecoder": self.bg_decoder,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) -> encoder tokens (B, s*s, C), s = input_size."""
        small = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return self.encoder(small).flatten(start_dim=2).permute(0, 2, 1)

    def decode_objects(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        """slots (B, K-1, C), their attention (B, K-1, s*s), token colours (B, s*s, 3)
        -> world Gaussians (B, K-1, G, ...), the box {"extent" (B, K-1, 3, scene
        frame), "box" (B, K-1, 8, 3) world corners, "anchor_init" (B, K-1, 3) world},
        the decoder state for refine_objects."""
        local, box, state = self.obj_decoder(slots, attn, colours, frame)
        box = {
            "extent": box["extent"],
            "box": points_to_world(box_corners(box["anchor"], box["extent"]), frame),
            "anchor_init": points_to_world(box["anchor_init"], frame),
        }
        return to_world(local, frame), box, state

    def decode_background(
        self,
        bg_slot: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
        enc: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Background slot (B, C), its attention (B, s*s), token colours (B, s*s, 3), encoder map
        (B, C, s, s) -> world background Gaussians (B, M, ...), depth share map (B, 1, s, s)."""
        depth = self.depth_decoder(enc, attn.unflatten(1, enc.shape[-2:])[:, None])
        return to_world(self.bg_decoder(bg_slot, attn, colours, frame, depth), frame), depth

    def refine_objects(
        self,
        state: dict,
        image: torch.Tensor,
        enc: torch.Tensor,
        attn: torch.Tensor,
        frame: SceneFrame,
        detach: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """One refine round -> world object Gaussians, new state."""
        if detach:
            state = {**state, "q": state["q"].detach()}
        local, state = self.obj_decoder.refine(state, image, enc, attn, frame)
        return to_world(local, frame), state

    @staticmethod
    def merge(objects: dict[str, torch.Tensor], background: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Object (B, K-1, G, ...) and background (B, M, ...) Gaussians -> every Gaussian (B, N, ...), objects first."""
        return {name: torch.cat([objects[name].flatten(1, 2), background[name]], 1) for name in objects}

    def render(
        self, objects: dict[str, torch.Tensor], background: dict[str, torch.Tensor], cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """World object and background Gaussians rendered into cam2world (B, V, 4, 4):
        all sorted and blended together; per-slot masks from the same blending."""
        b, n_obj, n_gauss = objects["mu"].shape[:3]
        every = self.merge(objects, background)
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        out = splat(every, *cams)
        image = out["rgb"]

        # Decomposition from the same blending (no gradient): per-slot weights
        # (0 = background and uncovered) and per-object colour contributions.
        with torch.no_grad():
            every = {name: value.detach() for name, value in every.items()}
            ids = torch.cat(
                [
                    torch.arange(1, n_obj + 1, device=image.device).repeat_interleave(n_gauss),
                    torch.zeros(background["mu"].shape[1], dtype=torch.long, device=image.device),
                ]
            )
            onehot = F.one_hot(ids, n_obj + 1).float().expand(b, -1, -1)  # (B, N, K)
            obj_rgb = (onehot[..., 1:, None] * every["rgb"][..., None, :]).flatten(-2)
            parts = splat(every, *cams, features=torch.cat([onehot, obj_rgb], -1))["features"]
            weights = parts[:, :, : n_obj + 1].clone()
            weights[:, :, 0] += (1.0 - out["alpha"][:, :, 0]).detach()
            slot_rgb = parts[:, :, n_obj + 1 :].unflatten(2, (n_obj, 3))
            bg_rgb = splat({name: value.detach() for name, value in background.items()}, *cams)["rgb"]

        return {
            "rgb": image,  # (B, V, 3, H, W)
            "fg_rgb": slot_rgb.sum(2),
            "slot_rgb": slot_rgb,  # (B, V, K-1, 3, H, W)
            "bg_rgb": bg_rgb,  # the background alone
            "slot_alpha": weights,  # (B, V, K, H, W), 0 = background
            "mask_idx": weights.argmax(dim=2),  # (B, V, H, W)
        }

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor) -> dict:
        """images (B, V, 3, H, W) in [0, 1], cam2world (B, V, 4, 4) -> render
        outputs (B, ...), attention and boxes."""
        tokens = self.encode(images[:, 0])
        frame = scene_frame(cam2world[:, 0], self.image_size, self.focal_ratio)
        bg_slot, slots, attn = self.slot_attention(tokens)
        small = F.interpolate(images[:, 0], size=self.input_size, mode="area")  # (B, 3, s, s)
        colours = small.flatten(2).transpose(1, 2)
        side = (self.input_size, self.input_size)
        enc = tokens.transpose(1, 2).unflatten(2, side)  # (B, C, s, s)
        background, bg_depth = self.decode_background(bg_slot, attn[:, 0], colours, frame, enc)
        objects, box, state = self.decode_objects(slots, attn[:, 1:], colours, frame)  # the sketch

        # Refine: each object Gaussian reads its view-0 pixel where its slot owns it.
        sketch = objects
        obj_attn = attn[:, 1:].unflatten(2, side)  # (B, K-1, s, s)
        for i in range(self.refine_steps):
            objects, state = self.refine_objects(state, images[:, 0], enc, obj_attn, frame, detach=i > 0)

        out = self.render(objects, background, cam2world)
        out.update(
            attn=attn, bg_depth=bg_depth, **box
        )  # attn (B, K, s*s); bg_depth (B, 1, s, s); extent, box, anchor_init
        if self.refine_steps:  # before refinement: auxiliary loss, panels
            out["rgb_pre"] = splat(
                self.merge(sketch, background), cam2world, self.image_size, self.focal_ratio, self.near
            )["rgb"]
        return out
