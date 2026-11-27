"""Slot model: view 0 -> slots -> Gaussians -> splatted into every view; a
batch of scenes at once.

    encode / slot attention   view 0 -> background slot + K-1 object slots
    decode_objects            object slot -> box on the floor at its anchor ->
                              Gaussians in it, fainter as its attention shrinks
    decode_background         background slot -> flat Gaussians on the floor
    refine                    objects decoded on once from points around each Gaussian's view-0 pixel
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
from gsod.model.decoders.objects import box_corners
from gsod.model.encoders import DinoEncoder, UNetEncoder
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
    ):
        """num_slots: K, the background slot included."""
        super().__init__()
        assert num_slots >= 2, "num_slots counts the background; need >= 2"
        self.image_size = tuple(image_size)
        self.input_size = input_size
        self.focal_ratio = tuple(focal_ratio)
        self.near = near

        self.encoder = DinoEncoder(z_dim, input_size)  # for slot attention
        self.refine_encoder = UNetEncoder(z_dim)  # pixel features read by refine
        self.slot_attention = SlotAttention(num_slots, z_dim, attn_iter)
        self.obj_decoder = ObjectDecoder(z_dim, gaussians_per_slot, decoder_hidden_dim, image_size, focal_ratio)
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "RefineEncoder": self.refine_encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "BgDecoder": self.bg_decoder,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) -> encoder tokens (B, s*s, C), s = input_size."""
        return self.encoder(images).flatten(start_dim=2).permute(0, 2, 1)

    def decode_objects(
        self, slots: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
        """slots (B, K-1, C), their attention (B, K-1, s*s), token colours (B, s*s, 3)
        -> world Gaussians (B, K-1, G, ...), the box {"extent" (B, K-1, 3, scene
        frame), "box" (B, K-1, 8, 3) world corners, "anchor_init" (B, K-1, 3) world},
        the decoder state for the refine."""
        local, box, state = self.obj_decoder(slots, attn, colours, frame)
        box = {
            "extent": box["extent"],
            "box": points_to_world(box_corners(box["anchor"], box["extent"]), frame),
            "anchor_init": points_to_world(box["anchor_init"], frame),
        }
        return to_world(local, frame), box, state

    def decode_background(
        self, bg_slot: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Background slots (B, C), attention (B, s*s), token colours
        (B, s*s, 3) -> world floor Gaussians (B, M, ...)."""
        return to_world(self.bg_decoder(bg_slot, attn, colours, frame), frame, rotate=False)

    @staticmethod
    def merge(objects: dict[str, torch.Tensor], floor: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Object (B, K-1, G, ...) and floor (B, M, ...) Gaussians -> every Gaussian (B, N, ...), objects first."""
        return {name: torch.cat([objects[name].flatten(1, 2), floor[name]], 1) for name in objects}

    def render(
        self, objects: dict[str, torch.Tensor], floor: dict[str, torch.Tensor], cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """World object and floor Gaussians rendered into cam2world (B, V, 4, 4):
        all sorted and blended together; per-slot masks from the same blending."""
        b, n_obj, n_gauss = objects["mu"].shape[:3]
        every = self.merge(objects, floor)
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        out = splat(every, *cams)
        image = out["rgb"]

        # Decomposition from the same blending (no gradient): per-slot weights
        # (0 = floor and uncovered) and per-object colour contributions.
        with torch.no_grad():
            every = {name: value.detach() for name, value in every.items()}
            ids = torch.cat(
                [
                    torch.arange(1, n_obj + 1, device=image.device).repeat_interleave(n_gauss),
                    torch.zeros(floor["mu"].shape[1], dtype=torch.long, device=image.device),
                ]
            )
            onehot = F.one_hot(ids, n_obj + 1).float().expand(b, -1, -1)  # (B, N, K)
            obj_rgb = (onehot[..., 1:, None] * every["rgb"][..., None, :]).flatten(-2)
            parts = splat(every, *cams, features=torch.cat([onehot, obj_rgb], -1))["features"]
            weights = parts[:, :, : n_obj + 1].clone()
            weights[:, :, 0] += (1.0 - out["alpha"][:, :, 0]).detach()
            slot_rgb = parts[:, :, n_obj + 1 :].unflatten(2, (n_obj, 3))
            bg_rgb = splat({name: value.detach() for name, value in floor.items()}, *cams)["rgb"]

        return {
            "rgb": image,  # (B, V, 3, H, W)
            "fg_rgb": slot_rgb.sum(2),
            "slot_rgb": slot_rgb,  # (B, V, K-1, 3, H, W)
            "bg_rgb": bg_rgb,  # the floor alone
            "slot_alpha": weights,  # (B, V, K, H, W), 0 = background
            "mask_idx": weights.argmax(dim=2),  # (B, V, H, W)
        }

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor) -> dict:
        """images (B, V, 3, H, W) in [0, 1], cam2world (B, V, 4, 4) -> render
        outputs (B, ...), attention and boxes."""
        tokens = self.encode(images[:, 0])  # DINO
        frame = scene_frame(cam2world[:, 0], self.image_size, self.focal_ratio)
        bg_slot, slots, attn = self.slot_attention(tokens)
        small = F.interpolate(images[:, 0], size=self.input_size, mode="area")  # (B, 3, s, s)
        colours = small.flatten(2).transpose(1, 2)
        floor = self.decode_background(bg_slot, attn[:, 0], colours, frame)
        objects, box, state = self.decode_objects(slots, attn[:, 1:], colours, frame)  # the sketch

        # Refine: each object Gaussian reads its view-0 pixel (U-Net features); decoding continues from its q.
        side = (self.input_size, self.input_size)
        view0 = F.interpolate(images[:, 0], size=side, mode="bilinear", align_corners=False)
        enc = self.refine_encoder(view0)  # (B, C, s, s)
        obj_attn = attn[:, 1:].unflatten(2, side)  # (B, K-1, s, s)
        refined = self.obj_decoder.refine(state, images[:, 0], enc, obj_attn, frame)

        out = self.render(to_world(refined, frame), floor, cam2world)
        out.update(attn=attn, **box)  # attn (B, K, s*s); extent, box, anchor_init
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        out["rgb_pre"] = splat(self.merge(objects, floor), *cams)["rgb"]  # the sketch: auxiliary loss, panels
        return out
