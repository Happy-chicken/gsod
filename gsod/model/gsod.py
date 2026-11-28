"""Splatter-OD: view 0 -> slots -> Gaussians on view 0's pixels -> splatted into every view; a
batch of scenes at once.

    encode          view 0 -> DINO tokens (s x s)
    slot_attention  tokens -> background slot + K-1 object slots, their attention
    obj_decoder     per object slot a box on the floor and a Splatter Image of view 0 inside it:
                    Gaussians on the window pixels' rays, within the box
    bg_decoder      background slot -> flat Gaussians on the floor (cannot draw objects)
    render          all Gaussians blended together; masks from the same blending

Gaussians are decoded in view 0's scene frame (utils/geometry.py; floor z = 0). Slot 0 is the
background.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.encoders import DinoEncoder
from gsod.model.decoders import BackgroundDecoder, ObjectDecoder
from gsod.model.decoders.objects import box_corners
from gsod.model.slot_attention import SlotAttention
from gsod.utils.geometry import points_to_world, scene_frame, to_world
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
        gaussians_per_point: int,
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

        self.encoder = DinoEncoder(z_dim, input_size)
        self.slot_attention = SlotAttention(num_slots, z_dim, attn_iter)
        self.obj_decoder = ObjectDecoder(z_dim, decoder_hidden_dim, gaussians_per_point, image_size, focal_ratio)
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "BgDecoder": self.bg_decoder,
        }

    def render(
        self, gaussians: dict[str, torch.Tensor], owner: torch.Tensor, cam2world: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """World Gaussians (B, N, ...) with owners (B, N, K) rendered into cam2world (B, V, 4, 4);
        per-slot masks and colours from the same blending."""
        n_slots = owner.shape[-1]
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        out = splat(gaussians, *cams)

        # Decomposition from the same blending (no gradient): per-slot weights
        # (0 = background and uncovered) and per-object colour contributions.
        with torch.no_grad():
            every = {name: value.detach() for name, value in gaussians.items()}
            owner = owner.detach()
            obj_rgb = (owner[..., 1:, None] * every["rgb"][..., None, :]).flatten(-2)
            parts = splat(every, *cams, features=torch.cat([owner, obj_rgb], -1))["features"]
            weights = parts[:, :, :n_slots].clone()
            weights[:, :, 0] += 1.0 - out["alpha"][:, :, 0]
            slot_rgb = parts[:, :, n_slots:].unflatten(2, (n_slots - 1, 3))
            bg_rgb = splat({**every, "opacity": every["opacity"] * owner[..., 0]}, *cams)["rgb"]

        return {
            "rgb": out["rgb"],  # (B, V, 3, H, W)
            "fg_rgb": slot_rgb.sum(2),
            "slot_rgb": slot_rgb,  # (B, V, K-1, 3, H, W)
            "bg_rgb": bg_rgb,  # the background's Gaussians alone
            "slot_alpha": weights,  # (B, V, K, H, W), 0 = background
            "mask_idx": weights.argmax(dim=2),  # (B, V, H, W)
        }

    def decode(self, images: torch.Tensor, cam2world: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """View 0 -> world Gaussians (B, N, ...), their slot one-hot (B, N, K), attention (B, K, s*s)
        and the boxes {"extent" (B, K-1, 3, scene frame), "box" (B, K-1, 8, 3) world corners,
        "anchor_init" (B, K-1, 3) world}."""
        tokens = self.encoder(images[:, 0]).flatten(2).transpose(1, 2)  # (B, s*s, C)
        frame = scene_frame(cam2world[:, 0], self.image_size, self.focal_ratio)
        bg_slot, slots, attn = self.slot_attention(tokens)
        small = F.interpolate(images[:, 0], size=self.input_size, mode="area")  # (B, 3, s, s)
        colours = small.flatten(2).transpose(1, 2)  # token colours
        floor = to_world(self.bg_decoder(bg_slot, attn[:, 0], colours, frame), frame, rotate=False)  # (B, M, ...)
        objects, box = self.obj_decoder(slots, attn[:, 1:], images[:, 0], frame)
        objects = to_world(objects, frame)  # (B, K-1, G, ...)
        box = {
            "extent": box["extent"],
            "box": points_to_world(box_corners(box["anchor"], box["extent"]), frame),
            "anchor_init": points_to_world(box["anchor_init"], frame),
        }

        # Every Gaussian, objects first, and its slot (0 = background).
        b, n_obj, n_gauss = objects["mu"].shape[:3]
        gaussians = {name: torch.cat([objects[name].flatten(1, 2), floor[name]], 1) for name in floor}
        slot = torch.arange(1, n_obj + 1, device=images.device).repeat_interleave(n_gauss)
        slot = F.pad(slot, (0, floor["mu"].shape[1]))
        owner = F.one_hot(slot, n_obj + 1).to(images.dtype).expand(b, -1, -1)  # (B, N, K)
        return gaussians, owner, attn, box

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor) -> dict:
        """images (B, V, 3, H, W) in [0, 1], cam2world (B, V, 4, 4) -> render outputs (B, ...) and attention."""
        gaussians, owner, attn, box = self.decode(images, cam2world)
        out = self.render(gaussians, owner, cam2world)
        out.update(attn=attn, **box)  # attn (B, K, s*s); extent, box, anchor_init
        return out
