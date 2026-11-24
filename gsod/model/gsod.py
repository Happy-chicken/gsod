"""Slot model: view 0 -> slots -> Gaussians -> splatted into every view; a
batch of scenes at once.

    encode / slot attention   view 0 -> background slot + K-1 object slots
    decode_objects            object slot -> box on the floor at its anchor ->
                              Gaussians in it, fainter as its attention shrinks
    decode_background         background slot -> flat Gaussians on the floor
    refine                    view 0 rendered once -> a feedback map; per token
                              -> one more object-slot update; at each object
                              Gaussian's pixel (+ encoder features) -> object
                              decoding continues from its q (refine_steps rounds)
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
from gsod.model.decoders.refine import Feedback, sample_at_gaussians
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
        refine_feedback: bool = True,
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
        self.bg_decoder = BackgroundDecoder(z_dim, decoder_hidden_dim)
        self.refine_steps = refine_steps
        self.feedback = Feedback(refine_feedback)

    def networks(self) -> dict[str, nn.Module]:
        """Named networks (checkpoint files)."""
        return {
            "Encoder": self.encoder,
            "SlotAttention": self.slot_attention,
            "ObjDecoder": self.obj_decoder,
            "BgDecoder": self.bg_decoder,
            "Feedback": self.feedback,
        }

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images (B, 3, H, W) -> encoder tokens (B, s*s, C), s = input_size."""
        small = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return self.encoder(small).flatten(start_dim=2).permute(0, 2, 1)

    def decode_objects(
        self,
        slots: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
        q: torch.Tensor | None = None,
        hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """slots (B, K-1, C), their attention (B, K-1, s*s), token colours
        (B, s*s, 3) -> world Gaussians (B, K-1, G, ...), box half sizes
        (B, K-1, 3, scene frame), world box corners (B, K-1, 8, 3) and anchor_init (B, K-1, 3)."""
        local, box, state = self.obj_decoder(slots, attn, colours, frame, q, hint)
        return {
            "gaussians": to_world(local, frame),
            "local": local,  # scene frame
            "state": state,  # q etc., for the refine round
            "extent": box["extent"],
            "box": points_to_world(box_corners(box["anchor"], box["extent"]), frame),
            "anchor_init": points_to_world(box["anchor_init"], frame),
        }

    def decode_background(
        self, bg_slot: torch.Tensor, attn: torch.Tensor, colours: torch.Tensor, frame: SceneFrame
    ) -> dict[str, torch.Tensor]:
        """Background slots (B, C), attention (B, s*s), token colours
        (B, s*s, 3) -> world floor Gaussians (B, M, ...)."""
        gaussians = self.bg_decoder(bg_slot, attn, colours, frame)
        return {"gaussians": to_world(gaussians, frame, rotate=False)}

    def scene(self, slots, attn, background, colours, frame, q=None, hint=None) -> dict:
        """Object slots (B, K-1, C), attention (B, K, s*s), decoded background -> scenes;
        q, hint: continue the object decoding (refine)."""
        return {
            "objects": self.decode_objects(slots, attn[:, 1:], colours, frame, q, hint),
            "background": background,
            "attn": attn,
            "slots": slots,
            "frame": frame,
        }

    @staticmethod
    def gaussians(scenes: dict) -> dict[str, torch.Tensor]:
        """Every Gaussian of each scene (B, N, ...), objects first."""
        obj, floor = scenes["objects"]["gaussians"], scenes["background"]["gaussians"]
        return {name: torch.cat([obj[name].flatten(1, 2), floor[name]], 1) for name in obj}

    def render(self, scenes: dict, cam2world: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scenes (forward(render=False)) rendered into cam2world (B, V, 4, 4):
        every Gaussian, object or floor, sorted and blended together."""
        obj, floor = scenes["objects"]["gaussians"], scenes["background"]["gaussians"]
        b, n_obj, n_gauss = obj["mu"].shape[:3]
        every = self.gaussians(scenes)
        cams = (cam2world, self.image_size, self.focal_ratio, self.near)
        out = splat(every, *cams)
        image = out["rgb"]

        # Decomposition from the same blending (no gradient): per-slot weights
        # (0 = floor and uncovered) and per-object colour contributions.
        with torch.no_grad():
            every = {name: value.detach() for name, value in every.items()}
            ids = torch.cat([
                torch.arange(1, n_obj + 1, device=image.device).repeat_interleave(n_gauss),
                torch.zeros(floor["mu"].shape[1], dtype=torch.long, device=image.device),
            ])
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
            "attn": scenes["attn"],  # (B, K, s*s)
            "extent": scenes["objects"]["extent"],  # (B, K-1, 3)
            "box": scenes["objects"]["box"],  # (B, K-1, 8, 3) world corners
            "anchor_init": scenes["objects"]["anchor_init"],  # (B, K-1, 3) world, on the floor
        }

    def forward(self, images: torch.Tensor, cam2world: torch.Tensor, render: bool = True) -> dict:
        """images (B, V, 3, H, W) in [0, 1], cam2world (B, V, 4, 4) -> render
        outputs (B, ...), or with render=False the decoded scenes."""
        tokens = self.encode(images[:, 0])
        frame = scene_frame(cam2world[:, 0], self.image_size, self.focal_ratio)
        k, v = self.slot_attention.keys_values(tokens)
        bg_slot, slots, attn = self.slot_attention(k, v)
        small = F.interpolate(images[:, 0], size=self.input_size, mode="area")  # (B, 3, s, s)
        colours = small.flatten(2).transpose(1, 2)
        background = self.decode_background(bg_slot, attn[:, 0], colours, frame)
        scenes = self.scene(slots, attn, background, colours, frame)  # the sketch

        # Refine: one view-0 render of the current Gaussians, read by the slots
        # (per token) and by each object Gaussian (at its pixel).
        pre = scenes
        enc = tokens.transpose(1, 2).unflatten(2, (self.input_size, self.input_size))  # (B, C, s, s)
        for i in range(self.refine_steps):
            view0 = splat(self.gaussians(scenes), cam2world[:, :1], self.image_size, self.focal_ratio, self.near)
            render0 = {name: value[:, 0].detach() for name, value in view0.items()}
            feedback = self.feedback(images[:, 0], render0)  # (B, 32, H/2, W/2)
            per_token = F.interpolate(feedback, size=self.input_size, mode="bilinear", align_corners=False)
            slots, attn = self.slot_attention.refine(k, v, bg_slot, slots, per_token.flatten(2).transpose(1, 2))
            objects = scenes["objects"]
            hint = sample_at_gaussians([feedback, enc], objects["local"]["mu"], render0, frame,
                                       self.image_size, self.focal_ratio)
            q = objects["state"]["q"]
            q = q.detach() if i > 0 else q
            scenes = self.scene(slots, attn, background, colours, frame, q, hint.flatten(0, 1))
        if not render:
            return scenes
        out = self.render(scenes, cam2world)
        if self.refine_steps:  # before refinement: auxiliary loss, panels
            out["rgb_pre"] = splat(self.gaussians(pre), cam2world, self.image_size, self.focal_ratio, self.near)["rgb"]
        return out
