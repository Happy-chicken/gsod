"""Refine round: what each object Gaussian sees at its view-0 pixel (the hint
that continued object decoding adds to its q)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gsod.utils.geometry import SceneFrame


def sample_at_gaussians(
    enc: torch.Tensor,
    mu: torch.Tensor,
    frame: SceneFrame,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
    render: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Encoder map (B, C, s, s) over view 0, Gaussian centres mu (B, K, G, 3,
    scene frame) [, view-0 render {"depth", "alpha"}] -> per Gaussian the encoder
    features at its pixel [+ occlusion] (B, K, G, C [+ 1]); 0 outside the image."""
    k, g = mu.shape[1:3]
    h, w = image_size
    # Project each centre into view 0 (inverse of the token rays in init_attention_anchor).
    cam = (mu.flatten(1, 2).detach() - frame.origin[:, None]) @ frame.rot  # (B, N, 3)
    z = cam[..., 2:]
    zs = z.clamp_min(1e-4)
    u = focal_ratio[0] * w * cam[..., :1] / zs + (w - 1) / 2
    v = focal_ratio[1] * h * cam[..., 1:2] / zs + (h - 1) / 2
    grid = torch.cat([(u + 0.5) / w * 2 - 1, (v + 0.5) / h * 2 - 1], -1)[:, :, None]  # (B, N, 1, 2)

    def sample(x: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, N, C)
        return F.grid_sample(x, grid, align_corners=False)[..., 0].transpose(1, 2)

    hint = [sample(enc)]
    if render is not None:
        # Occlusion: the Gaussian's depth - the rendered depth at its pixel (scene units).
        depth = render["depth"] / render["alpha"].clamp_min(1e-4) / frame.scale[:, None, None, None]
        hint.append(z - sample(depth.detach()))
    return torch.cat(hint, -1).unflatten(1, (k, g))
