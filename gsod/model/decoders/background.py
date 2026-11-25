"""Background on view 0's rays.

DepthDecoder: encoder features where the background is (objects masked out) -> per pixel the share
of the way to the support plane z = 0 (1 = on the plane; walls and the like come closer).
BackgroundDecoder: one Gaussian per ray of a grid over view 0 (plus a margin), placed at that share
of the ray; appearance from the background slot only, so it cannot copy an object. Scene frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from gsod.model.layers import logit, positional_encoding, shift_colour
from gsod.utils.geometry import SceneFrame, camera_directions, rotation_to_quat

FLOOR_HALF = 3.0  # extent of the support plane used by the object anchors (past view 0's camera)
BACKGROUND_THICKNESS = 0.005  # objects stay above this
RAY_SIDE = 80  # rays per side (RAY_SIDE ** 2 Gaussians)
RAY_MARGIN = 0.25  # the ray grid reaches this fraction of the image past each edge
FAR = 10.0  # ray length when it does not reach the plane (scene units)
OPACITY_INIT = 0.95
_PE_BANDS = 8
_N_OUT = 15  # log in-plane sizes (2), opacity, rgb residual (3), sh1 (9)


class DepthDecoder(nn.Module):
    """Encoder map (B, C, s, s), background attention (B, 1, s, s) -> share of the way to the plane
    (B, 1, s, s) in (0, 1). Dilated convolutions fill in where objects are masked out."""

    def __init__(self, z_dim: int, hidden: int = 64):
        super().__init__()
        layers, c = [], z_dim + 1
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv2d(c, hidden, 3, padding=dilation, dilation=dilation), nn.ReLU()]
            c = hidden
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, 1, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.fill_(6.0)  # on the plane at first

    def forward(self, enc: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.body(torch.cat([enc * attn, attn], 1))))


class BackgroundDecoder(nn.Module):
    """Background slot + depth share map -> one Gaussian per view-0 ray (B, RAY_SIDE ** 2, ...)."""

    def __init__(self, z_dim: int, hidden_dim: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]):
        super().__init__()
        self.image_size, self.focal_ratio = tuple(image_size), tuple(focal_ratio)
        h, w = image_size
        # Ray grid in view-0 pixels (centres at integers), past the image by RAY_MARGIN.
        u = torch.linspace(-RAY_MARGIN * w, (1 + RAY_MARGIN) * w - 1, RAY_SIDE)
        v = torch.linspace(-RAY_MARGIN * h, (1 + RAY_MARGIN) * h - 1, RAY_SIDE)
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        self.register_buffer("uv", torch.stack([uu, vv], -1).reshape(-1, 2), persistent=False)
        pe = positional_encoding(self.uv / self.uv.new_tensor([w, h]) * 2 - 1, _PE_BANDS)
        self.register_buffer("pe", pe, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(pe.shape[1] + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, _N_OUT),
        )
        with torch.no_grad():
            out = self.mlp[-1]
            out.weight.mul_(0.1)
            out.weight[6:].zero_()  # view-independent at first
            out.bias.zero_()
            out.bias[2] = logit(OPACITY_INIT)

    def forward(
        self,
        latent: torch.Tensor,
        attn: torch.Tensor,
        colours: torch.Tensor,
        frame: SceneFrame,
        depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """latent (B, C), background attention (B, N), token colours (B, N, 3), view 0's frames,
        depth share map (B, 1, s, s) -> Gaussians (B, RAY_SIDE ** 2, ...), scene frame."""
        b, n = latent.shape[0], self.uv.shape[0]
        h, w = self.image_size

        # Rays (camera z = 1, so t is view 0's depth) and their length to the plane.
        dirs = camera_directions(self.uv[:, 0], self.uv[:, 1], self.image_size, self.focal_ratio) @ frame.rot.mT
        down = dirs[..., 2] < -1e-6
        t_plane = -frame.origin[:, None, 2] / torch.where(down, dirs[..., 2], -torch.ones_like(dirs[..., 2]))
        t_max = torch.where(down, t_plane.clamp(max=FAR), FAR)  # (B, n)

        # Position: the predicted share of the way (edge values past the image).
        grid = ((self.uv + 0.5) / self.uv.new_tensor([w, h]) * 2 - 1).expand(b, -1, -1)[:, :, None]
        share = F.grid_sample(depth, grid, padding_mode="border", align_corners=False)[:, 0, :, 0]
        mu = frame.origin[:, None] + (share * t_max)[..., None] * dirs  # (B, n, 3)

        # Orientation and spacing from the neighbouring rays' points: flat on the local surface.
        grid_mu = mu.view(b, RAY_SIDE, RAY_SIDE, 3)
        du = torch.diff(grid_mu, dim=2, append=grid_mu[:, :, -1:] * 2 - grid_mu[:, :, -2:-1]).reshape(b, n, 3)
        dv = torch.diff(grid_mu, dim=1, append=grid_mu[:, -1:] * 2 - grid_mu[:, -2:-1]).reshape(b, n, 3)
        e1 = F.normalize(du, dim=-1)
        e3 = F.normalize(torch.cross(du, dv, dim=-1), dim=-1)
        e2 = torch.cross(e3, e1, dim=-1)
        quat = rotation_to_quat(torch.stack([e1, e2, e3], -1))
        spacing = torch.stack([du.norm(dim=-1), (dv * e2).sum(-1).abs()], -1)  # (B, n, 2)

        # Appearance from the background slot only.
        raw = self.mlp(torch.cat([self.pe.expand(b, -1, -1), latent[:, None].expand(-1, n, -1)], -1))
        mean = (attn[..., None] * colours).sum(-2) / attn.sum(-1, keepdim=True).clamp_min(1e-6)  # (B, 3)
        size = 0.5 * spacing * torch.exp(raw[..., 0:2])
        return {
            "mu": mu,
            "scale": torch.cat([size, BACKGROUND_THICKNESS * torch.ones_like(size[..., :1])], -1),
            "quat": quat,
            "opacity": torch.sigmoid(raw[..., 2]),
            "rgb": shift_colour(mean[:, None], raw[..., 3:6]),
            "sh1": raw[..., 6:15].unflatten(-1, (3, 3)),
        }
