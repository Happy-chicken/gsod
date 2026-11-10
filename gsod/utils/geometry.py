"""Cameras, the floor and the scene frame.

World: +z up, floor at z = 0; cameras OpenCV, cam2world 4x4; pixel centres
at integers. Scene frame (like uORF's nss2cam0): world turned so view 0
looks along +x, centred where its optical axis meets the floor, scaled by
1 / R (R: radius of the largest floor disc view 0 sees about that point).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F


# -- cameras and the floor ----------------------------------------------------
def camera_directions(u, v, size: tuple[int, int], focal_ratio):
    """Camera-frame ray directions (..., 3), z = 1 (numpy or torch)."""
    h, w = size
    x = (u - (w - 1) / 2) / (focal_ratio[0] * w)
    y = (v - (h - 1) / 2) / (focal_ratio[1] * h)
    if torch.is_tensor(x):
        return torch.stack([x, y, torch.ones_like(x)], -1)
    return np.stack([x, y, np.ones_like(x)], -1)


def floor_hits(origin, directions, floor_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rays meeting z = floor_z -> xy, t, and whether the ray goes down."""
    origin = np.asarray(origin, dtype=np.float64)
    dz = directions[..., 2]
    down = dz < -1e-6
    t = np.where(down, (floor_z - origin[..., 2]) / np.where(down, dz, -1.0), 0.0)
    return origin[..., :2] + t[..., None] * directions[..., :2], t, down


@dataclass
class Camera:
    """A pinhole camera."""

    cam2world: np.ndarray
    image_size: tuple[int, int]
    focal_ratio: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        self.cam2world = np.asarray(self.cam2world, dtype=np.float64).reshape(4, 4)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.focal_ratio = (float(self.focal_ratio[0]), float(self.focal_ratio[1]))

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """World origins and unit directions, (H*W, 3) each, row-major."""
        h, w = self.image_size
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        d = camera_directions(u, v, self.image_size, self.focal_ratio).reshape(-1, 3)
        d = d @ self.cam2world[:3, :3].T
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(self.cam2world[:3, 3], d.shape).copy(), d


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """cam2world (4x4) looking from eye at target; up sets the roll."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        raise ValueError("look_at: view direction is parallel to `up`")
    right = right / norm
    down = np.cross(forward, right)

    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = down
    cam2world[:3, 2] = forward
    cam2world[:3, 3] = eye
    return cam2world


def orbit(
    center,
    radius: float,
    elevation_deg: float,
    azimuth_deg: float,
    up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """cam2world on a sphere around center (azimuth from +x, CCW)."""
    elevation = np.deg2rad(elevation_deg)
    azimuth = np.deg2rad(azimuth_deg)
    offset = radius * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    eye = np.asarray(center, dtype=np.float64) + offset
    return look_at(eye, center, up)


# -- the scene frame ----------------------------------------------------------
_BORDER_SAMPLES = 64


class SceneFrame(NamedTuple):
    """yaw, centre c, scale R in the world; origin, rot: view 0's camera in
    the frame."""

    yaw: torch.Tensor
    centre: torch.Tensor
    scale: torch.Tensor
    origin: torch.Tensor
    rot: torch.Tensor


def scene_frame(
    cam2world: torch.Tensor, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> SceneFrame:
    """View 0's (4, 4) cam2world -> its SceneFrame."""
    rot, o = cam2world[:3, :3], cam2world[:3, 3]
    forward = rot[:, 2]
    yaw = torch.atan2(forward[1], forward[0])
    centre = o - (o[2] / forward[2]) * forward
    # R: nearest image-border point cast onto the floor.
    h, w = image_size
    s = torch.linspace(0.0, 1.0, _BORDER_SAMPLES, device=o.device, dtype=o.dtype)
    zero, one = torch.zeros_like(s), torch.ones_like(s)
    u = torch.cat([s, s, zero, one]) * (w - 1)
    v = torch.cat([zero, one, s, s]) * (h - 1)
    d = camera_directions(u, v, image_size, focal_ratio) @ rot.T
    down = d[:, 2] < -1e-6
    t = -o[2] / torch.where(down, d[:, 2], -torch.ones_like(u))
    reach = (o[:2] + t[:, None] * d[:, :2] - centre[:2]).norm(dim=-1)
    scale = torch.where(down, reach, torch.full_like(reach, float("inf"))).min()
    to_frame = yaw_matrix(-yaw).to(o.dtype)
    return SceneFrame(yaw, centre, scale, to_frame @ (o - centre) / scale, to_frame @ rot)


def pixel_rays(
    frame: "SceneFrame", side: int, image_size: tuple[int, int], focal_ratio: tuple[float, float]
) -> torch.Tensor:
    """(side*side, 3) scene-frame directions of view 0's rays through a
    side x side grid of pixel centres, row-major; unit camera depth (z = 1)."""
    h, w = image_size
    idx = torch.arange(side, device=frame.rot.device, dtype=frame.rot.dtype)
    u = (idx + 0.5) * w / side - 0.5
    v = (idx + 0.5) * h / side - 0.5
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return camera_directions(uu, vv, image_size, focal_ratio).reshape(-1, 3) @ frame.rot.T


def read_view0(points: torch.Tensor, maps: torch.Tensor, frame: SceneFrame, focal_ratio) -> torch.Tensor:
    """Scene-frame points (B, P, 3) -> maps (B, C, s, s) read bilinearly where
    each point projects in view 0 (B, P, C); 0 behind the camera or off the image."""
    cam = (points - frame.origin) @ frame.rot
    front = cam[..., 2] > 1e-3
    z = torch.where(front, cam[..., 2], torch.ones_like(cam[..., 2]))
    # grid_sample coordinates (align_corners=False): pixel centres at integers.
    grid = torch.stack([2.0 * focal_ratio[0] * cam[..., 0] / z, 2.0 * focal_ratio[1] * cam[..., 1] / z], -1)
    out = F.grid_sample(maps, grid[:, None], mode="bilinear", align_corners=False)[:, :, 0]
    return out.transpose(1, 2) * front[..., None]


def yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    """(3, 3) rotation by angle about +z."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [torch.stack([c, -s, zero]), torch.stack([s, c, zero]), torch.stack([zero, zero, one])]
    )


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (..., 4) wxyz quaternions."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def rotate_sh1(sh1: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Degree-1 SH (..., 3, 3) rotated by rot (coefficients as w = (-c3, -c1, c2))."""
    c1, c2, c3 = sh1.unbind(-2)
    w = torch.einsum("ij,...jc->...ic", rot.to(sh1.dtype), torch.stack([-c3, -c1, c2], -2))
    return torch.stack([-w[..., 1, :], w[..., 2, :], -w[..., 0, :]], -2)


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """(4,) wxyz quaternion of a rotation by angle about +z."""
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)])


def rotate_gaussians(
    gaussians: dict[str, torch.Tensor], rot: torch.Tensor, quat: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Gaussians rotated by rot (3, 3) / quat (4,)."""
    out = {
        **gaussians,
        "mu": gaussians["mu"] @ rot.to(gaussians["mu"].dtype).T,
        "quat": quat_mul(quat.to(gaussians["quat"].dtype).expand_as(gaussians["quat"]), gaussians["quat"]),
    }
    if "sh1" in gaussians:
        out["sh1"] = rotate_sh1(gaussians["sh1"], rot)
    return out


def to_world(
    gaussians: dict[str, torch.Tensor], frame: SceneFrame, rotate: bool = True
) -> dict[str, torch.Tensor]:
    """Gaussians, scene frame -> world."""
    if rotate:
        gaussians = rotate_gaussians(gaussians, yaw_matrix(frame.yaw), yaw_quat(frame.yaw))
    mu = gaussians["mu"]
    return {
        **gaussians,
        "mu": mu * frame.scale.to(mu.dtype) + frame.centre.to(mu.dtype),
        "scale": gaussians["scale"] * frame.scale.to(gaussians["scale"].dtype),
    }


def from_world(gaussians: dict[str, torch.Tensor], frame: SceneFrame) -> dict[str, torch.Tensor]:
    """Gaussians, world -> scene frame (inverse of to_world)."""
    dtype = gaussians["mu"].dtype
    moved = {
        **gaussians,
        "mu": (gaussians["mu"] - frame.centre.to(dtype)) / frame.scale.to(dtype),
        "scale": gaussians["scale"] / frame.scale.to(gaussians["scale"].dtype),
    }
    return rotate_gaussians(moved, yaw_matrix(-frame.yaw), yaw_quat(-frame.yaw))
