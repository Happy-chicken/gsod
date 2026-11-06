"""Geometry: cameras, the floor and the scene frame.

Conventions: world right-handed, +z up, the floor at z = 0; cameras OpenCV
(looking down +z, +x right, +y down), cam2world a 4x4 camera -> world
transform; pixel centres at integer coordinates, the principal point at the
image centre.

The scene frame, set by view 0's camera: the model predicts in it and maps
its Gaussians into the world before splatting. Like uORF's nss2cam0, it is
the world turned about +z so that view 0 looks along +x; it is centred at c,
where view 0's optical axis meets the floor, and scaled by 1 / R, the radius
of the largest disc about c that view 0's floor footprint holds. Every
length the model assumes is a constant in units of R, so no scene
statistics are needed (view 0 must look down at the floor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


# -- cameras and the floor ----------------------------------------------------
def camera_directions(u, v, size: tuple[int, int], focal_ratio):
    """Camera-frame ray directions (..., 3) (z = 1) through pixels (u, v):
    numpy arrays or torch tensors (the result is the same kind)."""
    h, w = size
    x = (u - (w - 1) / 2) / (focal_ratio[0] * w)
    y = (v - (h - 1) / 2) / (focal_ratio[1] * h)
    if torch.is_tensor(x):
        return torch.stack([x, y, torch.ones_like(x)], -1)
    return np.stack([x, y, np.ones_like(x)], -1)


def floor_hits(origin, directions, floor_z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where rays origin + t * directions (..., 3) meet z = floor_z: xy
    (..., 2), t (...,) and whether the ray goes down (elsewhere t = 0)."""
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
    """Build a cam2world (4x4) that looks from eye at target.

    up chooses the camera roll; it must not be parallel to the view
    direction. +z is up by default.
    """
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
    """cam2world of a camera on a sphere of radius around center.

    elevation_deg is measured from the horizontal plane, azimuth_deg
    from the world +x axis (counter-clockwise around up).
    """
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
# Image border points per edge sampled for R.
_BORDER_SAMPLES = 64


class SceneFrame(NamedTuple):
    """yaw: view 0's world azimuth; centre (3,) c and scale R: the frame's
    origin and unit in the world; origin (3,) and rot (3, 3): view 0's camera
    in the frame (columns: camera axes, OpenCV)."""

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
    # R: the nearest point of the footprint's boundary (the image border cast
    # onto the floor; border rays above the horizon do not bound it).
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
    """Degree-1 SH (..., 3, 3) (basis rows, gsplat order) rotated by rot.

    The degree-1 term is C1 (-y c1 + z c2 - x c3) = C1 d . w with
    w = (-c3, -c1, c2), so the coefficients rotate as the vector w."""
    c1, c2, c3 = sh1.unbind(-2)
    w = torch.einsum("ij,...jc->...ic", rot.to(sh1.dtype), torch.stack([-c3, -c1, c2], -2))
    return torch.stack([-w[..., 1, :], w[..., 2, :], -w[..., 0, :]], -2)


def points_to_world(points: torch.Tensor, frame: SceneFrame, rotate: bool = True) -> torch.Tensor:
    """Points (..., 3) from the scene frame into the world; rotate=False for
    a frame that is only centred and scaled (world-aligned)."""
    if rotate:
        points = points @ yaw_matrix(frame.yaw).to(points.dtype).T
    return points * frame.scale.to(points.dtype) + frame.centre.to(points.dtype)


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """(4,) wxyz quaternion of a rotation by angle about +z."""
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)])


def rotate_gaussians(
    gaussians: dict[str, torch.Tensor], rot: torch.Tensor, quat: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Gaussians turned by rot (3, 3), whose quaternion is quat (4,):
    positions, rotations and degree-1 SH."""
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
    """Gaussians from the scene frame into the world: with rotate turned by
    the yaw, then sizes scaled by R and positions moved to c."""
    if rotate:
        gaussians = rotate_gaussians(gaussians, yaw_matrix(frame.yaw), yaw_quat(frame.yaw))
    return {
        **gaussians,
        "mu": points_to_world(gaussians["mu"], frame, rotate=False),
        "scale": gaussians["scale"] * frame.scale.to(gaussians["scale"].dtype),
    }


def from_world(gaussians: dict[str, torch.Tensor], frame: SceneFrame) -> dict[str, torch.Tensor]:
    """Gaussians from the world into the scene frame (to_world's inverse;
    world-aligned parts are turned too, so the whole scene shares one frame)."""
    dtype = gaussians["mu"].dtype
    moved = {
        **gaussians,
        "mu": (gaussians["mu"] - frame.centre.to(dtype)) / frame.scale.to(dtype),
        "scale": gaussians["scale"] / frame.scale.to(gaussians["scale"].dtype),
    }
    return rotate_gaussians(moved, yaw_matrix(-frame.yaw), yaw_quat(-frame.yaw))
