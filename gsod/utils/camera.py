"""Cameras and camera-to-world transforms.

Conventions
-----------
* World: right-handed, +z up.
* Camera: OpenCV/NeRF convention -- looking down +z with +x right and
  +y down, so image rows increase downwards.
* cam2world is a 4x4 homogeneous matrix mapping camera to world coords.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Camera",
    "look_at",
    "orbit",
    "orbit_views",
    "camera_azimuth",
    "azimuth_rotation",
    "get_camera_matrices",
]

@dataclass
class Camera:
    """A pinhole camera.
    """

    cam2world: np.ndarray
    image_size: tuple[int, int]
    focal_ratio: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        self.cam2world = np.asarray(self.cam2world, dtype=np.float64).reshape(4, 4)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.focal_ratio = (float(self.focal_ratio[0]), float(self.focal_ratio[1]))

    @property
    def height(self) -> int:
        return self.image_size[0]

    @property
    def width(self) -> int:
        return self.image_size[1]

    @property
    def center(self) -> np.ndarray:
        return self.cam2world[:3, 3]

    def intrinsics(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix."""
        fx = self.focal_ratio[0] * self.width
        fy = self.focal_ratio[1] * self.height
        cx = (self.width - 1.0) / 2.0
        cy = (self.height - 1.0) / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def ray_directions(self) -> np.ndarray:
        """Unit world-space ray directions, shape (H, W, 3)."""
        fx = self.focal_ratio[0] * self.width
        fy = self.focal_ratio[1] * self.height
        cx = (self.width - 1.0) / 2.0
        cy = (self.height - 1.0) / 2.0

        us = np.arange(self.width, dtype=np.float64)
        vs = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)  # (H, W)
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (H, W, 3)
        rotation = self.cam2world[:3, :3]
        dirs_world = dirs_cam @ rotation.T
        return dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    def rays(self) -> tuple[np.ndarray, np.ndarray]:
        """(origins, directions) flattened to (H*W, 3) each."""
        directions = self.ray_directions().reshape(-1, 3)
        origins = np.broadcast_to(self.center, directions.shape).copy()
        return origins, directions

    def project(self, points_world) -> tuple[np.ndarray, np.ndarray]:
        """Project world points to pixels.

        Returns (uv, valid) where uv is (N, 2) and valid marks
        points in front of the camera.
        """
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        rotation = self.cam2world[:3, :3]
        points_cam = (points_world - self.center) @ rotation
        z = points_cam[:, 2]
        valid = z > 1e-8
        z_safe = np.where(valid, z, 1.0)
        fx, fy = self.focal_ratio[0] * self.width, self.focal_ratio[1] * self.height
        cx, cy = (self.width - 1.0) / 2.0, (self.height - 1.0) / 2.0
        u = fx * points_cam[:, 0] / z_safe + cx
        v = fy * points_cam[:, 1] / z_safe + cy
        return np.stack([u, v], axis=-1), valid


# camera settings
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


def orbit_views(
    center,
    radius,
    n_views: int,
    elevation_deg=15.0,
    azimuth_start_deg=0.0,
    azimuth_span_deg=360.0,
    up=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """(n_views, 4, 4) array of cameras evenly spaced on an orbit.

    elevation_deg may be a scalar or a per-view array.
    """
    n_views = int(n_views)
    azimuths = azimuth_start_deg + np.linspace(0.0, azimuth_span_deg, n_views, endpoint=False)
    elevations = np.broadcast_to(np.asarray(elevation_deg, dtype=np.float64), (n_views,))
    return np.stack(
        [orbit(center, radius, elevations[i], azimuths[i], up) for i in range(n_views)]
    )


def camera_azimuth(cam2world) -> float:
    """Azimuth (radians) of a camera's viewing direction about world +z."""
    forward = np.asarray(cam2world, dtype=np.float64)[:3, 2]
    return float(np.arctan2(forward[1], forward[0]))


def azimuth_rotation(cam2world) -> np.ndarray:
    """3x3 yaw rotation (about +z) of a camera's azimuth.

    This is the azi_rot stored per view. Its inverse is the world ->
    azimuth-frame map (uORF's nss2cam0).
    """
    angle = camera_azimuth(cam2world)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def get_camera_matrices(camera: Camera, near: float = 0.1, far: float = 100.0) -> dict:
    """Camera matrices for a 3DGS-style rasterizer.

    Returns transposed world_view_transform / projection_matrix /
    full_proj_transform (as the CUDA rasterizer expects), the camera center,
    and FoV/tan terms.
    """
    width, height = camera.width, camera.height
    fx = camera.focal_ratio[0] * width
    fy = camera.focal_ratio[1] * height

    fov_x = 2.0 * np.arctan(width / (2.0 * fx))
    fov_y = 2.0 * np.arctan(height / (2.0 * fy))

    world_view = np.linalg.inv(camera.cam2world)  # world -> camera

    tan_half_x = np.tan(fov_x / 2.0)
    tan_half_y = np.tan(fov_y / 2.0)
    top = tan_half_y * near
    bottom = -top
    right = tan_half_x * near
    left = -right

    projection = np.zeros((4, 4))
    projection[0, 0] = 2.0 * near / (right - left)
    projection[1, 1] = 2.0 * near / (top - bottom)
    projection[0, 2] = (right + left) / (right - left)
    projection[1, 2] = (top + bottom) / (top - bottom)
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    projection[3, 2] = 1.0

    world_view_t = world_view.T
    projection_t = projection.T
    return {
        "world_view_transform": world_view_t,
        "full_proj_transform": world_view_t @ projection_t,
        "projection_matrix": projection_t,
        "camera_center": camera.center.copy(),
        "FoVx": float(fov_x),
        "FoVy": float(fov_y),
        "tanfovx": float(tan_half_x),
        "tanfovy": float(tan_half_y),
        "image_height": height,
        "image_width": width,
        "znear": float(near),
        "zfar": float(far),
    }

