"""Scene container and the analytic ground-truth ray-caster."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gsod.utils.camera import Camera
from gsod.data_generator.primitives import Primitive

_EPS = 1e-6

# Label ids used by the per-pixel mask.
BACKGROUND_ID = 0  # sky and floor

@dataclass
class Scene:
    primitives: list[Primitive]
    colors: np.ndarray
    floor_z: float = 0.0
    floor_half: float = 8.0
    floor_color: tuple[float, float, float] = (0.55, 0.55, 0.60)
    floor_checker: float = 0.0  # checker tile size in world units; 0 disables
    # False: no floor is drawn; every pixel that misses all objects shows
    # background_color (one uniform colour, no floor/sky distinction).
    render_floor: bool = True
    background_color: tuple[float, float, float] = (0.93, 0.93, 0.95)
    ambient: float = 0.55
    light_dir: tuple[float, float, float] = (-0.4, -0.35, 0.85)

    @property
    def num_objects(self) -> int:
        return len(self.primitives)

    def _shade(self, normals: np.ndarray) -> np.ndarray:
        light = np.asarray(self.light_dir, dtype=np.float64)
        diffuse = np.maximum(normals @ light, 0.0)
        return self.ambient + (1.0 - self.ambient) * diffuse

def raycast(scene: Scene, camera: Camera) -> dict[str, np.ndarray]:
    """Render one view of ``scene`` analytically.

    Returns ``rgb`` ``(H, W, 3) uint8``, ``mask`` ``(H, W) int8`` (object index,
    0 = background), ``depth`` ``(H, W) float32`` (hit distance, 0 = sky) and
    ``normal`` ``(H, W, 3) float32``.
    """
    height, width = camera.image_size
    origins, directions = camera.rays()

    best_t = np.full(origins.shape[0], np.inf)
    best_id = np.zeros(origins.shape[0], dtype=np.int64)  # 0 = background
    best_normal = np.zeros_like(directions)

    for i, primitive in enumerate(scene.primitives):
        t, hit, normal = primitive.intersect(origins, directions)
        update = hit & (t < best_t)
        best_t[update] = t[update]
        best_normal[update] = normal[update]
        best_id[update] = i + 1

    # Floor plane (finite, so rays past it see the sky).
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = (scene.floor_z - origins[:, 2]) / directions[:, 2]
        floor_points = origins + t_floor[:, None] * directions
        on_floor = (np.abs(floor_points[:, 0]) <= scene.floor_half) & (
            np.abs(floor_points[:, 1]) <= scene.floor_half
        )
    hits_floor = (t_floor > _EPS) & on_floor & (t_floor < best_t)
    if not scene.render_floor:
        hits_floor[:] = False
    best_t[hits_floor] = t_floor[hits_floor]
    best_id[hits_floor] = -1
    best_normal[hits_floor] = [0.0, 0.0, 1.0]

    is_object = best_id > 0
    is_floor = best_id < 0
    shade = scene._shade(best_normal)

    rgb = np.empty((origins.shape[0], 3), dtype=np.float64)
    rgb[:] = np.asarray(scene.background_color)

    for i, color in enumerate(scene.colors):
        selected = best_id == i + 1
        rgb[selected] = np.asarray(color) * shade[selected, None]

    floor_base = np.asarray(scene.floor_color, dtype=np.float64)
    floor_rgb = np.broadcast_to(floor_base, (origins.shape[0], 3)).copy()
    if scene.floor_checker > 0.0:
        tile = (
            np.floor(floor_points[:, 0] / scene.floor_checker)
            + np.floor(floor_points[:, 1] / scene.floor_checker)
        )
        floor_rgb = np.where((tile % 2.0)[:, None] == 0, floor_base, floor_base * 0.82)
    rgb[is_floor] = floor_rgb[is_floor] * shade[is_floor, None]

    rgb = np.clip(rgb, 0.0, 1.0)

    depth = np.where(is_object | is_floor, best_t, 0.0)
    mask = np.where(is_object, best_id, BACKGROUND_ID)

    return {
        "rgb": (rgb * 255.0).round().astype(np.uint8).reshape(height, width, 3),
        "mask": mask.astype(np.int8).reshape(height, width),
        "depth": depth.astype(np.float32).reshape(height, width),
        "normal": best_normal.astype(np.float32).reshape(height, width, 3),
    }


def render_views(scene: Scene, cam2world, image_size, focal_ratio=(1.0, 1.0)) -> dict:
    """Ray-cast several views of ``scene``; stacks the per-view arrays."""
    images, masks, depths = [], [], []
    for view in np.asarray(cam2world, dtype=np.float64):
        out = raycast(scene, Camera(view, image_size, focal_ratio))
        images.append(out["rgb"])
        masks.append(out["mask"])
        depths.append(out["depth"])
    return {
        "images": np.stack(images),
        "mask": np.stack(masks),
        "depth": np.stack(depths),
    }
