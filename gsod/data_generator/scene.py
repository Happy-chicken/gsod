"""Scene container and the analytic ground-truth ray-caster."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gsod.utils.camera import Camera
from gsod.data_generator.primitives import Primitive

# Label id of every pixel that misses all objects.
BACKGROUND_ID = 0


@dataclass
class Scene:
    """Objects on one uniform background colour (no floor is drawn)."""

    primitives: list[Primitive]
    colors: np.ndarray
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
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
    0 = background) and ``depth`` ``(H, W) float32`` (hit distance along the
    unit ray, 0 = background).
    """
    height, width = camera.image_size
    origins, directions = camera.rays()

    best_t = np.full(origins.shape[0], np.inf)
    best_id = np.full(origins.shape[0], BACKGROUND_ID, dtype=np.int64)
    best_normal = np.zeros_like(directions)

    for i, primitive in enumerate(scene.primitives):
        t, hit, normal = primitive.intersect(origins, directions)
        update = hit & (t < best_t)
        best_t[update] = t[update]
        best_normal[update] = normal[update]
        best_id[update] = i + 1

    is_object = best_id > BACKGROUND_ID
    shade = scene._shade(best_normal)

    rgb = np.empty((origins.shape[0], 3), dtype=np.float64)
    rgb[:] = np.asarray(scene.background_color)
    for i, color in enumerate(scene.colors):
        selected = best_id == i + 1
        rgb[selected] = np.asarray(color) * shade[selected, None]
    rgb = np.clip(rgb, 0.0, 1.0)

    depth = np.where(is_object, best_t, 0.0)

    return {
        "rgb": (rgb * 255.0).round().astype(np.uint8).reshape(height, width, 3),
        "mask": best_id.astype(np.int8).reshape(height, width),
        "depth": depth.astype(np.float32).reshape(height, width),
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
