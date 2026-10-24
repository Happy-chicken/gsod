"""Random scene/view sampling for the Primordia dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gsod.data_generator.constants import PALETTE
from gsod.utils.camera import orbit
from gsod.data_generator.primitives import make_primitive
from gsod.data_generator.scene import Scene

__all__ = ["SceneGenConfig", "sample_scene", "sample_views", "generate"]


@dataclass
class SceneGenConfig:
    """All knobs for generating the dataset (serialised into meta.json)."""

    n_scenes: int = 200
    k_min: int = 3
    k_max: int = 5
    shapes: tuple[str, ...] = ("sphere", "cuboid")
    image_size: tuple[int, int] = (128, 128)  # (H, W)
    focal_ratio: tuple[float, float] = (1.0, 1.0)
    n_views: int = 6

    radius_range: tuple[float, float] = (0.35, 0.55)
    placement_radius: float = 1.8
    min_gap: float = 0.15
    max_placement_tries: int = 50

    elevation_range: tuple[float, float] = (12.0, 28.0)
    distance_range: tuple[float, float] = (4.5, 6.5)
    azimuth_jitter: float = 3.0
    target_height: float = 0.4

    # Objects rest on this plane; it is not drawn (one uniform background).
    floor_z: float = 0.0
    background_color: tuple[float, float, float] = (0.5, 0.5, 0.5)

    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 0


def _sample_colors(rng: np.random.Generator, k: int) -> np.ndarray:
    if k > len(PALETTE):
        raise ValueError(f"{k} objects exceed the {len(PALETTE)} palette colours")
    idx = rng.choice(len(PALETTE), size=k, replace=False)
    return np.asarray([PALETTE[i] for i in idx], dtype=np.float64)


def sample_scene(cfg: SceneGenConfig, rng: np.random.Generator) -> Scene:
    """Draw one random scene of k_min..k_max non-overlapping objects."""
    k = int(rng.integers(cfg.k_min, cfg.k_max + 1))
    colors = _sample_colors(rng, k)

    primitives = []
    placed: list[tuple[np.ndarray, float]] = []
    for i in range(k):
        kind = cfg.shapes[int(rng.integers(len(cfg.shapes)))]
        for _ in range(cfg.max_placement_tries):
            size = float(rng.uniform(*cfg.radius_range))
            angle = rng.uniform(0.0, 2.0 * np.pi)
            radius = cfg.placement_radius * np.sqrt(rng.uniform())
            xy = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            if all(
                np.linalg.norm(xy - other_xy) > size + other_size + cfg.min_gap
                for other_xy, other_size in placed
            ):
                break
        else:
            continue  # could not place this object; drop it
        center = np.array([xy[0], xy[1], cfg.floor_z + size])
        primitives.append(make_primitive(kind, center, size))
        placed.append((xy, size))

    colors = colors[: len(primitives)]
    return Scene(
        primitives=primitives,
        colors=colors,
        background_color=tuple(cfg.background_color),
    )


def sample_views(cfg: SceneGenConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw n_views orbiting cameras around the scene centre.

    Returns (n_views, 4, 4) cam2world matrices.
    """
    start = float(rng.uniform(0.0, 360.0))
    azimuths = np.linspace(0.0, 360.0, cfg.n_views, endpoint=False)
    azimuths = azimuths + start + rng.normal(0.0, cfg.azimuth_jitter, size=cfg.n_views)
    elevations = rng.uniform(*cfg.elevation_range, size=cfg.n_views)
    distances = rng.uniform(*cfg.distance_range, size=cfg.n_views)
    center = np.array([0.0, 0.0, cfg.target_height])
    return np.stack(
        [orbit(center, distances[i], elevations[i], azimuths[i]) for i in range(cfg.n_views)]
    )


def generate(cfg: SceneGenConfig):
    """Yield {"scene", "cam2world"} dicts, one per scene."""
    rng = np.random.default_rng(cfg.seed)
    for _ in range(cfg.n_scenes):
        scene = sample_scene(cfg, rng)
        yield {"scene": scene, "cam2world": sample_views(cfg, rng)}


def config_dict(cfg: SceneGenConfig) -> dict:
    """JSON-ready view of cfg (used by meta.json)."""
    return asdict(cfg)
