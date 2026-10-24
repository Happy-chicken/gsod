"""solid primitives and their ray intersections"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

_EPS = 1e-6

__all__ = ["Primitive", "Sphere", "Cuboid", "make_primitive"]

class Primitive(ABC):
    """Base class for a solid shape."""

    @abstractmethod
    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest intersection of ``P`` rays with the surface.

        Returns ``(t, hit, normal)``: distances ``(P,)`` (``inf`` where there is
        no hit), a boolean mask ``(P,)`` and unit normals ``(P, 3)`` (zero where
        there is no hit). ``direction`` need not be normalised.
        """

    @abstractmethod
    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``(n, 3)`` points sampled on the surface."""

    @abstractmethod
    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """``(center, radius)`` of the tightest enclosing sphere."""

class Sphere(Primitive):
    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.radius = float(radius)

    def intersect(self, origin, direction):
        oc = origin - self.center
        a = np.einsum("ij,ij->i", direction, direction)
        b = 2.0 * np.einsum("ij,ij->i", oc, direction)
        c = np.einsum("ij,ij->i", oc, oc) - self.radius ** 2
        disc = b * b - 4.0 * a * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

        t_near = (-b - sqrt_disc) / (2.0 * a)
        t_far = (-b + sqrt_disc) / (2.0 * a)
        t = np.where(t_near > _EPS, t_near, t_far)
        hit = (disc > 0.0) & (t > _EPS)
        t = np.where(hit, t, np.inf)

        points = origin + np.where(hit, t, 0.0)[:, None] * direction
        normal = (points - self.center) / self.radius
        normal[~hit] = 0.0
        return t, hit, normal

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # Fibonacci sphere: near-uniform without rejection sampling.
        i = np.arange(n, dtype=np.float64) + 0.5
        phi = np.arccos(1.0 - 2.0 * i / n)
        theta = np.pi * (1.0 + np.sqrt(5.0)) * i
        unit = np.stack(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
            axis=-1,
        )
        return self.center + self.radius * unit

    def bounding_sphere(self):
        return self.center.copy(), self.radius


class Cuboid(Primitive):
    """Axis-aligned box; ``half_extents`` are half side lengths per axis."""

    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.half_extents = np.abs(np.asarray(half_extents, dtype=np.float64).reshape(3))

    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.half_extents

    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.half_extents

    def intersect(self, origin, direction):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / direction
            t0 = (self.min_corner - origin) * inv
            t1 = (self.max_corner - origin) * inv
        t_lo = np.minimum(t0, t1)
        t_hi = np.maximum(t0, t1)

        t_near = np.max(t_lo, axis=1)
        t_far = np.min(t_hi, axis=1)
        hit = t_far >= np.maximum(t_near, _EPS)
        t = np.where(t_near > _EPS, t_near, t_far)
        hit &= t > _EPS
        t = np.where(hit, t, np.inf)

        rows = np.arange(origin.shape[0])
        axis = np.argmax(t_lo, axis=1)
        sign = np.where(direction[rows, axis] > 0.0, -1.0, 1.0)
        normal = np.zeros_like(direction)
        normal[rows, axis] = sign
        normal[~hit] = 0.0
        return t, hit, normal

    def sample_surface(self, n: int, rng: np.random.Generator) -> np.ndarray:
        hx, hy, hz = self.half_extents
        # +x/-x: 4*hy*hz, +y/-y: 4*hx*hz, +z/-z: 4*hx*hy
        face_areas = np.array([4.0 * hy * hz, 4.0 * hx * hz, 4.0 * hx * hy])
        probs = face_areas / face_areas.sum()

        axis = rng.choice(3, size=n, p=probs)
        sign = rng.choice([-1.0, 1.0], size=n)

        # Local coords on the [-1, 1]^2 face, then map to the two other axes.
        uv = rng.uniform(-1.0, 1.0, size=(n, 2))
        points = np.empty((n, 3))
        for a in range(3):
            sel = axis == a
            if not sel.any():
                continue
            others = [b for b in range(3) if b != a]
            points[sel, a] = sign[sel] * self.half_extents[a]
            points[sel, others[0]] = uv[sel, 0] * self.half_extents[others[0]]
            points[sel, others[1]] = uv[sel, 1] * self.half_extents[others[1]]
        return self.center + points

    def bounding_sphere(self):
        return self.center.copy(), float(np.linalg.norm(self.half_extents))


def make_primitive(kind: str, center, size, rng: np.random.Generator) -> Primitive:
    """Factory: build a ``kind`` primitive of rough scale ``size``.

    ``size`` is the sphere radius or the cube half side length. Cuboids are
    sampled as cubes for now (``kind in {"cuboid", "cube", "box"}``).
    """
    kind = kind.lower()
    if kind == "sphere":
        return Sphere(center, size)
    if kind in {"cuboid", "cube", "box"}:
        return Cuboid(center, [size, size, size])
    raise ValueError(f"unknown primitive kind: {kind!r}")

