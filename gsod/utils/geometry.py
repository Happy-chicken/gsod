"""Camera rays, the scene frame and moving Gaussians between frames.

World: +z up, floor at z = 0; cameras OpenCV, cam2world 4x4; pixel centres
at integers. Scene frame: world turned so view 0
looks along +x, centred where its optical axis meets the floor, scaled by
1 / R (R: radius of the largest floor disc view 0 sees about that point).
"""

from __future__ import annotations

from typing import NamedTuple

import torch


def camera_directions(u: torch.Tensor, v: torch.Tensor, size: tuple[int, int], focal_ratio) -> torch.Tensor:
    """Camera-frame ray directions (..., 3), z = 1."""
    h, w = size
    x = (u - (w - 1) / 2) / (focal_ratio[0] * w)
    y = (v - (h - 1) / 2) / (focal_ratio[1] * h)
    return torch.stack([x, y, torch.ones_like(x)], -1)


_BORDER_SAMPLES = 64


class SceneFrame(NamedTuple):
    """yaw, centre c, scale R in the world; origin, rot: view 0's camera in
    the frame. Fields may have a leading batch dim (one frame per scene)."""

    yaw: torch.Tensor
    centre: torch.Tensor
    scale: torch.Tensor
    origin: torch.Tensor
    rot: torch.Tensor


def scene_frame(cam2world: torch.Tensor, image_size: tuple[int, int], focal_ratio: tuple[float, float]) -> SceneFrame:
    """View 0's cam2world (..., 4, 4) -> its SceneFrame."""
    rot, o = cam2world[..., :3, :3], cam2world[..., :3, 3]
    forward = rot[..., 2]
    yaw = torch.atan2(forward[..., 1], forward[..., 0])
    centre = o - (o[..., 2:] / forward[..., 2:]) * forward
    # R: nearest image-border point cast onto the floor.
    h, w = image_size
    s = torch.linspace(0.0, 1.0, _BORDER_SAMPLES, device=o.device, dtype=o.dtype)
    zero, one = torch.zeros_like(s), torch.ones_like(s)
    u = torch.cat([s, s, zero, one]) * (w - 1)
    v = torch.cat([zero, one, s, s]) * (h - 1)
    d = camera_directions(u, v, image_size, focal_ratio) @ rot.mT
    down = d[..., 2] < -1e-6
    t = -o[..., 2:] / torch.where(down, d[..., 2], -torch.ones_like(u))
    reach = (o[..., None, :2] + t[..., None] * d[..., :2] - centre[..., None, :2]).norm(dim=-1)
    scale = torch.where(down, reach, torch.full_like(reach, float("inf"))).amin(-1)
    to_frame = yaw_matrix(-yaw).to(o.dtype)
    origin = (to_frame @ (o - centre)[..., None])[..., 0] / scale[..., None]
    return SceneFrame(yaw, centre, scale, origin, to_frame @ rot)


def yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation by angle (...) about +z."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack(
        [torch.stack([c, -s, zero], -1), torch.stack([s, c, zero], -1), torch.stack([zero, zero, one], -1)], -2
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
    """Degree-1 SH (..., M, 3, 3) rotated by rot (..., 3, 3) (coefficients as
    w = (-c3, -c1, c2))."""
    c1, c2, c3 = sh1.unbind(-2)
    w = rot.to(sh1.dtype)[..., None, :, :] @ torch.stack([-c3, -c1, c2], -2)
    return torch.stack([-w[..., 1, :], w[..., 2, :], -w[..., 0, :]], -2)


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz quaternion of a rotation by angle (...) about +z."""
    half = 0.5 * angle
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], -1)


def rotate_gaussians(
    gaussians: dict[str, torch.Tensor], rot: torch.Tensor, quat: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Gaussians (..., M, ...) rotated by rot (..., 3, 3) / quat (..., 4)."""
    out = {
        **gaussians,
        "mu": gaussians["mu"] @ rot.to(gaussians["mu"].dtype).mT,
        "quat": quat_mul(
            quat.to(gaussians["quat"].dtype)[..., None, :].expand_as(gaussians["quat"]), gaussians["quat"]
        ),
    }
    if "sh1" in gaussians:
        out["sh1"] = rotate_sh1(gaussians["sh1"], rot)
    return out


def points_to_world(points: torch.Tensor, frame: SceneFrame, rotate: bool = True) -> torch.Tensor:
    """Scene-frame points (B, ..., 3) -> world."""
    flat = points.flatten(1, -2)  # (B, M, 3)
    if rotate:
        flat = flat @ yaw_matrix(frame.yaw).to(points.dtype).mT
    flat = flat * frame.scale.to(points.dtype)[..., None, None] + frame.centre.to(points.dtype)[..., None, :]
    return flat.view(points.shape)


def to_world(gaussians: dict[str, torch.Tensor], frame: SceneFrame, rotate: bool = True) -> dict[str, torch.Tensor]:
    """Gaussians (B, ..., per-Gaussian dims), scene frame -> world."""
    lead = gaussians["mu"].shape[1:-1]  # e.g. (M,) or (slots, G)
    flat = {name: value.flatten(1, len(lead)) for name, value in gaussians.items()}  # (B, M, ...)
    out = rotate_gaussians(flat, yaw_matrix(frame.yaw), yaw_quat(frame.yaw)) if rotate else flat
    out = {
        **out,
        "mu": points_to_world(flat["mu"], frame, rotate),
        "scale": flat["scale"] * frame.scale.to(flat["scale"].dtype)[..., None, None],
    }
    return {name: value.unflatten(1, lead) for name, value in out.items()}


def from_world(gaussians: dict[str, torch.Tensor], frame: SceneFrame) -> dict[str, torch.Tensor]:
    """Gaussians, world -> scene frame (inverse of to_world)."""
    dtype = gaussians["mu"].dtype
    moved = {
        **gaussians,
        "mu": (gaussians["mu"] - frame.centre.to(dtype)[..., None, :]) / frame.scale.to(dtype)[..., None, None],
        "scale": gaussians["scale"] / frame.scale.to(gaussians["scale"].dtype)[..., None, None],
    }
    return rotate_gaussians(moved, yaw_matrix(-frame.yaw), yaw_quat(-frame.yaw))
