"""gsplat (https://github.com/nerfstudio-project/gsplat, >= 1.5.3 for batches of
scenes): Gaussians rendered as 2D Gaussian surfels (2DGS: flat discs, the ray meets each disc
exactly; the third scale is ignored) -> rgb, alpha, depth in all views, any leading batch dims;
optionally the 2DGS geometry: rendered normals, normals from depth, depth distortion."""

from __future__ import annotations

import torch

# Degree-0 real SH constant (gsplat / 3DGS).
SH_C0 = 0.28209479177387814
FAR = 100.0
EPS2D = 0.3


def splat(
    gaussians: dict[str, torch.Tensor],
    cam2world: torch.Tensor,
    image_size: tuple[int, int],
    focal_ratio: tuple[float, float],
    near: float,
    features: torch.Tensor | None = None,
    geometry: bool = False,
) -> dict[str, torch.Tensor]:
    """Gaussians (..., N, .), cam2world (..., V, 4, 4) -> (..., V, C, H, W)
    "rgb", "depth" (premultiplied by alpha), "alpha"; with features
    (..., N, D) instead of colours: "features" (..., V, D, H, W), "alpha". With geometry also
    "normal" (rendered, alpha-weighted), "depth_normal" (from the expected depth), both world,
    (..., V, 3, H, W), and "distortion" (..., V, 1, H, W)."""
    if not gaussians["mu"].is_cuda:
        raise RuntimeError("splat needs CUDA tensors; this project requires CUDA")

    from gsplat import rasterization_2dgs

    h, w = image_size
    device = gaussians["mu"].device
    mu = gaussians["mu"].float().contiguous()
    scale = gaussians["scale"].float().contiguous()
    quat = gaussians["quat"].float().contiguous()
    opacity = gaussians["opacity"].float().contiguous()
    if features is None:
        colors, sh_degree = sh_coefficients(gaussians)
    else:
        colors, sh_degree = features.float().contiguous(), None

    # gsplat takes OpenCV world-to-camera matrices.
    cam2world = cam2world.to(device=device)
    viewmats = torch.linalg.inv(cam2world.double()).float().contiguous()
    Ks = _intrinsics(w, h, focal_ratio, device).expand(*viewmats.shape[:-2], 3, 3).contiguous()

    rendered, alpha, normal, depth_normal, distortion, _, _ = rasterization_2dgs(
        means=mu,
        quats=quat,
        scales=scale,
        opacities=opacity,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=w,
        height=h,
        near_plane=near,
        far_plane=FAR,
        radius_clip=0.0,
        eps2d=EPS2D,
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED" if features is None else "RGB",
        distloss=geometry,
    )

    rendered, alpha = rendered.movedim(-1, -3), alpha.movedim(-1, -3)
    if features is not None:
        return {"features": rendered, "alpha": alpha}
    out = {"rgb": rendered[..., :3, :, :], "depth": rendered[..., 3:4, :, :] * alpha, "alpha": alpha}
    if geometry:
        out["normal"] = normal.movedim(-1, -3)
        out["depth_normal"] = depth_normal.reshape(normal.shape).movedim(-1, -3)
        out["distortion"] = distortion.movedim(-1, -3)
    return out


def sh_coefficients(gaussians: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int | None]:
    """gsplat colours: "rgb" (..., N, 3), or with "sh1" SH coefficients (..., N, 4, 3)."""
    rgb = gaussians["rgb"].float()
    if "sh1" not in gaussians:
        return rgb.contiguous(), None
    dc = (rgb - 0.5) / SH_C0
    return torch.cat([dc[..., None, :], gaussians["sh1"].float()], -2).contiguous(), 1


def _intrinsics(width: int, height: int, focal_ratio: tuple[float, float], device: torch.device) -> torch.Tensor:
    """Pinhole intrinsics; +0.5 matches gsplat's pixel centres to the dataset's."""
    fx = float(focal_ratio[0]) * width
    fy = float(focal_ratio[1]) * height
    cx = (width - 1.0) / 2.0 + 0.5
    cy = (height - 1.0) / 2.0 + 0.5
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
