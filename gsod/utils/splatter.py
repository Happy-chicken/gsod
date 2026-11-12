"""gsplat splatter: one set of Gaussians -> rgb, alpha, depth in all views."""

from __future__ import annotations

import torch

# Degree-0 real SH constant (gsplat / 3DGS).
SH_C0 = 0.28209479177387814


class GsplatSplatter:
    """gsplat rasterizer from https://github.com/nerfstudio-project/gsplat."""

    def __init__(
        self,
        far: float = 100.0,
        *,
        packed: bool = False,
        rasterize_mode: str = "classic",
        eps2d: float = 0.3,
        radius_clip: float = 0.0,
    ):
        self.far = float(far)
        self.packed = bool(packed)
        self.rasterize_mode = str(rasterize_mode)
        self.eps2d = float(eps2d)
        self.radius_clip = float(radius_clip)

    def __call__(
        self,
        gaussians: dict[str, torch.Tensor],
        cam2world: torch.Tensor,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
        features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """-> (V, C, H, W) "rgb", "depth" (premultiplied by alpha), "alpha";
        with features (N, D) instead of colours: "features" (V, D, H, W), "alpha"."""
        if not gaussians["mu"].is_cuda:
            raise RuntimeError(
                "GsplatSplatter needs CUDA tensors; this project requires CUDA"
            )

        from gsplat import rasterization

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
        Ks = (
            _intrinsics(w, h, focal_ratio, device)
            .unsqueeze(0)
            .expand(viewmats.shape[0], 3, 3)
            .contiguous()
        )

        rendered, alpha, _ = rasterization(
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
            far_plane=self.far,
            radius_clip=self.radius_clip,
            eps2d=self.eps2d,
            sh_degree=sh_degree,
            packed=self.packed,
            render_mode="RGB+D" if features is None else "RGB",
            rasterize_mode=self.rasterize_mode,
            camera_model="pinhole",
        )

        if features is not None:
            return {"features": rendered.permute(0, 3, 1, 2), "alpha": alpha.permute(0, 3, 1, 2)}
        return {
            "rgb": rendered[..., :3].permute(0, 3, 1, 2),
            "depth": rendered[..., 3:4].permute(0, 3, 1, 2),
            "alpha": alpha.permute(0, 3, 1, 2),
        }


def sh_coefficients(gaussians: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int | None]:
    """gsplat colours: "rgb" (N, 3), or with "sh1" SH coefficients (N, 4, 3)."""
    rgb = gaussians["rgb"].float()
    if "sh1" not in gaussians:
        return rgb.contiguous(), None
    dc = (rgb - 0.5) / SH_C0
    return torch.cat([dc[:, None], gaussians["sh1"].float()], 1).contiguous(), 1


def _intrinsics(
    width: int, height: int, focal_ratio: tuple[float, float], device: torch.device
) -> torch.Tensor:
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
