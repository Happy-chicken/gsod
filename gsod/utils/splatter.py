"""gsplat CUDA splatter backend.

``GSODModel`` only calls a splatter with a fixed interface; ``GsplatSplatter``
wraps the gsplat rasterizer. One call renders one set of Gaussians (one
slot) into all views at once: colour, alpha and depth.
"""

from __future__ import annotations

import torch


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
        channel_chunk: int = 32,
    ):
        self.far = float(far)
        self.packed = bool(packed)
        self.rasterize_mode = str(rasterize_mode)
        self.eps2d = float(eps2d)
        self.radius_clip = float(radius_clip)
        self.channel_chunk = int(channel_chunk)

    def __call__(
        self,
        gaussians: dict[str, torch.Tensor],
        cam2world: torch.Tensor,
        image_size: tuple[int, int],
        focal_ratio: tuple[float, float] = (1.0, 1.0),
        near: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """Returns, per view, (V, C, H, W): "rgb" (C = 3) and "depth" (C = 1),
        both accumulated over the Gaussians (premultiplied by alpha), and
        "alpha" (C = 1)."""
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
        colors = gaussians["rgb"].float().contiguous()

        # gsplat takes the OpenCV world-to-camera matrix, not the transposed
        # world_view_transform used by the original 3DGS rasterizer.
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
            sh_degree=None,
            packed=self.packed,
            # Accumulated depth (premultiplied), not the expected depth.
            render_mode="RGB+D",
            rasterize_mode=self.rasterize_mode,
            channel_chunk=self.channel_chunk,
            camera_model="pinhole",
        )

        return {
            "rgb": rendered[..., :3].permute(0, 3, 1, 2),
            "depth": rendered[..., 3:4].permute(0, 3, 1, 2),
            "alpha": alpha.permute(0, 3, 1, 2),
        }


def _intrinsics(
    width: int, height: int, focal_ratio: tuple[float, float], device: torch.device
) -> torch.Tensor:
    """Per-view pinhole intrinsics, with the same pixel centres as the dataset.

    The dataset ray-caster puts pixel (u, v)'s centre at integer coordinates
    (cx = (W - 1) / 2), while gsplat evaluates pixel (u, v) at (u + 0.5,
    v + 0.5). Shifting the principal point by +0.5 makes both agree.
    """
    fx = float(focal_ratio[0]) * width
    fy = float(focal_ratio[1]) * height
    cx = (width - 1.0) / 2.0 + 0.5
    cy = (height - 1.0) / 2.0 + 0.5
    return torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
