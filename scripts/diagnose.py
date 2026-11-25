#!/usr/bin/env python
"""Where does the blur come from? Two checks on a trained checkpoint, no training.

1. Gaussian statistics: per object slot, Gaussians with opacity > 0.1, their size in view-0
   pixels, the object's area in view 0.
2. Optimisation ceiling: the decoded Gaussians optimised directly per scene (no network) on all
   views, and on view 0 only; PSNR per view before / after, images saved.

    python scripts/diagnose.py --dataset room_diverse --checkpoint checkpoints/<run> --suffix best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsod.configs import DATASETS, config_for, run_config  # noqa: E402
from gsod.data_loader import make_dataset  # noqa: E402
from gsod.model import load_model  # noqa: E402
from gsod.model.losses import ssim  # noqa: E402
from gsod.utils.io import save_png  # noqa: E402
from gsod.utils.splatter import splat  # noqa: E402

ACTIVE = 0.1  # opacity of a Gaussian that draws
MIN_AREA = 20  # view-0 pixels of a slot that draws an object
# Optimisation: steps, learning rate per parameter (positions: times the scene's spread).
STEPS = 500
LR = {"mu": 1e-3, "log_scale": 5e-3, "quat": 1e-3, "opacity": 5e-2, "rgb": 1e-2, "sh1": 2.5e-3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--checkpoint", required=True, help="run directory")
    parser.add_argument("--suffix", default="best")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--scenes", type=int, default=50, help="scenes for the statistics")
    parser.add_argument("--optimize-scenes", type=int, default=5, help="scenes for the optimisation")
    return parser.parse_args()


def psnr(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Images (V, 3, H, W) in [0, 1] -> PSNR per view."""
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse)).tolist()


def gaussian_stats(model, outputs, cam2world: torch.Tensor) -> list[dict]:
    """One scene -> per drawing object slot: active Gaussians, their view-0 size (px), area (px)."""
    objects = outputs["gaussians"]["objects"]  # world, (1, K-1, G, ...)
    w = model.image_size[1]
    world2cam = torch.linalg.inv(cam2world[0])  # view 0
    cam = objects["mu"][0] @ world2cam[:3, :3].T + world2cam[:3, 3]  # (K-1, G, 3)
    size = model.focal_ratio[0] * w * objects["scale"][0].mean(-1) / cam[..., 2].clamp_min(1e-4)
    mask = outputs["mask_idx"][0, 0]  # view 0, 0 = background
    rows = []
    for k in range(objects["mu"].shape[1]):
        area = int((mask == k + 1).sum())
        if area < MIN_AREA:
            continue
        active = objects["opacity"][0, k] > ACTIVE
        rows.append(
            {
                "active": int(active.sum()),
                "size_px": float(size[k][active].mean()) if active.any() else float("nan"),
                "area_px": area,
            }
        )
    return rows


def optimise(model, gaussians: dict, images: torch.Tensor, cam2world: torch.Tensor, views: list[int]):
    """Every Gaussian (1, N, ...) optimised on the given views -> render of all views (V, 3, H, W)."""
    spread = gaussians["mu"].std().item()
    params = {
        "mu": gaussians["mu"].detach().clone(),
        "log_scale": gaussians["scale"].detach().clamp_min(1e-6).log(),
        "quat": gaussians["quat"].detach().clone(),
        "opacity": torch.logit(gaussians["opacity"].detach().clamp(1e-4, 1 - 1e-4)),
        "rgb": torch.logit(gaussians["rgb"].detach().clamp(1e-4, 1 - 1e-4)),
        "sh1": gaussians["sh1"].detach().clone(),
    }
    for p in params.values():
        p.requires_grad_()
    opt = torch.optim.Adam([{"params": [p], "lr": LR[k] * (spread if k == "mu" else 1.0)} for k, p in params.items()])
    cams = (model.image_size, model.focal_ratio, model.near)

    def render(cam: torch.Tensor) -> torch.Tensor:
        g = {
            "mu": params["mu"],
            "scale": params["log_scale"].exp(),
            "quat": F.normalize(params["quat"], dim=-1),
            "opacity": torch.sigmoid(params["opacity"]),
            "rgb": torch.sigmoid(params["rgb"]),
            "sh1": params["sh1"],
        }
        return splat(g, cam[None], *cams)["rgb"][0]

    target = images[views]
    for _ in range(STEPS):
        out = render(cam2world[views])
        loss = F.mse_loss(out, target) + 0.2 * (1 - ssim(out, target))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return render(cam2world).clamp(0, 1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("needs CUDA (gsplat)")
    device = torch.device("cuda")
    cfg = run_config(config_for(args.dataset), Path(args.checkpoint))
    root = args.data or (cfg.test_root if args.split == "test" and cfg.test_root else cfg.data_root)
    model = load_model(cfg, args.checkpoint, args.suffix, device).eval()
    dataset = make_dataset(cfg, root, args.split)
    out_dir = Path(args.checkpoint) / "diagnose" / f"{args.split}_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gaussian statistics.
    rows = []
    for i in range(min(args.scenes, len(dataset))):
        item = dataset[i]
        images, cams = item["images"].to(device), item["cam2world"].to(device)
        with torch.no_grad():
            rows += gaussian_stats(model, model(images[None], cams[None]), cams)
    active = np.array([r["active"] for r in rows])
    size = np.array([r["size_px"] for r in rows])
    area = np.array([r["area_px"] for r in rows])
    print(f"1. Gaussians per drawing object slot ({len(rows)} slots, {min(args.scenes, len(dataset))} scenes)")
    print(f"   active (opacity > {ACTIVE}): mean {active.mean():.0f}, median {np.median(active):.0f}")
    print(f"   size in view 0 (px):        mean {np.nanmean(size):.1f}, median {np.nanmedian(size):.1f}")
    print(f"   object area in view 0 (px): mean {area.mean():.0f}")
    print(f"   area per active Gaussian:   median {np.median(area / np.maximum(active, 1)):.1f} px")

    # 2. Optimisation ceiling.
    print(f"2. Optimised directly ({STEPS} steps); PSNR per view [view 0, novel views...]")
    for i in range(min(args.optimize_scenes, len(dataset))):
        item = dataset[i]
        images, cams = item["images"].to(device), item["cam2world"].to(device)
        with torch.no_grad():
            outputs = model(images[None], cams[None])
        g = outputs["gaussians"]
        every = model.merge(g["objects"], g["background"])
        before = outputs["rgb"][0].clamp(0, 1)
        all_views = optimise(model, every, images, cams, list(range(images.shape[0])))
        view0 = optimise(model, every, images, cams, [0])
        fmt = lambda x: "[" + ", ".join(f"{v:.1f}" for v in x) + "]"  # noqa: E731
        print(f"   scene {i}: model {fmt(psnr(before, images))}")
        print(f"            all views {fmt(psnr(all_views, images))}   view 0 only {fmt(psnr(view0, images))}")
        rows_img = [images, before, all_views, view0]  # rows: input, model, optimised (all views / view 0)
        grid = torch.cat([torch.cat(list(r), -1) for r in rows_img], -2)  # (3, 4H, VW)
        save_png(out_dir / f"scene{i:04d}.png", (grid.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    print(f"   images (rows: input, model, optimised on all views, on view 0 only): {out_dir}")


if __name__ == "__main__":
    main()
