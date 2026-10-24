"""Evaluation and rendering on a Primordia split.

Both use view 0 of each scene as the encoded reference. Metrics are reported
separately for view 0 (2D decomposition of the input) and for the remaining
views, which the encoder never sees (3D-consistent decomposition).

Ground-truth masks are read from item["masks"]: an (N, H, W) integer tensor,
0 = background, i = object i, at the same resolution as item["images"]. If the
loader does not provide them, only the reconstruction metrics are computed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gsod.data_loader.primordia import PrimordiaScenes
from gsod.model.gsod import GSODModel
from gsod.utils.metrics import SquaredError, ari, fg_ari, upsample_labels
from gsod.utils.png import save_png

# Full masks show slot 0 as dark gray; foreground-only masks show it as black.
MASK_BG = np.array([55, 55, 55], dtype=np.uint8)
# Slot k >= 1 gets palette[(k - 1) % len(palette)].
PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)


def run_model(model: GSODModel, item: dict, device: torch.device, seed: int) -> dict:
    """One forward pass with a fixed slot-initialisation seed.

    Slot attention samples its initial slots, so without a fixed seed two
    evaluations of the same checkpoint give different numbers. The global RNG
    state is restored afterwards so training is not affected.
    """
    images, cams = item["images"].to(device), item["cam2world"].to(device)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model(images, cams)


def attn_labels(result: dict, model: GSODModel, size: tuple[int, int]) -> np.ndarray:
    """Slot-attention argmax of view 0, upsampled to the render size."""
    attn = result["attn"].detach().cpu().numpy()  # (K, s*s)
    side = model.input_size
    labels = attn.argmax(0).reshape(side, side)
    return upsample_labels(labels, size)


@torch.no_grad()
def evaluate(
    model: GSODModel, dataset: PrimordiaScenes, device: torch.device, seed: int = 0
) -> dict[str, float]:
    """Reconstruction and segmentation metrics over every scene in dataset.

    Keys (nan when undefined, e.g. no GT masks):
      mse, psnr               novel views, all pixels (as before)
      psnr_fg, psnr_bg        novel views, GT object / background pixels
      psnr_view0              reference view, all pixels
      ari, fg_ari             novel views, rendered slot mask vs GT
      ari_view0, fg_ari_view0 reference view, rendered slot mask vs GT
      attn_fg_ari             reference view, slot-attention argmax vs GT
      attn_render_ari         reference view, slot-attention vs rendered mask
                              (on GT object pixels)
    """
    was_training = model.training
    model.eval()
    err_novel, err_ref = SquaredError(), SquaredError()
    scores: dict[str, list[float]] = defaultdict(list)

    for i in range(len(dataset)):
        item = dataset[i]
        result = run_model(model, item, device, seed + i)
        rgb = result["rgb"].clamp(0, 1).cpu().numpy()
        target = item["images"].cpu().numpy()
        pred_mask = result["mask_idx"].cpu().numpy()
        gt = item.get("masks")
        gt = None if gt is None else gt.cpu().numpy().astype(np.int64)
        size = pred_mask.shape[1:]
        attn = attn_labels(result, model, size)

        for v in range(rgb.shape[0]):
            gt_v = None if gt is None else gt[v]
            (err_ref if v == 0 else err_novel).add(rgb[v], target[v], gt_v)
            if gt_v is None:
                continue
            suffix = "_view0" if v == 0 else ""
            scores["ari" + suffix].append(ari(gt_v, pred_mask[v]))
            scores["fg_ari" + suffix].append(fg_ari(gt_v, pred_mask[v]))
        if gt is not None:
            scores["attn_fg_ari"].append(fg_ari(gt[0], attn))
        # Object pixels only (all pixels without GT): otherwise the shared
        # background makes any two partitions look alike.
        obj = None if gt is None else gt[0] > 0
        scores["attn_render_ari"].append(ari(attn, pred_mask[0], mask=obj))

    model.train(was_training)
    out = {
        "mse": err_novel.mse(),
        "psnr": err_novel.psnr(),
        "psnr_fg": err_novel.psnr("fg"),
        "psnr_bg": err_novel.psnr("bg"),
        "psnr_view0": err_ref.psnr(),
    }
    for key in (
        "ari",
        "fg_ari",
        "ari_view0",
        "fg_ari_view0",
        "attn_fg_ari",
        "attn_render_ari",
    ):
        values = [x for x in scores.get(key, []) if not math.isnan(x)]
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


@torch.no_grad()
def render_scenes(
    model: GSODModel,
    dataset: PrimordiaScenes,
    device: torch.device,
    out_dir: str | Path,
    seed: int = 0,
) -> None:
    """Write target, rendered, and mask PNGs for every view of every scene.

    Per view:
      view{v}_target.png / view{v}_render.png
      view{v}_mask.png      full slot partition; background slot dark gray
      view{v}_fg_mask.png   foreground slots only; background black
      view{v}_gt_mask.png   ground truth (if the loader provides masks)
      view{v}_slot{k}.png   alpha of foreground slot k, grayscale (white = 1)
    Per scene:
      attn_mask.png         slot-attention argmax on view 0 (same colours)
    """
    out = Path(out_dir)
    model.eval()
    for i in range(len(dataset)):
        item = dataset[i]
        result = run_model(model, item, device, seed + i)
        alpha = result["slot_alpha"]
        mask_idx = result["mask_idx"].cpu().numpy()
        gt = item.get("masks")
        scene_dir = out / f"sc{item['scene']:04d}"
        for v in range(mask_idx.shape[0]):
            save_png(scene_dir / f"view{v}_target.png", _to_uint8(item["images"][v]))
            save_png(scene_dir / f"view{v}_render.png", _to_uint8(result["rgb"][v]))
            save_png(scene_dir / f"view{v}_mask.png", _colorize(mask_idx[v], MASK_BG))
            save_png(scene_dir / f"view{v}_fg_mask.png", _colorize(mask_idx[v]))
            if gt is not None:
                save_png(
                    scene_dir / f"view{v}_gt_mask.png",
                    _colorize(gt[v].cpu().numpy().astype(np.int64), MASK_BG),
                )
            for k in range(1, alpha.shape[1]):
                gray = (alpha[v, k].clamp(0.0, 1.0) * 255.0 + 0.5).cpu().numpy()
                save_png(scene_dir / f"view{v}_slot{k}.png", gray.astype(np.uint8))
        attn = attn_labels(result, model, mask_idx.shape[1:])
        save_png(scene_dir / "attn_mask.png", _colorize(attn, MASK_BG))


def _to_uint8(chw: torch.Tensor) -> np.ndarray:
    hwc = chw.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (hwc * 255.0 + 0.5).astype(np.uint8)


def _colorize(mask_idx: np.ndarray, background=(0, 0, 0)) -> np.ndarray:
    out = np.empty(mask_idx.shape + (3,), dtype=np.uint8)
    out[...] = background
    for k in np.unique(mask_idx):
        if k == 0:
            continue
        out[mask_idx == k] = PALETTE[(int(k) - 1) % len(PALETTE)]
    return out
