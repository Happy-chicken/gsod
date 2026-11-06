"""Metrics on a split, view 0 (the encoded input) and the novel views
reported separately. GT masks come from item["masks"] ((N, H, W), 0 =
background); without them only reconstruction is scored.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from gsod.data_loader.primordia import PrimordiaScenes
from gsod.model.gsod import GSODModel
from gsod.utils.metrics import SquaredError, ari, fg_ari, upsample_labels

# The headline segmentation metrics (uORF): 2D ari / fg_ari on the input
# view, 3D nv_ari on the novel views.
SEGMENTATION_KEYS = ("ari", "fg_ari", "nv_ari")


def run_model(model: GSODModel, item: dict, device: torch.device) -> dict:
    """One forward pass (deterministic: object slots are seeded from the
    tokens, not sampled)."""
    return model(item["images"].to(device), item["cam2world"].to(device))


def attn_labels(result: dict, model: GSODModel, size: tuple[int, int]) -> np.ndarray:
    """View-0 attention argmax over the object slots (labels 1..K-1),
    upsampled to the render size. The background row is left out: FG-ARI
    only reads object pixels, where it would hide which object slot wins."""
    attn = result["attn"].detach().cpu().numpy()  # (K, s*s)
    side = model.input_size
    labels = (attn[1:].argmax(0) + 1).reshape(side, side)
    return upsample_labels(labels, size)


@torch.no_grad()
def evaluate(
    model: GSODModel,
    dataset: PrimordiaScenes,
    device: torch.device,
    per_scene: list[dict] | None = None,
) -> dict[str, float]:
    """Metrics over every scene; per_scene, if given, gets each scene's
    index, id, novel-view psnr and ari / fg_ari / nv_ari.

    Keys (nan when undefined, e.g. no GT masks). Segmentation follows uORF
    (Yu et al., ICLR 2022), rendered slot masks vs GT, all at render size:
      ari                     2D: input view (view 0), all pixels
      fg_ari                  2D: input view, GT object pixels
      nv_ari                  3D: novel views 1..V-1, all pixels
      nv_fg_ari               novel views, GT object pixels
    Reconstruction:
      mse, psnr               novel views, all pixels
      psnr_fg, psnr_bg        novel views, GT object / background pixels
      psnr_view0              input view, all pixels
    Attention (diagnostics):
      attn_fg_ari             input view, slot-attention argmax over the
                              object slots vs GT
      attn_render_ari         input view, slot-attention vs rendered mask
                              (on GT object pixels)
    """
    was_training = model.training
    model.eval()
    err_novel, err_ref = SquaredError(), SquaredError()
    scores: dict[str, list[float]] = defaultdict(list)

    for i in range(len(dataset)):
        item = dataset[i]
        result = run_model(model, item, device)
        rgb = result["rgb"].clamp(0, 1).cpu().numpy()
        target = item["images"].cpu().numpy()
        pred_mask = result["mask_idx"].cpu().numpy()
        gt = item.get("masks")
        gt = None if gt is None else gt.cpu().numpy().astype(np.int64)
        size = pred_mask.shape[1:]
        attn = attn_labels(result, model, size)

        scene_seg: dict[str, list[float]] = defaultdict(list)
        for v in range(rgb.shape[0]):
            gt_v = None if gt is None else gt[v]
            (err_ref if v == 0 else err_novel).add(rgb[v], target[v], gt_v)
            if gt_v is None:
                continue
            prefix = "" if v == 0 else "nv_"
            scene_seg[prefix + "ari"].append(ari(gt_v, pred_mask[v]))
            scene_seg[prefix + "fg_ari"].append(fg_ari(gt_v, pred_mask[v]))
        # Each scene counts once: its novel views are averaged first.
        seg = {key: _nanmean(values) for key, values in scene_seg.items()}
        for key, value in seg.items():
            scores[key].append(value)
        if gt is not None:
            scores["attn_fg_ari"].append(fg_ari(gt[0], attn))
        # Object pixels only (all pixels without GT): otherwise the shared
        # background makes any two partitions look alike.
        obj = None if gt is None else gt[0] > 0
        scores["attn_render_ari"].append(ari(attn, pred_mask[0], mask=obj))
        if per_scene is not None:
            mse = float(np.mean((rgb[1:] - target[1:]) ** 2)) if rgb.shape[0] > 1 else math.nan
            psnr = -10.0 * math.log10(max(mse, 1e-10)) if not math.isnan(mse) else math.nan
            per_scene.append(
                {
                    "index": i,
                    "scene": int(item.get("scene", i)),
                    "psnr": psnr,
                    **{key: seg.get(key, math.nan) for key in SEGMENTATION_KEYS},
                }
            )

    model.train(was_training)
    out = {
        "mse": err_novel.mse(),
        "psnr": err_novel.psnr(),
        "psnr_fg": err_novel.psnr("fg"),
        "psnr_bg": err_novel.psnr("bg"),
        "psnr_view0": err_ref.psnr(),
    }
    for key in (*SEGMENTATION_KEYS, "nv_fg_ari", "attn_fg_ari", "attn_render_ari"):
        out[key] = _nanmean(scores.get(key, []))
    return out


def _nanmean(values: list[float]) -> float:
    values = [x for x in values if not math.isnan(x)]
    return float(np.mean(values)) if values else float("nan")
