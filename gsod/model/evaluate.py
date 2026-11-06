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
    index, id and novel-view psnr.

    Keys (nan when undefined, e.g. no GT masks):
      mse, psnr               novel views, all pixels
      psnr_fg, psnr_bg        novel views, GT object / background pixels
      psnr_view0              reference view, all pixels
      ari, fg_ari             novel views, rendered slot mask vs GT
      ari_view0, fg_ari_view0 reference view, rendered slot mask vs GT
      attn_fg_ari             reference view, slot-attention argmax over the
                              object slots vs GT
      attn_render_ari         reference view, slot-attention vs rendered mask
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
        if per_scene is not None:
            mse = float(np.mean((rgb[1:] - target[1:]) ** 2)) if rgb.shape[0] > 1 else math.nan
            psnr = -10.0 * math.log10(max(mse, 1e-10)) if not math.isnan(mse) else math.nan
            per_scene.append({"index": i, "scene": int(item.get("scene", i)), "psnr": psnr})

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
