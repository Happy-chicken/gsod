"""Evaluation: one pass over a split -> metrics, per-scene rows, the slot
table and image panels. Shared by train.py and test.py.

Metrics (Evaluation.metrics; nan when undefined, e.g. without GT masks),
each scene weighted once (its novel views averaged first):

  Segmentation (uORF, Yu et al. ICLR 2022): the rendered slot mask (argmax
  of the compositing weights) vs GT, at render size
    ari          2D  input view (view 0), all pixels
    fg_ari       2D  input view, GT object pixels
    nv_ari       3D  novel views 1..V-1, all pixels

  Reconstruction, novel views
    psnr, ssim

  Diagnostics: where a low score comes from (input view, need GT)
    attn_fg_ari        slot attention (argmax over the object slots) vs GT:
                       low too -> the grouping fails; high -> the decoding
    obj_recall         share of GT objects of which one slot owns >= half
                       (every object counts the same: lost small objects)
    n_duplicate        slots on an object another slot draws better (splits)
    fg_on_bg           share of the GT background drawn by object slots (haze)
    gt_outside_region  share of an object's pixels outside its slot's
                       region (regions too small)

Slot table (Evaluation.slots, columns SLOT_COLUMNS), one row per object
slot; role: object (the best slot on a GT object), duplicate (on a GT object
another slot draws better), haze (mostly on the GT background), empty (draws
fewer than MIN_SLOT_PIXELS in view 0), draws (no GT to tell).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gsod.model.gsod import GSODModel
from gsod.model.losses import ssim
from gsod.utils.metrics import ari, fg_ari, psnr, upsample_labels
from gsod.utils.visualize import Tiles, scene_images

SEGMENTATION_KEYS = ("ari", "fg_ari", "nv_ari")
RECONSTRUCTION_KEYS = ("psnr", "ssim")
DIAGNOSTIC_KEYS = ("attn_fg_ari", "obj_recall", "n_duplicate", "fg_on_bg", "gt_outside_region")
METRIC_KEYS = SEGMENTATION_KEYS + RECONSTRUCTION_KEYS + DIAGNOSTIC_KEYS

SLOT_COLUMNS = (
    "scene",
    "slot",
    "role",
    "gt_object",
    "iou",
    "draw",
    "occupancy",
    "count",
    "extent_x",
    "extent_y",
    "gt_outside",
)
# Share of a GT object's pixels one object slot must own for it to be found.
OBJECT_FOUND = 0.5
# An object slot owning fewer view-0 pixels than this draws nothing.
MIN_SLOT_PIXELS = 20

NAN = float("nan")


@dataclass
class Evaluation:
    """metrics: METRIC_KEYS averaged over the scenes; per_scene: index,
    scene id and METRIC_KEYS of every scene; slots: the slot table of every
    scene; images: panels of the first n_images scenes (scene_images)."""

    metrics: dict[str, float] = field(default_factory=dict)
    per_scene: list[dict] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    images: dict[str, Tiles] = field(default_factory=dict)


@torch.no_grad()
def evaluate(model: GSODModel, dataset, device: torch.device, n_images: int = 0) -> Evaluation:
    """Every scene of dataset, once (deterministic: object slots are seeded
    from the tokens, not sampled)."""
    was_training = model.training
    model.eval()
    out = Evaluation()
    for i in range(len(dataset)):
        rec = collect(model, dataset[i], device)
        if rec["scene"] < 0:
            rec["scene"] = i
        rows = slot_table(rec)
        out.slots.extend(rows)
        out.per_scene.append({"index": i, "scene": rec["scene"], **scene_metrics(rec, rows)})
        if i < n_images:
            out.images.update(scene_images(rec, rows))
    out.metrics = {key: _nanmean([s[key] for s in out.per_scene]) for key in METRIC_KEYS}
    model.train(was_training)
    return out


# -- one scene ----------------------------------------------------------------
@torch.no_grad()
def collect(model: GSODModel, item: dict, device: torch.device) -> dict:
    """One forward pass -> what the metrics and panels read, as numpy."""
    images = item["images"].to(device)
    result = model(images, item["cam2world"].to(device))
    rgb = result["rgb"].clamp(0, 1)
    n_views = images.shape[0]
    novel_ssim = [float(ssim(rgb[v : v + 1], images[v : v + 1])) for v in range(1, n_views)]

    def np_(x):
        return x.detach().float().cpu().numpy()

    side = model.input_size
    gt = item.get("masks")
    # Object slots: count (sum of attention), occupancy (mean share of the
    # tokens held), as in gsod.draw_factor.
    attn = np_(result["attn"])
    count = attn[1:].sum(-1)
    occupancy = (attn[1:] ** 2).sum(-1) / np.maximum(count, 1e-8)
    return {
        "scene": int(item.get("scene", -1)),
        "target": np_(images),
        "rgb": np_(rgb),
        "fg_rgb": np_(result["fg_rgb"].clamp(0, 1)),
        "bg_rgb": np_(result["bg_rgb"].clamp(0, 1)),
        "slot_rgb": np_(result["slot_rgb"].clamp(0, 1)),  # (V, K-1, 3, H, W)
        "alpha": np_(result["slot_alpha"]),  # (V, K, H, W), 0 = uncovered
        "mask_idx": result["mask_idx"].cpu().numpy(),  # (V, H, W)
        "attn": attn.reshape(-1, side, side),  # (K, s, s), row 0 = background
        "count": count,
        "occupancy": occupancy,
        "draw": np_(result["draw"]),  # (K-1,)
        "extent": np_(result["extent"]),  # (K-1, 3)
        "region_box": _project_boxes(
            np_(result["corners"]),
            item["cam2world"][0].numpy(),
            model.focal_ratio,
            tuple(images.shape[-2:]),
        ),  # (K-1, 4) view 0's u0, v0, u1, v1
        "ssim": _nanmean(novel_ssim),
        "gt": None if gt is None else gt.cpu().numpy().astype(np.int64),
    }


def scene_metrics(rec: dict, rows: list[dict]) -> dict[str, float]:
    """METRIC_KEYS of one scene (see the module docstring)."""
    out = dict.fromkeys(METRIC_KEYS, NAN)
    if rec["rgb"].shape[0] > 1:
        out["psnr"] = psnr(float(np.mean((rec["rgb"][1:] - rec["target"][1:]) ** 2)))
        out["ssim"] = rec["ssim"]
    gt = rec["gt"]
    if gt is None:
        return out
    mask = rec["mask_idx"]
    size = mask.shape[1:]
    attn = np.stack([upsample_labels(a, size) for a in rec["attn"]])  # nearest
    bg = gt[0] == 0
    roles = [row["role"] for row in rows]
    out.update(
        {
            "ari": ari(gt[0], mask[0]),
            "fg_ari": fg_ari(gt[0], mask[0]),
            "nv_ari": _nanmean([ari(gt[v], mask[v]) for v in range(1, gt.shape[0])]),
            # Object slots only: the background row can win on object tokens.
            "attn_fg_ari": fg_ari(gt[0], attn[1:].argmax(0) + 1),
            "obj_recall": object_recall(gt[0], mask[0]),
            "n_duplicate": float(roles.count("duplicate")),
            "fg_on_bg": float(1.0 - rec["alpha"][0, 0][bg].mean()) if bg.any() else NAN,
            "gt_outside_region": _nanmean([r["gt_outside"] for r in rows if r["role"] == "object"]),
        }
    )
    return out


def slot_table(rec: dict) -> list[dict]:
    """One row per object slot k = 1..K-1 (columns SLOT_COLUMNS)."""
    gt = rec["gt"]
    mask0 = rec["mask_idx"][0]
    rows = []
    for k in range(1, rec["draw"].shape[0] + 1):
        pixels = mask0 == k
        row = {
            "scene": rec["scene"],
            "slot": k,
            "role": "empty",
            "gt_object": -1,
            "iou": NAN,
            "draw": float(rec["draw"][k - 1]),
            "occupancy": float(rec["occupancy"][k - 1]),
            "count": float(rec["count"][k - 1]),
            "extent_x": float(rec["extent"][k - 1, 0]),
            "extent_y": float(rec["extent"][k - 1, 1]),
            "gt_outside": NAN,
        }
        if pixels.sum() >= MIN_SLOT_PIXELS:
            row["role"] = "draws"
            if gt is not None:
                labels = gt[0][pixels]
                if (labels > 0).mean() >= 0.5:
                    obj = int(np.bincount(labels[labels > 0]).argmax())
                    target = gt[0] == obj
                    row["gt_object"] = obj
                    row["iou"] = float((pixels & target).sum() / (pixels | target).sum())
                    row["gt_outside"] = _outside(target, rec["region_box"][k - 1])
                    row["role"] = "object"
                else:
                    row["role"] = "haze"
        rows.append(row)
    # Of several slots on the same GT object, all but the best are duplicates.
    best: dict[int, dict] = {}
    for row in rows:
        if row["role"] != "object":
            continue
        other = best.get(row["gt_object"])
        if other is None or row["iou"] > other["iou"]:
            if other is not None:
                other["role"] = "duplicate"
            best[row["gt_object"]] = row
        else:
            row["role"] = "duplicate"
    return rows


def object_recall(gt: np.ndarray, labels: np.ndarray) -> float:
    """Share of the GT objects of which one object slot owns at least
    OBJECT_FOUND of the pixels."""
    found = []
    for o in np.unique(gt):
        if o == 0:
            continue
        own = labels[gt == o]
        own = own[own > 0]
        best = np.bincount(own).max() if own.size else 0
        found.append(best >= OBJECT_FOUND * (gt == o).sum())
    return float(np.mean(found)) if found else NAN


def _project_boxes(corners, cam2world, focal, size) -> np.ndarray:
    """Regions' world corners (K, 8, 3) -> their bounding rectangles in view
    0's pixels (K, 4): u0, v0, u1, v1."""
    h, w = size
    c = (corners - cam2world[:3, 3]) @ cam2world[:3, :3]
    z = np.maximum(c[..., 2], 1e-6)
    u = focal[0] * w * c[..., 0] / z + (w - 1) / 2
    v = focal[1] * h * c[..., 1] / z + (h - 1) / 2
    return np.stack([u.min(-1), v.min(-1), u.max(-1), v.max(-1)], -1)


def _outside(mask: np.ndarray, box: np.ndarray) -> float:
    """Share of a mask's pixels outside a pixel rectangle."""
    v, u = np.nonzero(mask)
    if not len(u):
        return NAN
    inside = (u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])
    return float(1.0 - inside.mean())


def _nanmean(values) -> float:
    values = [float(x) for x in values if not np.isnan(x)]
    return float(np.mean(values)) if values else NAN
