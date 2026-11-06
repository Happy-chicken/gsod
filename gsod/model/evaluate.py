"""Evaluation over a split: metrics, per-scene rows, slot table, images.

Metrics (nan without GT masks), averaged per scene:
    seg     ari, fg_ari (2D, input view), nv_ari (3D, novel views)  [uORF]
    recon   psnr, ssim, lpips (novel views; lpips: AlexNet, as uORF)
    diag    attn_fg_ari        attention argmax (background included) vs GT
            obj_recall         GT objects with an owning slot
            n_duplicate        objects split over slots
            n_merged           slots holding several objects
            fg_on_bg           GT background drawn by object slots
            gt_outside_region  object pixels outside its slot's region
            pixel_drawn        object pixels its slot's pixel Gaussians draw

Slot roles: object, duplicate, haze (on background), empty, draws (no GT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from gsod.model.gsod import GSODModel
from gsod.model.losses import ssim
from gsod.utils.metrics import ari, fg_ari, psnr, upsample_labels
from gsod.utils.visualize import Tiles, scene_images

_LPIPS = {}


def _lpips(device: torch.device):
    """The LPIPS (AlexNet) network on device, loaded once; needs `pip install lpips`."""
    if device not in _LPIPS:
        import lpips

        _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[device]

SEGMENTATION_KEYS = ("ari", "fg_ari", "nv_ari")
RECONSTRUCTION_KEYS = ("psnr", "ssim", "lpips")
DIAGNOSTIC_KEYS = (
    "attn_fg_ari",
    "obj_recall",
    "n_duplicate",
    "n_merged",
    "fg_on_bg",
    "gt_outside_region",
    "pixel_drawn",
)
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
# Share of a GT object's pixels a slot must hold to count as holding it.
MERGED_SHARE = 0.3
# Share of a GT object's pixels one object slot must own for it to be found.
OBJECT_FOUND = 0.5
# An object slot owning fewer view-0 pixels than this draws nothing.
MIN_SLOT_PIXELS = 20

NAN = float("nan")


@dataclass
class Evaluation:
    metrics: dict[str, float] = field(default_factory=dict)
    per_scene: list[dict] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    images: dict[str, Tiles] = field(default_factory=dict)


@torch.no_grad()
def evaluate(model: GSODModel, dataset, device: torch.device, n_images: int = 0) -> Evaluation:
    """Every scene once; images of the first n_images."""
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
    """One forward pass, as numpy."""
    images = item["images"].to(device)
    result = model(images, item["cam2world"].to(device))
    rgb = result["rgb"].clamp(0, 1)
    n_views = images.shape[0]
    novel_ssim = [float(ssim(rgb[v : v + 1], images[v : v + 1])) for v in range(1, n_views)]
    novel_lpips = NAN
    if n_views > 1:  # inputs in [-1, 1]
        novel_lpips = float(_lpips(images.device)(2 * rgb[1:] - 1, 2 * images[1:] - 1).mean())

    def np_(x):
        return x.detach().float().cpu().numpy()

    side = model.input_size
    gt = item.get("masks")
    # As in gsod.draw_factor.
    attn = np_(result["attn"])
    count = attn[1:].sum(-1)
    occupancy = (attn[1:] ** 2).sum(-1) / np.maximum(count, 1e-8)
    pixel = result["pixel_opacity"]
    # Pixel Gaussian opacities, scaled by the draw factor as in the render.
    pixel = None if pixel is None else np_(pixel * result["draw"][:, None]).reshape(-1, side, side)
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
        "pixel": pixel,  # (K-1, s, s) or None
        "extent": np_(result["extent"]),  # (K-1, 3)
        "region_box": _project_boxes(
            np_(result["corners"]),
            item["cam2world"][0].numpy(),
            model.focal_ratio,
            tuple(images.shape[-2:]),
        ),  # (K-1, 4) view 0's u0, v0, u1, v1
        "ssim": _nanmean(novel_ssim),
        "lpips": novel_lpips,
        "gt": None if gt is None else gt.cpu().numpy().astype(np.int64),
    }


def scene_metrics(rec: dict, rows: list[dict]) -> dict[str, float]:
    """METRIC_KEYS of one scene."""
    out = dict.fromkeys(METRIC_KEYS, NAN)
    if rec["rgb"].shape[0] > 1:
        out["psnr"] = psnr(float(np.mean((rec["rgb"][1:] - rec["target"][1:]) ** 2)))
        out["ssim"] = rec["ssim"]
        out["lpips"] = rec["lpips"]
    gt = rec["gt"]
    if gt is None:
        return out
    mask = rec["mask_idx"]
    size = mask.shape[1:]
    attn = np.stack([upsample_labels(a, size) for a in rec["attn"]])  # nearest
    bg = gt[0] == 0
    roles = [row["role"] for row in rows]
    objects = [r for r in rows if r["role"] == "object"]
    holds = [
        sum((mask[0][gt[0] == o] == r["slot"]).mean() > MERGED_SHARE for o in np.unique(gt[0]) if o > 0)
        for r in rows
    ]
    drawn = NAN
    if rec["pixel"] is not None:
        pixel = np.stack([upsample_labels(p, size) for p in rec["pixel"]])  # nearest
        drawn = _nanmean([(pixel[r["slot"] - 1][gt[0] == r["gt_object"]] > 0.5).mean() for r in objects])
    out.update(
        {
            "ari": ari(gt[0], mask[0]),
            "fg_ari": fg_ari(gt[0], mask[0]),
            "nv_ari": _nanmean([ari(gt[v], mask[v]) for v in range(1, gt.shape[0])]),
            # Background row included, as in the rendered mask.
            "attn_fg_ari": fg_ari(gt[0], attn.argmax(0)),
            "obj_recall": object_recall(gt[0], mask[0]),
            "n_duplicate": float(roles.count("duplicate")),
            "n_merged": float(sum(h >= 2 for h in holds)),
            "fg_on_bg": float(1.0 - rec["alpha"][0, 0][bg].mean()) if bg.any() else NAN,
            "gt_outside_region": _nanmean([r["gt_outside"] for r in objects]),
            "pixel_drawn": drawn,
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
    """Share of GT objects one slot owns >= OBJECT_FOUND of."""
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
    """World corners (K, 8, 3) -> view-0 pixel boxes (K, 4): u0, v0, u1, v1."""
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
