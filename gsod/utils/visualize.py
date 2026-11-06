"""Slot diagnostics, shared by train.py and test.py.

    collect(model, item)       one forward pass, as numpy
    slot_table(record)         one row per object slot
    scene_stats(record, rows)  per-scene statistics
    panels(record, rows)       captioned image tiles
    report(model, dataset, n)  all of them over a dataset
    wandb_*                    their W&B forms (train.py, test.py)

Slot roles: object (mostly on one GT object, the best slot there),
duplicate (on a GT object another slot draws better), haze (on the GT
background), empty (draws ~nothing in view 0), draws (no GT to tell).

Statistics:
    attention   slot_cos_mean, attn_fg_ari, attn_dominance, bg_attn_on_bg,
                occ_object, occ_object_min, occ_other
    segment     ari, fg_ari (input view), nv_ari, nv_fg_ari (novel views)
    render      render_dominance, attn_render_ari, n_duplicate,
                obj_recall_view0, bg_on_objects, fg_on_bg, n_haze
    centres     centre_min_dist
    regions     extent_mean, extent_at_min, extent_at_max (object regions'
                xy half width in units of R and the share on a bound),
                gt_outside_region (an object's GT pixels outside its slot's
                region in view 0)

Masks: per-pixel argmax; background dark grey, slot k PALETTE[(k-1) % 8].
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gsod.utils.io import save_png
from gsod.utils.metrics import ari, fg_ari, upsample_labels

# Mask colours: label / slot 0 (background) dark grey, k >= 1
# PALETTE[(k-1) % 8].
MASK_BG = np.array([55, 55, 55], dtype=np.uint8)
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

OBJECT_FOUND = 0.5  # share of a GT object's pixels one object slot must own
# An object slot owning fewer view-0 pixels than this draws nothing.
MIN_SLOT_PIXELS = 20
# The slot table's columns, in order.
SLOT_COLUMNS = (
    "scene",
    "slot",
    "role",
    "draw",
    "occupancy",
    "count",
    "attn_share",
    "draw_view0",
    "draw_novel",
    "gt_object",
    "iou",
    "on_object",
    "extent_x",
    "extent_y",
    "extent_clamp",
    "gt_outside",
)


# ----------------------------------------------------------------------------
# 1. Collect (the only part that touches torch / the model)
# ----------------------------------------------------------------------------
def collect(model, item: dict, device) -> dict:
    """One forward pass  -> what the diagnostics read, as numpy."""
    import torch

    images = item["images"].to(device)
    with torch.no_grad():
        result = model(images, item["cam2world"].to(device))

    def np_(x):
        return x.detach().float().cpu().numpy()

    side = model.input_size
    gt = item.get("masks")
    g = result["gaussians"]
    # Object slots: count (sum of attention), occupancy (mean share of the
    # tokens held) and draw (the model's draw_factor).
    a = np_(result["attn"])[1:]
    count = a.sum(-1)
    occupancy = (a**2).sum(-1) / np.maximum(count, 1e-8)
    return {
        "scene": int(item.get("scene", -1)),
        "target": np_(images),
        "rgb": np_(result["rgb"].clamp(0, 1)),
        "fg_rgb": np_(result["fg_rgb"].clamp(0, 1)),
        "bg_rgb": np_(result["bg_rgb"].clamp(0, 1)),
        "slot_rgb": np_(result["slot_rgb"].clamp(0, 1)),  # (N, K-1, 3, H, W)
        "alpha": np_(result["slot_alpha"]),  # (N, K, H, W)
        "mask_idx": result["mask_idx"].cpu().numpy(),  # (N, H, W)
        "attn": np_(result["attn"]).reshape(-1, side, side),  # (K, s, s)
        "count": count,  # (K-1,)
        "draw": np_(result["draw"]),  # (K-1,)
        "extent": np_(result["extent"]),  # (K-1, 3)
        "extent_range": model.decoder.extent_range,
        "region_box": _project_boxes(
            np_(result["corners"]), item["cam2world"][0].numpy(), model.focal_ratio,
            tuple(images.shape[-2:]),
        ),  # (K-1, 4) view 0's u0, v0, u1, v1
        "occupancy": occupancy,
        "slots": np_(result["slots"]),  # (K-1, C) view-0 object slots
        "mu": np_(g["mu"]),
        "opacity": np_(g["opacity"]),
        "slot_ids": result["slot_ids"].cpu().numpy(),
        "gt": None if gt is None else gt.cpu().numpy().astype(np.int64),
    }


# ----------------------------------------------------------------------------
# 2. Analyse (numpy only)
# ----------------------------------------------------------------------------
def _project_boxes(corners, cam2world, focal, size) -> np.ndarray:
    """Regions' world corners (K, 8, 3) -> their bounding rectangles in view 0's
    pixels (K, 4): u0, v0, u1, v1 (corners behind the camera ignored)."""
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
        return float("nan")
    inside = (u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])
    return float(1.0 - inside.mean())


def _clamp(extent_xy: np.ndarray, extent_range) -> str:
    """"min" / "max" when either xy half width is within 2% of the range from
    its bound, else ""."""
    lo, hi = extent_range
    tol = 0.02 * (hi - lo)
    if np.any(extent_xy <= lo + tol):
        return "min"
    if np.any(extent_xy >= hi - tol):
        return "max"
    return ""


def _extent_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"extent_mean": float("nan"), "extent_at_min": float("nan"), "extent_at_max": float("nan")}
    return {
        "extent_mean": float(np.mean([(r["extent_x"] + r["extent_y"]) / 2 for r in rows])),
        "extent_at_min": float(np.mean([r["extent_clamp"] == "min" for r in rows])),
        "extent_at_max": float(np.mean([r["extent_clamp"] == "max" for r in rows])),
    }


def slot_table(rec: dict) -> list[dict]:
    """One row per object slot k = 1..K-1 (columns SLOT_COLUMNS)."""
    n_obj = rec["draw"].shape[0]
    gt = rec["gt"]
    mask0 = rec["mask_idx"][0]
    attn = rec["attn"].reshape(n_obj + 1, -1)
    rows = []
    for k in range(1, n_obj + 1):
        pixels = mask0 == k
        row = {
            "scene": rec["scene"],
            "slot": k,
            "role": "empty",
            "draw": float(rec["draw"][k - 1]),
            "occupancy": float(rec["occupancy"][k - 1]),
            "count": float(rec["count"][k - 1]),
            "attn_share": float(attn[k].mean()),
            "draw_view0": float(rec["alpha"][0, k].mean()),
            "draw_novel": float(rec["alpha"][1:, k].mean())
            if rec["alpha"].shape[0] > 1
            else float("nan"),
            "gt_object": -1,
            "iou": float("nan"),
            "on_object": float("nan"),
            "extent_x": float(rec["extent"][k - 1, 0]),
            "extent_y": float(rec["extent"][k - 1, 1]),
            "extent_clamp": _clamp(rec["extent"][k - 1, :2], rec["extent_range"]),
            "gt_outside": float("nan"),
        }
        if pixels.sum() >= MIN_SLOT_PIXELS:
            row["role"] = "draws"
            if gt is not None:
                labels = gt[0][pixels]
                row["on_object"] = float((labels > 0).mean())
                if row["on_object"] >= 0.5:
                    obj = int(np.bincount(labels[labels > 0]).argmax())
                    union = np.logical_or(pixels, gt[0] == obj).sum()
                    row["gt_object"] = obj
                    row["iou"] = float(np.logical_and(pixels, gt[0] == obj).sum() / union)
                    row["gt_outside"] = _outside(gt[0] == obj, rec["region_box"][k - 1])
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


def _dominance(weights: np.ndarray, pixels: np.ndarray) -> float:
    """Mean over `pixels` of max_k w_k / sum_k w_k for weights (K_fg, H, W).

    1.0: every object pixel is owned by one slot. 1/K_fg: all slots share it
    equally (redundant drawing / no separation).
    """
    if weights.shape[0] == 0 or not pixels.any():
        return float("nan")
    w = weights[:, pixels]
    total = w.sum(0)
    keep = total > 1e-6
    if not keep.any():
        return float("nan")
    return float((w[:, keep].max(0) / total[keep]).mean())


def object_recall(gt: np.ndarray, labels: np.ndarray) -> float:
    """Share of the GT objects of which one object slot owns at least
    OBJECT_FOUND of the pixels. Unlike FG-ARI, every object counts the same,
    so small objects that are lost show up."""
    found = []
    for o in np.unique(gt):
        if o == 0:
            continue
        own = labels[gt == o]
        own = own[own > 0]
        best = np.bincount(own).max() if own.size else 0
        found.append(best >= OBJECT_FOUND * (gt == o).sum())
    return float(np.mean(found)) if found else float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def scene_stats(rec: dict, rows: list[dict]) -> dict[str, float]:
    """Per-scene statistics (see the module docstring)."""
    n_slots = rec["alpha"].shape[1]  # K, slot 0 = background
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]
    alpha0 = rec["alpha"][0]
    attn = np.stack([upsample_labels(a, size) for a in rec["attn"]])  # nearest
    # Argmax over the object slots: the background (row 0) can win on object
    # tokens too (see evaluate.attn_labels).
    attn_lab = attn[1:].argmax(0) + 1
    # Object pixels: GT if available, else what the attention gives to objects.
    obj = gt[0] > 0 if gt is not None else attn[0] < 0.5

    z = rec["slots"]
    z_n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    off = (z_n @ z_n.T)[~np.eye(len(z), dtype=bool)]

    # Opacity-weighted centres of the object slots that draw in view 0.
    centres = []
    for k in range(1, n_slots):
        if alpha0[k].mean() <= 0.005:
            continue
        sel = rec["slot_ids"] == k
        w = rec["opacity"][sel]
        centres.append((w[:, None] * rec["mu"][sel]).sum(0) / max(float(w.sum()), 1e-8))
    centre_min_dist = float("nan")
    if len(centres) > 1:
        c = np.stack(centres)
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)
        centre_min_dist = float(d[~np.eye(len(c), dtype=bool)].min())

    roles = [row["role"] for row in rows]
    out = {
        "slot_cos_mean": float(np.mean(off)) if off.size else float("nan"),
        "bg_on_objects": float(alpha0[0][obj].mean()) if obj.any() else float("nan"),
        "attn_dominance": _dominance(attn[1:], obj),
        "render_dominance": _dominance(alpha0[1:], obj),
        # On object pixels: on all pixels the background dominates.
        "attn_render_ari": ari(attn_lab, rec["mask_idx"][0], mask=obj),
        "centre_min_dist": centre_min_dist,
        "n_visible": float((rec["draw"] > 0.5).sum()),
        "n_empty": float(roles.count("empty")),
        # Regions of the slots that draw: xy half width, the share on a bound.
        **_extent_stats([r for r in rows if r["role"] != "empty"]),
    }
    if gt is not None:
        bg = gt[0] == 0
        views = range(1, gt.shape[0])
        nv_ari = [ari(gt[v], rec["mask_idx"][v]) for v in views]
        nv_fg_ari = [fg_ari(gt[v], rec["mask_idx"][v]) for v in views]
        on_object = [r["occupancy"] for r in rows if r["role"] in ("object", "duplicate")]
        out.update(
            {
                "attn_fg_ari": fg_ari(gt[0], attn_lab),
                # uORF's metrics: 2D on the input view, 3D on novel views.
                "ari": ari(gt[0], rec["mask_idx"][0]),
                "fg_ari": fg_ari(gt[0], rec["mask_idx"][0]),
                "nv_ari": _mean([x for x in nv_ari if not np.isnan(x)]),
                "nv_fg_ari": _mean([x for x in nv_fg_ari if not np.isnan(x)]),
                "obj_recall_view0": object_recall(gt[0], rec["mask_idx"][0]),
                # The background's attention on the GT background, and the
                # share of the GT background drawn by object slots (a haze).
                "bg_attn_on_bg": float(attn[0][bg].mean()) if bg.any() else float("nan"),
                "fg_on_bg": float(1.0 - alpha0[0][bg].mean()) if bg.any() else float("nan"),
                "occ_object": _mean(on_object),
                "occ_object_min": float(np.min(on_object)) if on_object else float("nan"),
                "occ_other": _mean([r["occupancy"] for r in rows if r["role"] == "haze"]),
                # GT pixels of an object outside its slot's region (truncation).
                "gt_outside_region": _mean([r["gt_outside"] for r in rows if r["role"] == "object"]),
                "n_object": float(roles.count("object")),
                "n_duplicate": float(roles.count("duplicate")),
                "n_haze": float(roles.count("haze")),
            }
        )
    return out


# ----------------------------------------------------------------------------
# 3. Visualise
# ----------------------------------------------------------------------------
def colorize(labels: np.ndarray, background=MASK_BG) -> np.ndarray:
    """Labels -> RGB: 0 (background) dark gray, k -> PALETTE[(k - 1) % 8]."""
    out = np.empty(labels.shape + (3,), dtype=np.uint8)
    out[...] = background
    for k in np.unique(labels):
        if k == 0:
            continue
        out[labels == k] = PALETTE[(int(k) - 1) % len(PALETTE)]
    return out


def _rgb(chw: np.ndarray) -> np.ndarray:
    return (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)


def _gray(hw: np.ndarray) -> np.ndarray:
    g = (np.clip(hw, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def _slot_caption(row: dict) -> str:
    target = f" #{row['gt_object']}" if row["gt_object"] >= 0 else ""
    return (
        f"s{row['slot']} {row['role']}{target} "
        f"d{row['draw']:.2f} occ{row['occupancy']:.2f}"
    )


Tiles = list[tuple[np.ndarray, str]]


def panels(rec: dict, rows: list[dict]) -> dict[str, dict[str, Tiles]]:
    """One scene's captioned tiles: {view: {panel: tiles}}; panels: images
    (input, render, foreground, background, GT / render masks), slots (each
    slot's contribution), attention (view 0: mask and each row)."""
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]

    def view(v: int) -> dict[str, Tiles]:
        images = [
            (_rgb(rec["target"][v]), "input"),
            (_rgb(rec["rgb"][v]), "render"),
            (_rgb(rec["fg_rgb"][v]), "foreground"),
            (_rgb(rec["bg_rgb"][v]), "background"),
        ]
        if gt is not None:
            images.append((colorize(gt[v]), "GT mask"))
        images.append((colorize(rec["mask_idx"][v]), "render mask"))
        slots = [
            (_rgb(rec["slot_rgb"][v, k - 1]), _slot_caption(row))
            for k, row in zip(range(1, len(rows) + 1), rows)
        ]
        return {"images": images, "slots": slots}

    out = {"view0": view(0)}
    attn_up = np.stack([upsample_labels(a, size) for a in rec["attn"]])
    out["view0"]["attention"] = [(colorize(attn_up.argmax(0)), "attention mask")] + [
        (_gray(a), f"{'bg' if k == 0 else f's{k}'} {float(a.mean()):.0%}")
        for k, a in enumerate(attn_up)
    ]
    if rec["rgb"].shape[0] > 1:
        out["view1"] = view(1)
    return out


def save_panels(view_panels: dict[str, dict[str, Tiles]], scene_dir: str | Path) -> None:
    """Write one scene's panels as PNGs: <scene_dir>/<view>_<panel>_<i>_<caption>.png."""
    for view, named in view_panels.items():
        for name, tiles in named.items():
            for i, (image, caption) in enumerate(tiles):
                slug = re.sub(r"[^A-Za-z0-9.#]+", "_", caption).strip("_")
                save_png(Path(scene_dir) / f"{view}_{name}_{i:02d}_{slug}.png", image)


# ----------------------------------------------------------------------------
# 4. Over a dataset
# ----------------------------------------------------------------------------
@dataclass
class Report:
    """images: W&B keys "scene####_view<v>/<panel>" -> tiles (the first
    n_images scenes); stats: the per-scene statistics averaged; rows: the
    slot table of every scene; per_scene: the statistics of every scene."""

    images: dict[str, Tiles] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    per_scene: list[dict] = field(default_factory=list)


def scene_images(rec: dict, rows: list[dict], prefix: str = "") -> dict[str, Tiles]:
    """One scene's panels keyed "<prefix>scene####_view<v>/<panel>": W&B
    sections are one level deep (the key up to the slash), so each scene has
    one section per view."""
    out = {}
    for view, named in panels(rec, rows).items():
        section = f"{prefix}scene{rec['scene']:04d}_{view}"
        out.update({f"{section}/{name}": tiles for name, tiles in named.items()})
    return out


def summarize(per_scene: list[dict]) -> dict[str, float]:
    if not per_scene:
        return {}
    keys = [k for k, v in per_scene[0].items() if isinstance(v, float)]
    return {
        k: float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
        for k in keys
        for values in [np.array([s.get(k, np.nan) for s in per_scene], dtype=float)]
    }


def report(model, dataset, device, n_images: int) -> Report:
    """Slot table and statistics of every scene of dataset, images of the
    first n_images."""
    was_training = model.training
    model.eval()
    out = Report()
    for i in range(len(dataset)):
        rec = collect(model, dataset[i], device)
        if rec["scene"] < 0:
            rec["scene"] = i
        rows = slot_table(rec)
        stats = scene_stats(rec, rows)
        out.rows.extend(rows)
        out.per_scene.append({"scene": float(rec["scene"]), **stats})
        if i < n_images:
            out.images.update(scene_images(rec, rows))
    out.stats = summarize([{k: v for k, v in s.items() if k != "scene"} for s in out.per_scene])
    model.train(was_training)
    return out


# -- W&B (imported on use: only train.py and test.py need wandb) -------------
def wandb_images(tiles_by_key: dict) -> dict:
    """{key: [(image, caption), ...]} -> {key: [wandb.Image, ...]}."""
    import wandb

    return {
        key: [wandb.Image(image, caption=caption) for image, caption in tiles]
        for key, tiles in tiles_by_key.items()
    }


def wandb_slot_table(rows: list[dict]):
    """The slot table as a W&B table, to sort and filter in the UI (e.g.
    role == "haze" and draw > 0.5)."""
    import wandb

    return wandb.Table(columns=list(SLOT_COLUMNS), data=[[row[c] for c in SLOT_COLUMNS] for row in rows])


def wandb_histograms(rows: list[dict], prefix: str = "slots") -> dict:
    """Occupancy and draw factor of the drawing object slots, by role: how
    objects, duplicates and haze differ."""
    import wandb

    out = {}
    for role in ("object", "duplicate", "haze"):
        chosen = [row for row in rows if row["role"] == role]
        if chosen:
            for name in ("occupancy", "draw"):
                out[f"{prefix}/{name}_{role}"] = wandb.Histogram([row[name] for row in chosen])
    return out
