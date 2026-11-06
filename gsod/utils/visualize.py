"""Image panels of one scene and their W&B forms (train.py, test.py);
the metrics and the slot table are in gsod/model/evaluate.py.

    panels(record, rows)   captioned tiles: input, render, foreground,
                           background, GT / render masks, each slot's
                           contribution, view 0's attention
    save_panels            as PNGs
    scene_images           W&B keys "scene####_view<v>/<panel>"
    wandb_*                W&B images and the slot table

Masks: per-pixel argmax; background dark grey, slot k PALETTE[(k-1) % 8].
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gsod.utils.io import save_png
from gsod.utils.metrics import upsample_labels

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

Tiles = list[tuple[np.ndarray, str]]


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


def scene_images(rec: dict, rows: list[dict], prefix: str = "") -> dict[str, Tiles]:
    """One scene's panels keyed "<prefix>scene####_view<v>/<panel>": W&B
    sections are one level deep (the key up to the slash), so each scene has
    one section per view."""
    out = {}
    for view, named in panels(rec, rows).items():
        section = f"{prefix}scene{rec['scene']:04d}_{view}"
        out.update({f"{section}/{name}": tiles for name, tiles in named.items()})
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
    """The slot table (gsod/model/evaluate.py) as a W&B table, to sort and
    filter in the UI (e.g. role == "haze" and draw > 0.5)."""
    import wandb

    columns = list(rows[0]) if rows else []
    return wandb.Table(columns=columns, data=[[row[c] for c in columns] for row in rows])
