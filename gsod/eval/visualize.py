"""Image panels of one scene, as PNGs or W&B images."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gsod.eval.metrics import upsample_labels
from gsod.utils.io import save_png

WANDB_PROJECT = "gsod"

# Mask colours: background dark grey, label k PALETTE[(k-1) % 8].
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


# Box corners: 0-3 bottom, 4-7 top.
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def draw_boxes(image: np.ndarray, box: np.ndarray, anchor_init: np.ndarray, cam2world: np.ndarray,
               focal_ratio, slots: list[int]) -> np.ndarray:
    """Image (H, W, 3), world box corners (K-1, 8, 3), anchor_init (K-1, 3) -> the
    image with slot k's box in its mask colour, anchor_init (ring), anchor (dot)."""
    h, w = image.shape[:2]
    anchor = box[:, :4].mean(1, keepdims=True)  # bottom centre
    points = np.concatenate([box, anchor_init[:, None], anchor], 1)  # (K-1, 10, 3)
    # Project (pinhole, pixel centres at integers).
    world2cam = np.linalg.inv(cam2world)
    cam = points @ world2cam[:3, :3].T + world2cam[:3, 3]
    uv = cam[..., :2] / cam[..., 2:] * [focal_ratio[0] * w, focal_ratio[1] * h] + [(w - 1) / 2, (h - 1) / 2]
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for k in slots:
        if (cam[k - 1, :, 2] <= 0).any():  # behind the camera
            continue
        p = [tuple(xy) for xy in uv[k - 1]]
        colour = tuple(int(c) for c in PALETTE[(k - 1) % len(PALETTE)])
        for i, j in BOX_EDGES:
            draw.line([p[i], p[j]], fill=colour)
        (x, y), (a, b) = p[8], p[9]
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=colour)  # anchor_init
        draw.ellipse([a - 2, b - 2, a + 2, b + 2], fill=colour, outline=(255, 255, 255))  # anchor
    return np.asarray(canvas)


def _slot_caption(row: dict) -> str:
    target = f" #{row['gt_object']}" if row["gt_object"] >= 0 else ""
    return (
        f"s{row['slot']} {row['role']}{target} "
        f"occ{row['occupancy']:.2f}"
    )


def panels(rec: dict, rows: list[dict]) -> dict[str, dict[str, Tiles]]:
    """{view: {panel: captioned tiles}}: images, slots, attention."""
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]

    def view(v: int) -> dict[str, Tiles]:
        foreground = _rgb(rec["fg_rgb"][v])
        if "box" in rec:  # boxes of the slots that draw something; ring = anchor_init, dot = anchor
            shown = [row["slot"] for row in rows if row["role"] != "empty"]
            foreground = draw_boxes(foreground, rec["box"], rec["anchor_init"], rec["cam2world"][v],
                                    rec["focal_ratio"], shown)
        images = [
            (_rgb(rec["target"][v]), "input"),
            (_rgb(rec["rgb"][v]), "render"),
            (foreground, "foreground"),
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
    """Panels -> <scene_dir>/<view>_<panel>_<i>_<caption>.png."""
    for view, named in view_panels.items():
        for name, tiles in named.items():
            for i, (image, caption) in enumerate(tiles):
                slug = re.sub(r"[^A-Za-z0-9.#]+", "_", caption).strip("_")
                save_png(Path(scene_dir) / f"{view}_{name}_{i:02d}_{slug}.png", image)


def scene_images(rec: dict, rows: list[dict], prefix: str = "") -> dict[str, Tiles]:
    """Panels keyed "<prefix>scene####_view<v>/<panel>"."""
    out = {}
    for view, named in panels(rec, rows).items():
        section = f"{prefix}scene{rec['scene']:04d}_{view}"
        out.update({f"{section}/{name}": tiles for name, tiles in named.items()})
    return out


def wandb_images(tiles_by_key: dict) -> dict:
    """{key: [(image, caption), ...]} -> {key: [wandb.Image, ...]}."""
    import wandb

    return {
        key: [wandb.Image(image, caption=caption) for image, caption in tiles]
        for key, tiles in tiles_by_key.items()
    }


def wandb_slot_table(rows: list[dict]):
    """Slot table rows -> W&B table."""
    import wandb

    columns = list(rows[0]) if rows else []
    return wandb.Table(columns=columns, data=[[row[c] for c in columns] for row in rows])
