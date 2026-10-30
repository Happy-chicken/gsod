"""Slot diagnostics for W&B: per-scene sheets and averaged statistics.

train.py logs them at every validation, test.py once on the test split
(``sheets``). Per scene and on average they answer:

1. Initialisation   -- are the Gaussians of a slot spread out, or one point?
2. Slot attention   -- are the slot latents distinct? Does one slot take the
                       whole image? Does the attention map separate objects
                       (attn FG-ARI, dominance)?
3. Decoder          -- does the rendered slot partition follow the attention,
                       or do several slots redundantly draw the same object
                       (render dominance, attn/render ARI)? Is every object
                       owned by some slot (object recall)?
4. Slot centres     -- are the centres of the drawing slots' Gaussians
                       distinct?

A sheet (make_grid) has a title with the scene's key scores and one block
per view: view 0 (images and masks; slot attention; rendered slot weights),
then view 1 (images and masks; rendered slot weights). Each slot tile has
its mask colour and share of the image. Needs Pillow.
"""

from __future__ import annotations

import numpy as np

from gsod.utils.metrics import ari, fg_ari, upsample_labels

# Same colours as evaluate.py: label / slot 0 (background) dark gray,
# k >= 1 PALETTE[(k-1) % 8].
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
ACTIVE_OPACITY = 0.05


# ----------------------------------------------------------------------------
# 1. Collect (the only part that touches torch / the model)
# ----------------------------------------------------------------------------
def collect(model, item: dict, device, seed: int = 0) -> dict[str, np.ndarray]:
    """Run the model on one scene and return everything as numpy arrays."""
    import torch

    images = item["images"].to(device)
    cams = item["cam2world"].to(device)
    devices = [device] if getattr(device, "type", "") == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        result = model(images, cams)

    def np_(x):
        return None if x is None else x.detach().float().cpu().numpy()

    side = model.input_size
    gt = item.get("masks")
    g = result["gaussians"]
    return {
        "target": np_(images),
        "rgb": np_(result["rgb"].clamp(0, 1)),
        "alpha": np_(result["slot_alpha"]),  # (N, K, H, W)
        "mask_idx": result["mask_idx"].cpu().numpy(),  # (N, H, W)
        "attn": np_(result["attn"]).reshape(-1, side, side),  # (K, s, s)
        "slots": np_(result["slots"]),  # (K-1, C) view-0 object slots
        "mu": np_(g["mu"]),
        "scale": np_(g["scale"]),
        "opacity": np_(g["opacity"]),
        "slot_ids": result["slot_ids"].cpu().numpy(),
        "gt": None if gt is None else gt.cpu().numpy().astype(np.int64),
    }


# ----------------------------------------------------------------------------
# 2. Analyse (numpy only)
# ----------------------------------------------------------------------------
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


def analyse(rec: dict) -> dict:
    n_slots = rec["alpha"].shape[1]  # K, slot 0 = background
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]
    alpha0 = rec["alpha"][0]
    attn = np.stack([upsample_labels(a, size) for a in rec["attn"]])  # nearest
    # Argmax over the object slots: the empty option (row 0) wins on object
    # tokens too (see evaluate.attn_labels).
    attn_lab = attn[1:].argmax(0) + 1

    # Object pixels: GT if available, else what the attention gives to objects.
    obj = gt[0] > 0 if gt is not None else attn[0] < 0.5

    z = rec["slots"]  # object slots
    z_n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    cos = z_n @ z_n.T
    off = cos[~np.eye(len(z), dtype=bool)]

    # Gaussians per slot; centre weighted by opacity.
    per_slot, centres = [], []
    for k in range(n_slots):
        sel = rec["slot_ids"] == k
        mu = rec["mu"][sel]
        w = rec["opacity"][sel]
        centres.append((w[:, None] * mu).sum(0) / max(float(w.sum()), 1e-8))
        # Slots without Gaussians (the background is not decoded): nan.
        if sel.any():
            gauss = {
                "spread": float(np.linalg.norm(mu - mu.mean(0), axis=1).mean()),
                "mean_scale": float(rec["scale"][sel].mean()),
                "mean_opacity": float(w.mean()),
                "active_frac": float((w > ACTIVE_OPACITY).mean()),
            }
        else:
            gauss = dict.fromkeys(
                ("spread", "mean_scale", "mean_opacity", "active_frac"), float("nan")
            )
        per_slot.append(
            {
                "slot": k,
                **gauss,
                "coverage_view0": float(alpha0[k].mean()),
                "render_share_view0": float((rec["mask_idx"][0] == k).mean()),
                "attn_share": float(rec["attn"][k].mean()),
            }
        )

    # Centres of the object slots that actually draw something.
    centre_min_dist = float("nan")
    active = [k for k in range(1, n_slots) if per_slot[k]["coverage_view0"] > 0.005]
    if len(active) > 1:
        c = np.stack([centres[k] for k in active])
        d = np.linalg.norm(c[:, None] - c[None], axis=-1)
        centre_min_dist = float(d[~np.eye(len(c), dtype=bool)].min())

    out = {
        "slot_cos_mean": float(np.mean(off)),
        "slot_cos_max": float(np.max(off)),
        "attn_obj_max_share": float(max(p["attn_share"] for p in per_slot[1:])),
        "bg_on_objects": float(alpha0[0][obj].mean()) if obj.any() else float("nan"),
        "attn_dominance": _dominance(attn[1:], obj),
        "render_dominance": _dominance(alpha0[1:], obj),
        # On object pixels: on all pixels the background slot dominates.
        "attn_render_ari": ari(attn_lab, rec["mask_idx"][0], mask=obj),
        "centre_min_dist": centre_min_dist,
        "per_slot": per_slot,
    }
    if gt is not None:
        out["attn_fg_ari"] = fg_ari(gt[0], attn_lab)
        out["render_fg_ari_view0"] = fg_ari(gt[0], rec["mask_idx"][0])
        out["obj_recall_view0"] = object_recall(gt[0], rec["mask_idx"][0])
        novel = [fg_ari(gt[v], rec["mask_idx"][v]) for v in range(1, gt.shape[0])]
        novel = [x for x in novel if not np.isnan(x)]
        out["render_fg_ari_novel"] = float(np.mean(novel)) if novel else float("nan")
    return out


# ----------------------------------------------------------------------------
# 3. Visualise
# ----------------------------------------------------------------------------
def _colorize(labels: np.ndarray, background=MASK_BG) -> np.ndarray:
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


def _up(maps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(C, h, w) -> (C, H, W), nearest."""
    return np.stack([upsample_labels(m, size) for m in maps])


def _gray(hw: np.ndarray) -> np.ndarray:
    g = (np.clip(hw, 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


_TEXT = {
    "view0": "View 0",
    "view1": "View 1",
    "row_images": "images (masks: argmax slot per pixel)",
    "row_attn": "slot attention",
    "row_alpha": "rendered slot weight",
    "image": "input image",
    "render": "render",
    "gt": "GT mask",
    "mask": "render mask",
    "attn_mask": "attention (objects)",
    "slot": "{slot} {share:.0%}",
}
_FONTS = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "Arial.ttf",
)


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _txt(key: str, **fmt) -> str:
    return _TEXT[key].format(**fmt)


def _slot_name(k: int) -> str:
    return "bg" if k == 0 else f"s{k}"


def _slot_color(k: int) -> np.ndarray:
    return MASK_BG if k == 0 else PALETTE[(k - 1) % len(PALETTE)]


def make_grid(
    rec: dict,
    pad: int = 4,
    title: str | None = None,
    min_tile: int = 128,
) -> np.ndarray:
    """Diagnostic sheet: one block per view, a caption under every tile.

    View 0  images: input, render, GT mask, render mask, attention argmax
            over the object slots
            slot attention per slot (bg = the empty option)
            rendered weight per slot
    View 1  images: input, render, GT mask, render mask (novel view)
            rendered weight per slot
    View 0 is the encoded input; view 1 is a novel view the encoder never
    sees, so it tests the 3D. Slot attention: share of each token given to
    the slot. Rendered slot weight: the slot's compositing weight per pixel
    (bg: the share no object covers). Grey tiles: white = 1 (all of
    the token / pixel), black = 0. Slot captions carry the slot's mask colour
    and its share of the image (the tile mean), so empty slots read 0%.
    Captions are English.
    """
    from PIL import Image, ImageDraw

    n_views = rec["rgb"].shape[0]
    size = rec["mask_idx"].shape[1:]
    blank = np.zeros(size + (3,), dtype=np.uint8)
    attn_up = np.stack([upsample_labels(a, size) for a in rec["attn"]])
    gt = rec["gt"]

    scale = max(1, min_tile // max(size))
    tile_h, tile_w = size[0] * scale, size[1] * scale
    font = _load_font(max(10, tile_w // 10))
    cap_h = font.size + 8 if hasattr(font, "size") else 16
    head_h = cap_h

    def image_tiles(v: int) -> list:
        return [
            (_rgb(rec["target"][v]), _txt("image"), None),
            (_rgb(rec["rgb"][v]), _txt("render"), None),
            (_colorize(gt[v]) if gt is not None else blank, _txt("gt"), None),
            (_colorize(rec["mask_idx"][v]), _txt("mask"), None),
        ]

    def slot_tiles(maps: np.ndarray) -> list:
        return [
            (_gray(m), _txt("slot", slot=_slot_name(k), share=float(m.mean())), k)
            for k, m in enumerate(maps)
        ]

    # Blocks of (view heading, [(row heading, [(image, caption, slot or None)])]).
    blocks = [
        (
            _txt("view0"),
            [
                (
                    _txt("row_images"),
                    image_tiles(0)
                    + [
                        (_colorize(attn_up[1:].argmax(0) + 1), _txt("attn_mask"), None),
                    ],
                ),
                (_txt("row_attn"), slot_tiles(rec["attn"])),
                (_txt("row_alpha"), slot_tiles(rec["alpha"][0])),
            ],
        )
    ]
    if n_views > 1:
        blocks.append(
            (
                _txt("view1"),
                [
                    (_txt("row_images"), image_tiles(1)),
                    (_txt("row_alpha"), slot_tiles(rec["alpha"][1])),
                ],
            )
        )

    n_cols = max(len(row) for _, rows in blocks for _, row in rows)
    block_h = tile_h + cap_h
    top = head_h if title else 0
    height = (
        top
        + sum(head_h + pad + len(rows) * (head_h + block_h + pad) for _, rows in blocks)
        + pad
    )
    width = n_cols * (tile_w + pad) + pad
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    def fit(text: str, max_w: int) -> str:
        if draw.textlength(text, font=font) <= max_w:
            return text
        while text and draw.textlength(text + "…", font=font) > max_w:
            text = text[:-1]
        return text + "…"

    if title:
        draw.text((pad, 4), fit(title, width - 2 * pad), fill=(0, 0, 0), font=font)
    y = top + pad
    for view_heading, rows in blocks:
        # View heading on a light band, so the two views read as blocks.
        draw.rectangle([0, y, width, y + head_h - 2], fill=(230, 230, 236))
        draw.text((pad, y + 2), fit(view_heading, width - 2 * pad), fill=(0, 0, 0), font=font)
        y += head_h + pad
        for heading, row in rows:
            draw.text(
                (pad, y), fit(heading, width - 2 * pad), fill=(90, 90, 90), font=font
            )
            y += head_h
            for c, (img, caption, slot) in enumerate(row):
                x = pad + c * (tile_w + pad)
                tile = Image.fromarray(img).resize((tile_w, tile_h), Image.NEAREST)
                sheet.paste(tile, (x, y))
                # Thin frame so white tiles stay visible on the white sheet.
                draw.rectangle(
                    [x - 1, y - 1, x + tile_w, y + tile_h], outline=(170, 170, 170)
                )
                text_x = x
                if slot is not None:
                    sw = cap_h - 10
                    draw.rectangle(
                        [x, y + tile_h + 5, x + sw, y + tile_h + 5 + sw],
                        fill=tuple(int(c_) for c_ in _slot_color(slot)),
                        outline=(0, 0, 0),
                    )
                    text_x = x + sw + 4
                draw.text(
                    (text_x, y + tile_h + 3),
                    fit(caption, tile_w - (text_x - x)),
                    fill=(0, 0, 0),
                    font=font,
                )
            y += block_h + pad
    return np.asarray(sheet)


# ----------------------------------------------------------------------------
# 4. Scenes
# ----------------------------------------------------------------------------
def summarize(per_scene: list[dict]) -> dict:
    keys = [k for k, v in per_scene[0].items() if isinstance(v, float)]
    mean = {k: float(np.nanmean([s[k] for s in per_scene])) for k in keys}
    spreads = [p["spread"] for s in per_scene for p in s["per_slot"][1:]]
    mean["spread_min"] = float(np.min(spreads))
    return mean


def _title(stats: dict) -> str:
    def fmt(value) -> str:
        return "-" if value is None or np.isnan(value) else f"{value:.2f}"

    g = stats.get
    return (
        f"scene {stats['scene']:04d} | FG-ARI: attention {fmt(g('attn_fg_ari'))}, "
        f"render v0 {fmt(g('render_fg_ari_view0'))}, "
        f"render novel {fmt(g('render_fg_ari_novel'))} | dominance: attention "
        f"{fmt(g('attn_dominance'))}, render {fmt(g('render_dominance'))} | "
        f"bg on objects {fmt(g('bg_on_objects'))} | "
        f"objects found {fmt(g('obj_recall_view0'))}"
    )


def sheets(
    model, dataset, device, n_scenes: int, seed: int = 0
) -> tuple[list[np.ndarray], list[str], dict[str, float]]:
    """Diagnostic sheets of the first n_scenes scenes of dataset.

    Returns (sheets, (H, W, 3) uint8 each; their captions, the scene and its
    key scores; the statistics of analyse averaged over the scenes).
    """
    was_training = model.training
    model.eval()
    images, captions, per_scene = [], [], []
    for i in range(min(n_scenes, len(dataset))):
        item = dataset[i]
        rec = collect(model, item, device, seed + i)
        stats = analyse(rec)
        stats["scene"] = int(item.get("scene", i))
        per_scene.append(stats)
        title = _title(stats)
        images.append(make_grid(rec, title=title))
        captions.append(title)
    model.train(was_training)
    return images, captions, summarize(per_scene)
