"""Slot diagnostics: where does object discovery break?

Answers, per scene and on average:

1. Initialisation   -- are the Gaussians of a slot spread out, or one point?
2. Slot attention   -- are the foreground slot latents distinct? Does the
                       attention map separate objects (attn FG-ARI, dominance)?
3. Decoder          -- does the rendered slot partition follow the attention,
                       or do several slots redundantly draw the same object
                       (render dominance, attn/render ARI)?
4. Frames           -- are foreground frame centres distinct?

Usage (from your own test script, after loading a checkpoint):

    from gsod.utils.diagnose import diagnose
    report = diagnose(model, val_dataset, device, "debug/step_020000")

Also run it once on a freshly initialised model: the "init" checks should
pass before training starts.

Outputs in out_dir:
    sc####_diag.png   captioned sheet (see make_grid): title with the scene's
                      key scores; headed rows for view 0 (input, render,
                      masks, background-only render, F), view 1 (novel view),
                      slot attention, and rendered slot weights of views 0
                      and 1, each slot tile with its mask colour and share of
                      the image (bg = background). Needs Pillow.
    summary.json      per-scene and mean statistics plus verdicts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gsod.utils.metrics import ari, fg_ari, upsample_labels
from gsod.utils.png import save_png

# Same colours as evaluate.py: slot 0 dark gray, slot k -> PALETTE[(k-1) % 8].
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

# Verdict thresholds (heuristics; tune once you have a working run to compare).
SPREAD_COLLAPSED = 0.05  # mean distance of a slot's Gaussians to their mean
SLOT_COS_COLLAPSED = 0.95  # mean pairwise cosine of foreground slot latents
DOMINANCE_GOOD = 0.8  # share of the strongest fg slot on object pixels
DOMINANCE_BAD = 0.6
ARI_GOOD = 0.7
CENTRE_COLLAPSED = 0.05  # min pairwise distance of active fg frame centres
ACTIVE_OPACITY = 0.05


# ----------------------------------------------------------------------------
# 1. Collect (the only part that touches torch / the model)
# ----------------------------------------------------------------------------
def collect(model, item: dict, device, seed: int = 0) -> dict[str, np.ndarray]:
    """Run the model on one scene and return everything as numpy arrays."""
    import torch

    from gsod.model.losses import background_residual_target

    images = item["images"].to(device)
    cams = item["cam2world"].to(device)
    devices = [device] if getattr(device, "type", "") == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        result = model(images, cams)
        # Foreground target F of view 0, as the losses compute it.
        fg_target = background_residual_target(result, images)[0]

    def np_(x):
        return None if x is None else x.detach().float().cpu().numpy()

    side = model.input_size
    gt = item.get("masks")
    g = result["gaussians"]
    return {
        "target": np_(images),
        "rgb": np_(result["rgb"].clamp(0, 1)),
        "alpha": np_(result["slot_alpha_full"]),  # (N, K, H, W)
        "mask_idx": result["mask_idx"].cpu().numpy(),  # (N, H, W)
        "attn": np_(result["attn"]).reshape(-1, side, side),  # (K, s, s)
        "slots": np_(result["slots"]),  # (K-1, C) object slots
        # Object frames as used in the forward pass: (K-1, 3) or None.
        "centre": np_(result["frame_centre"]),
        "mu": np_(g["mu"]),
        "scale": np_(g["scale"]),
        "opacity": np_(g["opacity"]),
        "slot_ids": result["slot_ids"].cpu().numpy(),
        "gt": None if gt is None else gt.cpu().numpy().astype(np.int64),
        "bg_view0": np_(result["bg_rgb"][0]),
        "fg_target": np_(fg_target),
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


def analyse(rec: dict) -> dict:
    n_slots = rec["alpha"].shape[1]  # K, background included
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]
    alpha0 = rec["alpha"][0]
    attn = np.stack([upsample_labels(a, size) for a in rec["attn"]])  # nearest
    attn_lab = attn.argmax(0)

    # Object pixels: GT if available, else pixels mostly covered by fg slots.
    if gt is not None:
        obj = gt[0] > 0
    else:
        obj = alpha0[1:].sum(0) > 0.3

    fg = rec["slots"]  # object slot latents
    fg_n = fg / (np.linalg.norm(fg, axis=1, keepdims=True) + 1e-8)
    cos = fg_n @ fg_n.T
    off = cos[~np.eye(len(fg), dtype=bool)] if len(fg) > 1 else np.array([np.nan])

    # Gaussians per slot.
    per_slot = []
    for k in range(n_slots):
        sel = rec["slot_ids"] == k
        mu = rec["mu"][sel]
        per_slot.append(
            {
                "slot": k,
                "spread": float(np.linalg.norm(mu - mu.mean(0), axis=1).mean()),
                "mean_scale": float(rec["scale"][sel].mean()),
                "mean_opacity": float(rec["opacity"][sel].mean()),
                "active_frac": float((rec["opacity"][sel] > ACTIVE_OPACITY).mean()),
                "coverage_view0": float(alpha0[k].mean()),
                "render_share_view0": float((rec["mask_idx"][0] == k).mean()),
                "attn_share": float(rec["attn"][k].mean()),
                "bg_on_objects": float(alpha0[k][obj].mean())
                if (k == 0 and obj.any())
                else None,
            }
        )

    # Frame centres of foreground slots that actually draw something.
    centre_min_dist = float("nan")
    if rec["centre"] is not None:
        active = [k for k in range(1, n_slots) if per_slot[k]["coverage_view0"] > 0.005]
        c = rec["centre"][[k - 1 for k in active]]  # frames are object slots 1..K-1
        if len(c) > 1:
            d = np.linalg.norm(c[:, None] - c[None], axis=-1)
            centre_min_dist = float(d[~np.eye(len(c), dtype=bool)].min())

    out = {
        "fg_slot_cos_mean": float(np.nanmean(off)),
        "fg_slot_cos_max": float(np.nanmax(off)),
        "attn_dominance": _dominance(attn[1:], obj),
        "render_dominance": _dominance(alpha0[1:], obj),
        # On object pixels only: on all pixels the shared background dominates.
        "attn_render_ari": ari(attn_lab, rec["mask_idx"][0], mask=obj),
        "fg_centre_min_dist": centre_min_dist,
        "per_slot": per_slot,
    }
    if gt is not None:
        out["attn_fg_ari"] = fg_ari(gt[0], attn_lab)
        out["render_fg_ari_view0"] = fg_ari(gt[0], rec["mask_idx"][0])
        novel = [fg_ari(gt[v], rec["mask_idx"][v]) for v in range(1, gt.shape[0])]
        novel = [x for x in novel if not np.isnan(x)]
        out["render_fg_ari_novel"] = float(np.mean(novel)) if novel else float("nan")
    return out


def verdicts(mean: dict) -> list[str]:
    """Turn mean statistics into a ranked list of likely problems."""
    msgs = []
    g = lambda key: mean.get(key, float("nan"))  # noqa: E731

    if g("fg_spread_min") < SPREAD_COLLAPSED:
        msgs.append(
            "[init] some fg slot has all Gaussians at one point "
            f"(min spread {g('fg_spread_min'):.3f}); check offset-bias / query-bias init."
        )
    if g("fg_slot_cos_mean") > SLOT_COS_COLLAPSED:
        msgs.append(
            f"[slot attention] fg slot latents nearly identical (mean cos "
            f"{g('fg_slot_cos_mean'):.3f}); slots never separate. Try soft position "
            "embedding + feature MLP, longer training / LR warmup."
        )
    attn_ok = g("attn_dominance") > DOMINANCE_GOOD or g("attn_fg_ari") > ARI_GOOD
    render_bad = (
        g("render_dominance") < DOMINANCE_BAD or g("render_fg_ari_view0") < ARI_GOOD
    )
    if attn_ok and render_bad:
        redundant = g("attn_dominance") - g("render_dominance") > 0.1
        cause = (
            "several slots redraw the same object (render dominance well below "
            "attention's); use the fg_render loss."
            if redundant
            else "slots draw their own objects, but the render is soft or misplaced "
            "(edges / thin parts lost to the background); see bg_on_objects."
        )
        msgs.append(
            "[decoder] render segments worse than attention "
            f"(FG-ARI attn {g('attn_fg_ari'):.2f} vs render {g('render_fg_ari_view0'):.2f}; "
            f"dominance attn {g('attn_dominance'):.2f} vs render {g('render_dominance'):.2f}): "
            + cause
        )
    elif not attn_ok and g("attn_dominance") < DOMINANCE_BAD:
        msgs.append(
            f"[slot attention] attention does not separate objects (attn dom "
            f"{g('attn_dominance'):.2f}, attn FG-ARI {g('attn_fg_ari'):.2f}); fix this "
            "before the decoder."
        )
    if g("fg_centre_min_dist") < CENTRE_COLLAPSED:
        msgs.append(
            f"[frame] two active fg frames share a centre (min dist "
            f"{g('fg_centre_min_dist'):.3f}); slots stack on the same object."
        )
    if g("bg_on_objects") > 0.3:
        msgs.append(
            f"[background] bg slot holds {g('bg_on_objects'):.2f} alpha on object "
            "pixels; it is explaining objects."
        )
    if (
        g("render_fg_ari_view0") > ARI_GOOD
        and g("render_fg_ari_novel") < g("render_fg_ari_view0") - 0.2
    ):
        msgs.append(
            "[3D] view 0 is segmented but novel views are not; objects are flat or "
            "at the wrong depth. Check multi-view loss weighting and cameras."
        )
    if not msgs:
        msgs.append("no failure pattern matched")
    return msgs


# ----------------------------------------------------------------------------
# 3. Visualise
# ----------------------------------------------------------------------------
def _colorize(labels: np.ndarray, background=MASK_BG) -> np.ndarray:
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


_TEXT = {
    "row_view0": "View 0 - the encoded input view (masks: argmax slot per pixel)",
    "row_view1": "View 1 - a novel view, never seen by the encoder (tests the 3D)",
    "row_attn": "Slot attention, view 0 - share of each token given to the slot "
    "(white = 1; % = share of the image)",
    "row_alpha": "Rendered slot weight, view {v} - share of each pixel drawn by "
    "the slot's Gaussians (bg includes uncovered pixels; % = share of the image)",
    "image": "input image",
    "render": "render",
    "gt": "GT mask",
    "mask": "render mask",
    "attn_mask": "attention mask",
    "bg_only": "bg-only render",
    "fg_target": "F: differs from bg",
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
    """Diagnostic sheet: one headed row per topic, a caption under every tile.

    View 0     input image, render, GT mask, render mask, attention mask,
               background-only render, foreground target F of the losses.
    View 1     input image, render, GT mask, render mask (novel view).
    Attention  slot attention of view 0 per slot (bg = background query).
    Weight v0  rendered weight per slot, view 0.
    Weight v1  rendered weight per slot, view 1.
    Slot captions carry the slot's mask colour and its share of the image,
    so empty slots read 0%. Captions are English.
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

    def view_tiles(v: int) -> list:
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

    # (heading, [(image, caption, slot index for the colour swatch or None)])
    rows = [
        (
            _txt("row_view0"),
            view_tiles(0)
            + [
                (_colorize(attn_up.argmax(0)), _txt("attn_mask"), None),
                (_rgb(rec["bg_view0"]), _txt("bg_only"), None),
                (_gray(rec["fg_target"]), _txt("fg_target"), None),
            ],
        )
    ]
    if n_views > 1:
        rows.append((_txt("row_view1"), view_tiles(1)))
    rows.append((_txt("row_attn"), slot_tiles(rec["attn"])))
    for v in range(min(n_views, 2)):
        rows.append((_txt("row_alpha", v=v), slot_tiles(rec["alpha"][v])))

    n_cols = max(len(r) for _, r in rows)
    block_h = tile_h + cap_h
    top = head_h if title else 0
    height = top + sum(head_h + block_h + pad for _ in rows) + pad
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
    for heading, row in rows:
        draw.text((pad, y), fit(heading, width - 2 * pad), fill=(90, 90, 90), font=font)
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
# 4. Driver
# ----------------------------------------------------------------------------
def summarize(per_scene: list[dict]) -> dict:
    keys = [k for k, v in per_scene[0].items() if isinstance(v, float)]
    mean = {k: float(np.nanmean([s[k] for s in per_scene])) for k in keys}
    fg_spreads = [p["spread"] for s in per_scene for p in s["per_slot"][1:]]
    bg_obj = [s["per_slot"][0]["bg_on_objects"] for s in per_scene]
    bg_obj = [x for x in bg_obj if x is not None]
    mean["fg_spread_min"] = float(np.min(fg_spreads)) if fg_spreads else float("nan")
    mean["bg_on_objects"] = float(np.mean(bg_obj)) if bg_obj else float("nan")
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
        f"bg on objects {fmt(stats['per_slot'][0]['bg_on_objects'])}"
    )


def diagnose(
    model,
    dataset,
    device,
    out_dir,
    n_scenes: int = 8,
    seed: int = 0,
) -> dict:
    out = Path(out_dir)
    was_training = model.training
    model.eval()
    per_scene = []
    for i in range(min(n_scenes, len(dataset))):
        item = dataset[i]
        rec = collect(model, item, device, seed + i)
        stats = analyse(rec)
        stats["scene"] = int(item.get("scene", i))
        per_scene.append(stats)
        save_png(
            out / f"sc{stats['scene']:04d}_diag.png",
            make_grid(rec, title=_title(stats)),
        )
    model.train(was_training)

    mean = summarize(per_scene)
    report = {"mean": mean, "verdicts": verdicts(mean), "scenes": per_scene}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2))

    print(f"[diagnose] {len(per_scene)} scenes -> {out}")
    for key in (
        "fg_slot_cos_mean",
        "attn_dominance",
        "render_dominance",
        "attn_fg_ari",
        "render_fg_ari_view0",
        "render_fg_ari_novel",
        "attn_render_ari",
        "fg_centre_min_dist",
        "fg_spread_min",
        "bg_on_objects",
    ):
        if key in mean:
            print(f"  {key:22s} {mean[key]:.3f}")
    for msg in report["verdicts"]:
        print("  ->", msg)
    return report
