"""Slot diagnostics for W&B: per-scene image panels and averaged statistics.

train.py logs them at every validation, test.py once on the test split
(``scene_panels``). Per scene and on average they answer:

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

Each scene's images are logged as separate captioned tiles grouped into
panels by view (``panels``): view0/images, view0/attention, view0/slots,
view1/images, view1/slots.
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


def _slot_name(k: int) -> str:
    return "bg" if k == 0 else f"s{k}"


def _slot_tiles(maps: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """Grey tile per slot (white = 1), captioned with its share of the image."""
    return [(_gray(m), f"{_slot_name(k)} {float(m.mean()):.0%}") for k, m in enumerate(maps)]


def panels(rec: dict) -> dict[str, list[tuple[np.ndarray, str]]]:
    """One scene's images as separate captioned tiles, grouped into panels
    by view:

    view0/images     input, render, GT mask, render mask, attention argmax
                     over the object slots (view 0, the encoded input)
    view0/attention  each slot's share of the view-0 tokens (bg = empty
                     option)
    view0/slots      each slot's compositing weight per pixel (bg: the
                     share no object covers)
    view1/images, view1/slots   the same for view 1, a novel view the
                     encoder never sees, so it tests the 3D

    Masks: argmax slot per pixel, background dark grey, slot k in PALETTE.
    """
    size = rec["mask_idx"].shape[1:]
    gt = rec["gt"]

    def images(v: int) -> list[tuple[np.ndarray, str]]:
        tiles = [
            (_rgb(rec["target"][v]), "input"),
            (_rgb(rec["rgb"][v]), "render"),
            (_colorize(rec["mask_idx"][v]), "render mask"),
        ]
        if gt is not None:
            tiles.insert(2, (_colorize(gt[v]), "GT mask"))
        return tiles

    attn_up = _up(rec["attn"], size)
    out = {
        "view0/images": images(0)
        + [(_colorize(attn_up[1:].argmax(0) + 1), "attention (objects)")],
        "view0/attention": _slot_tiles(attn_up),
        "view0/slots": _slot_tiles(rec["alpha"][0]),
    }
    if rec["rgb"].shape[0] > 1:
        out["view1/images"] = images(1)
        out["view1/slots"] = _slot_tiles(rec["alpha"][1])
    return out


# ----------------------------------------------------------------------------
# 4. Scenes
# ----------------------------------------------------------------------------
def summarize(per_scene: list[dict]) -> dict:
    keys = [k for k, v in per_scene[0].items() if isinstance(v, float)]
    mean = {k: float(np.nanmean([s[k] for s in per_scene])) for k in keys}
    spreads = [p["spread"] for s in per_scene for p in s["per_slot"][1:]]
    mean["spread_min"] = float(np.min(spreads))
    return mean


def scene_panels(
    model, dataset, device, n_scenes: int, seed: int = 0
) -> tuple[dict[str, list[tuple[np.ndarray, str]]], dict[str, float]]:
    """Panels of the first n_scenes scenes of dataset, keyed
    "scene####/view<v>/<panel>", and the statistics of analyse averaged over
    the scenes."""
    was_training = model.training
    model.eval()
    out, per_scene = {}, []
    for i in range(min(n_scenes, len(dataset))):
        item = dataset[i]
        rec = collect(model, item, device, seed + i)
        stats = analyse(rec)
        stats["scene"] = int(item.get("scene", i))
        per_scene.append(stats)
        out.update({f"scene{stats['scene']:04d}/{k}": v for k, v in panels(rec).items()})
    model.train(was_training)
    return out, summarize(per_scene)
