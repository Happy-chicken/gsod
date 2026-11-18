#!/usr/bin/env python
"""Probe: slot-state collision. Are two object slots on the same GT object
(a duplicate pair) nearly the same vector?

Per scene, object slots get roles from the rendered view-0 mask (as in
evaluate.slot_table); every pair of object slots is labelled
    same       both on one GT object (one is its duplicate)
    different  on two GT objects
    other      a slot that is empty / haze
and scored by the cosine of the slot vectors and of the attention maps.
Collision: "same" pairs near 1 while "different" pairs are not. Needs GT
masks and CUDA (rendering).

    python scripts/probe_collision.py --dataset room_chair --checkpoint <run dir>

Writes <checkpoint>/probe/collision.json and collision_pairs.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from gsod.configs import DATASETS, config_for, run_config
from gsod.data_loader import WithGTMasks, dataset_meta, make_dataset
from gsod.model import build_model
from gsod.model.evaluate import collect, slot_table
from gsod.utils.io import latest_run_dir, load_networks


def pair_kind(a: dict, b: dict) -> str:
    on = ("object", "duplicate")
    if a["role"] not in on or b["role"] not in on:
        return "other"
    return "same" if a["gt_object"] == b["gt_object"] else "different"


def summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = np.asarray(values)
    return {
        "n": len(v),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="room_chair")
    parser.add_argument("--data", default=None, help="dataset root with GT masks")
    parser.add_argument("--checkpoint", default=None, help="run directory")
    parser.add_argument("--suffix", default="latest")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scenes", type=int, default=100, help="first N scenes")
    args = parser.parse_args()

    cfg = config_for(args.dataset)
    checkpoint = args.checkpoint or latest_run_dir(cfg.dataset_mode)
    cfg = run_config(cfg, Path(checkpoint))
    root = args.data or (cfg.test_root if args.split == "test" and cfg.test_root else cfg.data_root)
    device = torch.device("cuda")
    kwargs = dict(dataset_mode=cfg.dataset_mode, image_size=cfg.dataset_image_size,
                  focal_ratio=cfg.dataset_focal_ratio)
    model = build_model(cfg, dataset_meta(root, **kwargs)).to(device).eval()
    load_networks(checkpoint, args.suffix, model.networks(), device=device)
    data = make_dataset(root, args.split, n_views=cfg.n_views, val_fraction=cfg.val_fraction,
                        seed=cfg.seed, load_masks=True, **kwargs)
    if cfg.dataset_mode == "primordia":
        data = WithGTMasks(data, root)

    pairs = []
    for i in range(min(args.scenes, len(data))):
        item = data[i]
        if item.get("masks") is None:
            raise ValueError("the probe needs GT masks")
        scene = model(item["images"].to(device), item["cam2world"].to(device), render=False)
        rows = slot_table(collect(model, item, device))
        slots = F.normalize(scene["slots"].float(), dim=-1)  # (K-1, C)
        attn = F.normalize(scene["attn"][1:].float(), dim=-1)  # (K-1, N)
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                pairs.append({
                    "scene": i,
                    "slot_a": a + 1,
                    "slot_b": b + 1,
                    "kind": pair_kind(rows[a], rows[b]),
                    "slot_cos": float(slots[a] @ slots[b]),
                    "attn_cos": float(attn[a] @ attn[b]),
                })

    init = F.normalize(model.slot_attention.slots_init.weight[1:].float(), dim=-1)
    init_cos = init @ init.T
    off = ~torch.eye(len(init), dtype=torch.bool, device=init.device)
    metrics = {
        kind: {
            "slot_cos": summary([p["slot_cos"] for p in pairs if p["kind"] == kind]),
            "attn_cos": summary([p["attn_cos"] for p in pairs if p["kind"] == kind]),
        }
        for kind in ("same", "different", "other")
    }
    metrics["init_cos"] = summary(init_cos[off].tolist())  # learned object queries

    out = Path(checkpoint) / "probe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "collision.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "collision_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)

    for kind in ("same", "different", "other"):
        for score in ("slot_cos", "attn_cos"):
            s = metrics[kind][score]
            if s["n"]:
                print(f"{kind:9s} {score}  n={s['n']:4d}  mean={s['mean']:.3f}  "
                      f"median={s['median']:.3f}  p10={s['p10']:.3f}  p90={s['p90']:.3f}")
    s = metrics["init_cos"]
    print(f"init      slot_cos  mean={s['mean']:.3f}  p10={s['p10']:.3f}  p90={s['p90']:.3f}")
    print(f"wrote {out}/collision.json and collision_pairs.csv")


if __name__ == "__main__":
    main()
