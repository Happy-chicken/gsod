"""Run slot diagnostics on a trained checkpoint or a freshly initialised model.

Examples
--------
    # newest primordia run, val split, 8 scenes
    python scripts/run_diagnose.py

    # newest clevr567 run, test split (data/clevr/clevr_567_test: the
    # training root has no masks, so its val split has no GT metrics)
    python scripts/run_diagnose.py --dataset clevr567

    # a specific run / split / more scenes
    python scripts/run_diagnose.py --checkpoint checkpoints/primordia_20261022_143000 \
        --split test --n-scenes 16

    # untrained model: checks the initialisation only
    python scripts/run_diagnose.py --init
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: the gsod package lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import torch

from gsod.configs import config_for
from gsod.data_loader import WithGTMasks, dataset_meta, make_dataset
from gsod.model import build_model, load_networks
from gsod.utils.diagnose import diagnose
from gsod.utils.paths import latest_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["primordia", "clevr567"], default="primordia"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="run dir; default newest run of --dataset"
    )
    parser.add_argument("--suffix", default="latest", help="checkpoint suffix")
    parser.add_argument(
        "--init", action="store_true", help="diagnose an untrained model"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="default: test when the profile has a test_root (masks), else val",
    )
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = config_for(args.dataset)
    split = args.split or ("test" if cfg.test_root else "val")
    data_root = args.data or (
        cfg.test_root if split == "test" and cfg.test_root else cfg.data_root
    )
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    meta = dataset_meta(
        data_root,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
    )
    # Same factory as train.py / test.py, so the decoder class matches.
    model = build_model(cfg, meta).to(device)

    if args.init:
        out = Path(args.out or "debug1/init")
    else:
        ckpt = args.checkpoint or latest_run_dir(cfg.dataset_mode)
        if ckpt is None:
            raise SystemExit("no checkpoint found; pass --checkpoint or --init")
        result = load_networks(ckpt, args.suffix, model.networks(), device=device)
        for name, status in result.items():
            if status.missing_keys or status.unexpected_keys:
                print(
                    f"[warn] {name}: missing {status.missing_keys}, "
                    f"unexpected {status.unexpected_keys}"
                )
        out = Path(args.out or Path(ckpt) / "diagnose")

    dataset = make_dataset(
        data_root,
        split,
        dataset_mode=cfg.dataset_mode,
        image_size=cfg.dataset_image_size,
        focal_ratio=cfg.dataset_focal_ratio,
        n_views=cfg.n_views,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
        load_masks=True,  # CLEVR-567 *_mask.png; ignored by Primordia
    )
    if cfg.dataset_mode == "primordia":
        dataset = WithGTMasks(dataset, data_root)
    print(f"{args.dataset} {split} split: {data_root} ({len(dataset)} scenes)")
    diagnose(model, dataset, device, out, n_scenes=args.n_scenes, seed=args.seed)


if __name__ == "__main__":
    main()
