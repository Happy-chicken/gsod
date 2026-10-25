"""Run slot diagnostics on a trained checkpoint or a freshly initialised model.

Examples
--------
    # newest primordia run, val split, 8 scenes
    python scripts/run_diagnose.py

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

from gsod.configs.primordia import PrimordiaConfig
from gsod.data_generator.io import load_meta
from gsod.data_loader.gt_masks import WithGTMasks
from gsod.data_loader.primordia import PrimordiaScenes
from gsod.model import build_model, load_networks
from gsod.utils.diagnose import diagnose
from gsod.utils.paths import latest_run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None, help="run dir; default newest primordia run"
    )
    parser.add_argument("--suffix", default="latest", help="checkpoint suffix")
    parser.add_argument(
        "--init", action="store_true", help="diagnose an untrained model"
    )
    parser.add_argument("--data", default=None, help="dataset root")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = PrimordiaConfig()
    data_root = args.data or cfg.data_root
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    # Same factory as train.py / test.py, so the decoder class matches.
    model = build_model(cfg, load_meta(data_root)).to(device)

    if args.init:
        out = Path(args.out or "debug1/init")
    else:
        ckpt = args.checkpoint or latest_run_dir("primordia")
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

    dataset = WithGTMasks(PrimordiaScenes(data_root, split=args.split), data_root)
    diagnose(model, dataset, device, out, n_scenes=args.n_scenes, seed=args.seed)


if __name__ == "__main__":
    main()
