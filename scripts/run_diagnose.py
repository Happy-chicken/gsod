"""Run slot diagnostics on a trained checkpoint or a freshly initialised model.

Examples
--------
    # newest primordia run, val split, 8 scenes
    python run_diagnose.py

    # a specific run / split / more scenes
    python run_diagnose.py --checkpoint checkpoints/primordia_20261022_143000 \
        --split test --n-scenes 16

    # untrained model: checks the initialisation only
    python run_diagnose.py --init

Lines marked ADAPT depend on code I have not seen (dataset constructor,
checkpoint file format); adjust them once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gsod.configs.primordia import PrimordiaConfig
from gsod.data_generator.io import load_meta
from gsod.data_loader.primordia import PrimordiaScenes
from gsod.data_loader.gt_masks import WithGTMasks
from gsod.utils.diagnose import diagnose  # ADAPT: where you put diagnose.py
from gsod.model.gsod import GSODModel
from gsod.utils.splatter import GsplatSplatter  # ADAPT: splatter module path
from gsod.utils.paths import latest_run_dir


def build_model(cfg: PrimordiaConfig, meta: dict, device: torch.device) -> GSODModel:
    gen = meta.get(
        "config", meta
    )  # ADAPT: key under which meta.json stores SceneGenConfig
    model = GSODModel(
        splat=GsplatSplatter(),
        image_size=tuple(gen["image_size"]),
        input_size=cfg.input_size,
        z_dim=cfg.z_dim,
        num_slots=cfg.num_slots,
        attn_iter=cfg.attn_iter,
        gaussians_per_slot=cfg.gaussians_per_slot,
        bg_gaussians_per_slot=cfg.bg_gaussians_per_slot,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_type="attention",  # ADAPT: same decoder_type as training
        focal_ratio=tuple(gen["focal_ratio"]),
        near=cfg.near,
        scene_config=gen,
    )
    return model.to(device)


def load_checkpoint(model: GSODModel, ckpt_dir: Path, device: torch.device) -> None:
    # ADAPT: assumes latest_net_<Name>.pth holds that network's state_dict.
    for name, net in model.networks().items():
        path = ckpt_dir / f"latest_net_{name}.pth"
        state = torch.load(path, map_location=device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[warn] {name}: missing {missing}, unexpected {unexpected}")
        print(f"loaded {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default=None, help="run dir; default newest primordia run"
    )
    parser.add_argument(
        "--init", action="store_true", help="diagnose an untrained model"
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-scenes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = PrimordiaConfig()
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    meta = load_meta(cfg.data_root)
    model = build_model(cfg, meta, device)

    if args.init:
        out = Path(args.out or "debug1/init")
    else:
        ckpt = args.checkpoint or latest_run_dir("primordia")
        if ckpt is None:
            raise SystemExit("no checkpoint found; pass --checkpoint or --init")
        load_checkpoint(model, Path(ckpt), device)
        out = Path(args.out or Path(ckpt) / "diagnose")

    dataset = WithGTMasks(
        PrimordiaScenes(cfg.data_root, split=args.split), cfg.data_root
    )  # ADAPT: constructor
    diagnose(model, dataset, device, out, n_scenes=args.n_scenes, seed=args.seed)


if __name__ == "__main__":
    main()
