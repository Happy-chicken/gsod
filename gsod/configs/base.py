"""Base configuration; dataset profiles (datasets.py) override it."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.utils.io import default_out_dir


@dataclass
class LossConfig:
    l1: float = 1.0  # novel views
    ssim: float = 0.2  # novel views
    view0: float = 0.2  # L1 on the encoded view (decoders read it: copying is cheap)
    bg_explained: float = 0.05  # object weight where the background alone matches
    bg_sigma: float = 0.05  # L1 error at which the background explains 1/e


@dataclass
class BaseConfig:
    # --- model
    input_size: int = 128  # encoder input side
    z_dim: int = 64
    num_slots: int = 6  # background + K-1 objects
    attn_iter: int = 3
    gaussians_per_slot: int = 512
    decoder_hidden_dim: int = 128
    near: float = 0.1

    # --- optimisation (one scene per step; ramps as fractions of training)
    epochs: int = 100
    lr: float = 3e-4  # warm-up, then cosine to lr_final * lr
    lr_warmup: float = 0.06
    lr_final: float = 0.02
    draw_ramp: float = 0.1  # draw factor power 0 -> 1
    bg_explained_ramp: float = 0.2  # bg_explained 0 -> 1 from draw_ramp to this
    grad_clip: float = 1.0  # 0 = off
    seed: int = 0
    log_every: int = 100  # steps
    eval_every: int = 5
    save_every: int = 10
    num_workers: int = 4
    loss: LossConfig = field(default_factory=LossConfig)

    # --- data (image size, focal, views: multi-view datasets only)
    dataset_mode: str = "primordia"
    dataset_image_size: int = 128
    dataset_focal_ratio: tuple[float, float] = (350.0 / 320.0, 350.0 / 240.0)
    n_views: int = 4
    val_fraction: float = 0.1
    train_scenes: int | None = None  # quick runs: first N scenes
    val_scenes: int | None = None
    monitor_scenes: int = 50  # test scenes scored during training
    data_root: str = "data/primordia"
    test_root: str | None = None
    out_dir: str = field(default_factory=lambda: default_out_dir("run"))
