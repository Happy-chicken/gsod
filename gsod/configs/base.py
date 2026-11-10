"""Base configuration; dataset profiles (datasets.py) override it."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.utils.io import default_out_dir


@dataclass
class LossConfig:
    """As uORF, every view."""

    recon: float = 1.0  # MSE, images in [-1, 1]
    percept: float = 0.006  # MSE of VGG16 relu4_3 features


@dataclass
class BaseConfig:
    # --- model
    input_size: int = 128  # encoder input side
    z_dim: int = 64
    num_slots: int = 6  # background + K-1 objects
    attn_iter: int = 3
    ray_side: int = 64  # object Gaussians: rays per side of view 0
    ray_layers: int = 3  # Gaussians per ray and object slot (visible surface + back)
    decoder_hidden_dim: int = 128
    near: float = 0.1

    # --- optimisation (one scene per step; ramps as fractions of training)
    epochs: int = 100
    lr: float = 3e-4  # warm-up, then cosine to lr_final * lr
    lr_warmup: float = 0.06
    lr_final: float = 0.02
    percept_start: float = 0.2  # perceptual loss on from here (uORF: epoch 20)
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
