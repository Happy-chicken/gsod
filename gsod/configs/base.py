"""Base configuration; dataset profiles (datasets.py) override it."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.utils.io import default_out_dir


@dataclass
class LossConfig:
    """Every view."""

    recon: float = 1.0  # MSE, images in [-1, 1]
    ssim: float = 0.2  # 1 - SSIM
    lpips: float = 0.05  # LPIPS (VGG); 0 = off
    late_start: float = 0.3  # LPIPS from this fraction of training on


@dataclass
class BaseConfig:
    # --- model
    input_size: int = 64  # encoder input side
    z_dim: int = 64
    num_slots: int = 6  # background + K-1 objects
    attn_iter: int = 3
    gaussians_per_point: int = 2  # per object window point: the first on the surface, the others free
    decoder_hidden_dim: int = 128
    near: float = 0.1

    # --- optimisation (ramps as fractions of training)
    epochs: int = 100
    batch_scenes: int = 2  # scenes per step
    lr: float = 3e-4  # warm-up, then cosine to lr_final * lr
    lr_warmup: float = 0.06
    lr_final: float = 0.02
    grad_clip: float = 1.0  # 0 = off
    seed: int = 0
    log_every: int = 100  # steps
    eval_every: int = 5
    save_every: int = 10
    num_workers: int = 4
    loss: LossConfig = field(default_factory=LossConfig)

    # --- data (dataset_mode, data_root, test_root: set by the dataset profile)
    dataset_mode: str = ""
    dataset_image_size: int = 128  # square
    dataset_focal_ratio: tuple[float, float] = (350.0 / 320.0, 350.0 / 240.0)  # fx / W, fy / H
    n_views: int = 4
    val_fraction: float = 0.1
    train_scenes: int | None = None  # quick runs: first N scenes
    val_scenes: int | None = None
    monitor_scenes: int = 50  # test scenes scored during training
    data_root: str = ""
    test_root: str | None = None
    out_dir: str = ""  # default: checkpoints/<dataset_mode>_<time>

    def __post_init__(self) -> None:
        if not self.out_dir:
            self.out_dir = default_out_dir(self.dataset_mode)
