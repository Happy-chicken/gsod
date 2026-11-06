"""Base configuration (model, optimisation, data, loss); the dataset
profiles in datasets.py override the dataset-specific fields."""

from __future__ import annotations

from dataclasses import dataclass, field

from gsod.utils.io import default_out_dir


@dataclass
class LossConfig:
    """Loss weights (gsod/model/losses.py)."""

    # Reconstruction on the novel views 1..V-1.
    l1: float = 1.0
    ssim: float = 0.2
    # L1 on view 0, the encoded view.
    view0: float = 1.0


@dataclass
class BaseConfig:
    # --- model -----------------------------------------------------------------
    # Side length of the square image passed to the encoder.
    input_size: int = 64
    # Channel width of encoder features and slot latents.
    z_dim: int = 64
    # Slots K: the background (index 0) + K-1 object slots, as many as the
    # most objects in a scene (spare slots are a risk, not a help).
    num_slots: int = 6
    # Number of slot-attention refinement iterations.
    attn_iter: int = 3
    # Gaussians per object slot (the background's: decoders/background.py).
    gaussians_per_slot: int = 512
    # Attention width d_model of the Gaussian decoders.
    decoder_hidden_dim: int = 128
    # Near clipping distance used when projecting Gaussians.
    near: float = 0.1
    # Decoders also read view 0's encoder tokens through their slot's
    # attention (gsod/model/gsod.py); False: the slot latent alone.
    pixel_features: bool = True

    # --- optimisation ----------------------------------------------------------
    # Epochs; one scene per step. Schedules below are in (fractional) epochs.
    epochs: int = 100
    # Adam: linear warm-up, then cosine decay to lr_final * lr at the last epoch.
    lr: float = 3e-4
    lr_warmup_epochs: float = 6
    lr_final: float = 0.02
    # The draw factor's share exponent rises from 0 to 1 over these epochs.
    draw_ramp_epochs: float = 10
    # Gradient-norm clipping (0 = off).
    grad_clip: float = 1.0
    # Seed for torch, scene/view sampling, and model initialisation.
    seed: int = 0
    # Log the training terms every N steps.
    log_every: int = 100
    # Run validation on the val split every N epochs.
    eval_every: int = 5
    # Write an epoch_<n> checkpoint every N epochs.
    save_every: int = 10
    # DataLoader worker processes.
    num_workers: int = 4
    # Loss term weights.
    loss: LossConfig = field(default_factory=LossConfig)

    # --- data ------------------------------------------------------------------
    # Dataset profile (datasets.DATASETS).
    dataset_mode: str = "primordia"
    # Multi-view datasets only (Primordia reads these from meta.json):
    # square image size (uORF trains at 128x128),
    dataset_image_size: int = 128
    # focal ratio (fx/W, fy/H) of the 320x240 renders,
    dataset_focal_ratio: tuple[float, float] = (350.0 / 320.0, 350.0 / 240.0)
    # views loaded per scene (uORF uses four),
    n_views: int = 4
    # fraction of the training root held out for validation.
    val_fraction: float = 0.1
    # Quick runs: use only the first N training / val scenes (None: all).
    train_scenes: int | None = None
    val_scenes: int | None = None
    # Test scenes (first N, with GT masks) for segmentation during training.
    monitor_scenes: int = 50

    # Dataset root (Primordia: meta.json; multi-view: a flat directory).
    data_root: str = "data/primordia"
    # Separate test root (multi-view datasets); None: inside data_root.
    test_root: str | None = None
    # Directory for checkpoints and their *_state.json files.
    out_dir: str = field(default_factory=lambda: default_out_dir("run"))
