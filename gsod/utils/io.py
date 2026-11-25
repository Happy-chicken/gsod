"""Run directories, checkpoints (uORF layout: {suffix}_net_{name}.pth),
PNG writer and 3DGS .ply export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from gsod.utils.splatter import SH_C0


# -- run directories ----------------------------------------------------------
def timestamp() -> str:
    """Local date and time, e.g. ``20261022_143000``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_out_dir(tag: str) -> str:
    """Default checkpoint directory for a dataset profile."""
    return str(Path("checkpoints") / f"{tag}_{timestamp()}")


def latest_run_dir(tag: str, root: str | Path = "checkpoints") -> str | None:
    """Newest directory matching ``<root>/<tag>_*`` with a latest checkpoint."""
    root = Path(root)
    candidates = [path for path in root.glob(f"{tag}_*") if path.is_dir() and any(path.glob("latest_net_*.pth"))]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: (path.stat().st_mtime, path.name)))


# -- checkpoints --------------------------------------------------------------
def save_networks(
    save_dir: str | Path,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, net in networks.items():
        torch.save(net.state_dict(), save_dir / f"{suffix}_net_{name}.pth")
    for i, opt in enumerate(optimizers):
        torch.save(opt.state_dict(), save_dir / f"{suffix}_optimizer_{i}.pth")


def load_networks(
    save_dir: str | Path,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Load weights (strict=False, mismatched keys are reported); missing files
    are skipped. A network none of whose keys match raises."""
    save_dir = Path(save_dir)
    result: dict[str, object] = {}
    for name, net in networks.items():
        path = save_dir / f"{suffix}_net_{name}.pth"
        if not path.is_file():
            print(f"not found: {path}, skip {name}")
            continue
        state = torch.load(path, map_location=device)
        result[name] = keys = net.load_state_dict(state, strict=False)
        if keys.missing_keys or keys.unexpected_keys:
            print(
                f"{name}: {len(keys.missing_keys)} missing, {len(keys.unexpected_keys)} unexpected keys: "
                f"{(keys.missing_keys + keys.unexpected_keys)[:4]}"
            )
        if len(keys.missing_keys) == len(net.state_dict()) > 0:
            raise RuntimeError(f"{path}: no key of {name} matches (a checkpoint of an older model?)")
    for i, opt in enumerate(optimizers):
        path = save_dir / f"{suffix}_optimizer_{i}.pth"
        if path.is_file():
            opt.load_state_dict(torch.load(path, map_location=device))
    return result


# -- PNG ----------------------------------------------------------------------
def save_png(path, array) -> Path:
    """Write an ``(H, W)`` grayscale or ``(H, W, 3)`` RGB ``uint8`` array (as RGB)."""
    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) or (H, W), got {array.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    return path


# -- PLY ----------------------------------------------------------------------
_REST = 15  # SH coefficients per channel beyond DC, degree 3


def ply_fields(gaussians: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Gaussians -> 3DGS .ply fields."""
    g = {k: v.detach().float().cpu() for k, v in gaussians.items()}
    n = g["mu"].shape[0]
    rest = torch.zeros(n, 3, _REST)
    if "sh1" in g:
        rest[:, :, :3] = g["sh1"].transpose(1, 2)  # (N, rgb, basis)
    return {
        "xyz": g["mu"].numpy(),
        "normal": np.zeros((n, 3), np.float32),
        "f_dc": ((g["rgb"] - 0.5) / SH_C0).numpy(),
        "f_rest": rest.flatten(1).numpy(),
        "opacity": torch.logit(g["opacity"].clamp(1e-6, 1 - 1e-6))[:, None].numpy(),
        "scale": g["scale"].clamp_min(1e-8).log().numpy(),
        "rot": torch.nn.functional.normalize(g["quat"], dim=-1).numpy(),
    }


def write_ply(path: str | Path, gaussians: dict[str, torch.Tensor]) -> Path:
    """Write Gaussians as a binary little-endian 3DGS .ply."""
    f = ply_fields(gaussians)
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(3 * _REST)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    data = np.concatenate(
        [f["xyz"], f["normal"], f["f_dc"], f["f_rest"], f["opacity"], f["scale"], f["rot"]], 1
    ).astype("<f4")
    header = (
        "\n".join(
            ["ply", "format binary_little_endian 1.0", f"element vertex {data.shape[0]}"]
            + [f"property float {name}" for name in names]
            + ["end_header"]
        )
        + "\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        out.write(header.encode("ascii"))
        out.write(data.tobytes())
    return path
