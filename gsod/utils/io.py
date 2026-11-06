"""Files: run directories, checkpoints, PNG previews and 3DGS .ply exports.

Checkpoints (uORF's BaseModel layout), per suffix (e.g. 'latest' or
'epoch_25'): {suffix}_net_{name}.pth per network, {suffix}_optimizer_{i}.pth
per optimizer, {suffix}_lr_scheduler_{i}.pth per scheduler.

PNG: a dependency-free 8-bit RGB writer.

PLY (SuperSplat, SIBR and other viewers): x y z, nx ny nz (0), f_dc_0..2,
f_rest_0..44 (SH degree 3, channel-major; only degree 1 is filled), opacity
(logit), scale_0..2 (log), rot_0..3 (wxyz).
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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
    candidates = [
        path
        for path in root.glob(f"{tag}_*")
        if path.is_dir() and any(path.glob("latest_net_*.pth"))
    ]
    if not candidates:
        # Backward-compatible non-timestamped directory.
        legacy = root / tag
        if legacy.is_dir() and any(legacy.glob("latest_net_*.pth")):
            return str(legacy)
        return None
    return str(max(candidates, key=lambda path: (path.stat().st_mtime, path.name)))


# -- checkpoints --------------------------------------------------------------
def save_networks(
    save_dir: str | Path,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
    schedulers: Sequence = (),
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, net in networks.items():
        torch.save(net.state_dict(), save_dir / f"{suffix}_net_{name}.pth")
    for i, opt in enumerate(optimizers):
        torch.save(opt.state_dict(), save_dir / f"{suffix}_optimizer_{i}.pth")
    for i, sch in enumerate(schedulers):
        torch.save(sch.state_dict(), save_dir / f"{suffix}_lr_scheduler_{i}.pth")


def load_networks(
    save_dir: str | Path,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
    schedulers: Sequence = (),
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Load weights with strict=False, as uORF does; returns the load result
    per network. Missing network files are reported and skipped; optimizer
    and scheduler state is restored only when its file exists."""
    save_dir = Path(save_dir)
    result: dict[str, object] = {}
    for name, net in networks.items():
        path = save_dir / f"{suffix}_net_{name}.pth"
        if not path.is_file():
            print(f"not found: {path}, skip {name}")
            continue
        result[name] = net.load_state_dict(torch.load(path, map_location=device), strict=False)
    for i, opt in enumerate(optimizers):
        path = save_dir / f"{suffix}_optimizer_{i}.pth"
        if path.is_file():
            opt.load_state_dict(torch.load(path, map_location=device))
    for i, sch in enumerate(schedulers):
        path = save_dir / f"{suffix}_lr_scheduler_{i}.pth"
        if path.is_file():
            sch.load_state_dict(torch.load(path, map_location=device))
    return result


# -- PNG ----------------------------------------------------------------------
def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def save_png(path, array) -> Path:
    """Write an ``(H, W)`` grayscale or ``(H, W, 3)`` RGB ``uint8`` array."""
    array = np.asarray(array, dtype=np.uint8)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) or (H, W), got {array.shape}")
    height, width, _ = array.shape

    raw = bytearray()
    for row in array:
        raw.append(0)  # filter type: none
        raw.extend(row.tobytes())

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)
    return path


# -- PLY ----------------------------------------------------------------------
_REST = 15  # SH coefficients per channel beyond DC, degree 3


def ply_fields(gaussians: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """mu, scale, quat, opacity, rgb and optional sh1 (N, 3 basis, 3 rgb) ->
    3DGS fields, each (N,) or (N, k)."""
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
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", f"element vertex {data.shape[0]}"]
        + [f"property float {name}" for name in names]
        + ["end_header"]
    ) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        out.write(header.encode("ascii"))
        out.write(data.tobytes())
    return path


def read_ply(path: str | Path) -> dict[str, np.ndarray]:
    """A 3DGS .ply written by write_ply -> {property name: (N,)}."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode("ascii").splitlines()
    names = [line.split()[-1] for line in lines if line.startswith("property")]
    count = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
    data = np.frombuffer(raw[end:], "<f4").reshape(count, len(names))
    return {name: data[:, i] for i, name in enumerate(names)}
