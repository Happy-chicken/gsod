"""Minimal dependency-free PNG writer (8-bit RGB) for previews."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

__all__ = ["save_png"]


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
