"""Run-directory helpers.

Each training invocation gets a timestamped checkpoint directory, e.g.
``checkpoints/primordia_20261022_143000``. Testing without ``--checkpoint``
uses the newest matching run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


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
