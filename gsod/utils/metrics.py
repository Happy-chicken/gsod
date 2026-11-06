"""Segmentation and reconstruction metrics (numpy, framework-independent).

Conventions follow the object-discovery literature (Slot Attention, uORF):

* ARI     -- adjusted Rand index over all pixels (background included).
* FG-ARI  -- ARI over ground-truth foreground pixels only (gt > 0).
* PSNR    -- of a mean squared error, images in [0, 1].

Which views and how they are averaged: gsod/model/evaluate.py.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["ari", "fg_ari", "psnr", "upsample_labels"]


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1.0) / 2.0


def ari(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Adjusted Rand index between two label maps of the same shape.

    mask (bool) restricts the pixels that are scored. Returns nan when fewer
    than two pixels are scored. Matches sklearn.metrics.adjusted_rand_score,
    including its degenerate cases (both single-cluster -> 1.0).
    """
    gt = np.asarray(gt).ravel()
    pred = np.asarray(pred).ravel()
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).ravel()
        gt, pred = gt[keep], pred[keep]
    n = gt.size
    if n < 2:
        return float("nan")
    _, gt = np.unique(gt, return_inverse=True)
    _, pred = np.unique(pred, return_inverse=True)
    n_pred = int(pred.max()) + 1
    cont = np.bincount(gt * n_pred + pred, minlength=(int(gt.max()) + 1) * n_pred)
    cont = cont.reshape(-1, n_pred).astype(np.float64)

    sum_ij = _comb2(cont).sum()
    sum_a = _comb2(cont.sum(axis=1)).sum()
    sum_b = _comb2(cont.sum(axis=0)).sum()
    expected = sum_a * sum_b / _comb2(float(n))
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def fg_ari(gt: np.ndarray, pred: np.ndarray) -> float:
    """ARI over ground-truth foreground pixels (gt > 0); nan if none."""
    return ari(gt, pred, mask=np.asarray(gt) > 0)


def upsample_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of an (h, w) label map to size = (H, W)."""
    h, w = labels.shape
    rows = (np.arange(size[0]) * h) // size[0]
    cols = (np.arange(size[1]) * w) // size[1]
    return labels[rows[:, None], cols[None, :]]


def psnr(mse: float) -> float:
    """PSNR (dB) of a mean squared error, images in [0, 1]; nan stays nan."""
    if math.isnan(mse):
        return mse
    return -10.0 * math.log10(max(mse, 1e-12))
