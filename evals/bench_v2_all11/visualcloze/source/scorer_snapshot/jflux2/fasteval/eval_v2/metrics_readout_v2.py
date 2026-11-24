"""Inactive, NumPy-only BSDS count readout with explicit duplicate-recall policy.

AP follows collect_eval_bdry.m in the pinned BIDS BSDS500 2013 package:
unique recall, linear interpolation at 0:.01:1, zero outside the measured
recall interval, and zero area for at most one unique recall. Its /100
convention is retained, including the possible 1.01 result. This module is
not wired into any scorer; frozen metrics_v1 and its ODS/OIS are preserved.
Only AP uses collector precision zero at empty-prediction thresholds; the
frozen primary program retains its original precision-one empty convention.

The collector calls unqualified MATLAB unique(R). FIRST is MATLAB's
R2013a+ default; earlier releases used LAST. The actual historical runtime
is unknown, so callers must explicitly choose occurrence="first" or "last".
This implements the collector's AP rules, not certified historical runtime
or bitwise MATLAB parity. Duplicate selection uses original threshold order.
"""
from __future__ import annotations

import numpy as np


def _occurrence(value: str) -> str:
    if value not in ("first", "last"):
        raise ValueError("occurrence must explicitly be 'first' or 'last'")
    return value


def bsds_average_precision(precision, recall, *, occurrence: str) -> float:
    """Sampled collector AP; no extrapolation, precision envelope, or clamp.

Indices of duplicate recalls refer to the input threshold order, before
recall sorting. Exact equality defines duplicates, as in MATLAB unique.
An empty curve also has at most one unique recall and returns zero.
"""
    occurrence = _occurrence(occurrence)
    p = np.asarray(precision, np.float64)
    r = np.asarray(recall, np.float64)
    if p.ndim != 1 or r.ndim != 1 or p.shape != r.shape:
        raise ValueError("precision and recall must be equal-length vectors")
    if (not np.isfinite(p).all() or not np.isfinite(r).all()
            or np.any((p < 0) | (p > 1)) or np.any((r < 0) | (r > 1))):
        raise ValueError("precision and recall must be finite values in [0, 1]")
    if occurrence == "first":
        ru, indices = np.unique(r, return_index=True)
    else:
        ru, reverse_indices = np.unique(r[::-1], return_index=True)
        indices = len(r) - 1 - reverse_indices
    if len(ru) <= 1:
        return 0.0
    pu = p[indices]
    grid = np.linspace(0, 1, 101)
    interpolated = np.interp(grid, ru, pu, left=0.0, right=0.0)
    return float(interpolated.sum() / 100.0)


def _f(prec, rec):
    return 0.0 if prec + rec <= 0 else 2 * prec * rec / (prec + rec)


def bsds_dataset_scores(all_counts: list, *, occurrence: str) -> dict:
    """Frozen ODS/OIS plus explicit-policy AP from [image][threshold][4] counts.

Count fields are matched predictions, predictions, matched GT, GT. OIS is
computed by pooling each image's best-threshold counts, as in metrics_v1,
not averaging its per-image F. Ties retain the first threshold. Empty
dataset NaNs match the frozen interface; malformed nonempty counts refuse.
"""
    _occurrence(occurrence)
    c = np.asarray(all_counts, np.float64)
    if c.size == 0:
        return {"ods": float("nan"), "ois": float("nan"), "ap": float("nan")}
    if c.ndim != 3 or c.shape[1] == 0 or c.shape[2] != 4:
        raise ValueError("counts must have shape [image][threshold][4]")
    if (not np.isfinite(c).all() or np.any(c < 0) or np.any(c != np.floor(c))
            or np.any(c[:, :, 0] > c[:, :, 1]) or np.any(c[:, :, 2] > c[:, :, 3])):
        raise ValueError("counts must be finite nonnegative integers with valid matches")
    # Preserve the frozen primary program and its reduction order verbatim.
    S = c.sum(0)
    P = np.where(S[:, 1] > 0, S[:, 0] / np.maximum(S[:, 1], 1), 1.0)
    R = np.where(S[:, 3] > 0, S[:, 2] / np.maximum(S[:, 3], 1), 0.0)
    F = np.array([_f(p, r) for p, r in zip(P, R)])
    best = []
    for img in c:
        pi = np.where(img[:, 1] > 0, img[:, 0] / np.maximum(img[:, 1], 1), 1.0)
        ri = np.where(img[:, 3] > 0, img[:, 2] / np.maximum(img[:, 3], 1), 0.0)
        best.append(img[int(np.argmax([_f(a, b) for a, b in zip(pi, ri)]))])
    B = np.sum(best, 0)
    ois_ = _f(B[0] / B[1] if B[1] else 1.0, B[2] / B[3] if B[3] else 0.0)
    ap_precision = np.where(S[:, 1] > 0, P, 0.0)
    return {"ods": float(F.max()), "ods_threshold_index": int(F.argmax()), "ois": float(ois_),
            "ap": bsds_average_precision(ap_precision, R, occurrence=occurrence)}
