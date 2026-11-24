"""eval v2 post-processing, v1: a model's RGB output (256 x 256 x 3 uint8) -> the metric's native format.

Every function takes the prediction exactly as saved (uint8 HWC) and is deterministic. The parameter choices and the
controls that fixed them are in docs/review_postproc_2026-10-06/REPORT.md (section 3) and its cross_cutting/ checks.

  seg_label_map       semantic / instance: label map from adaptive colour clusters (no vocabulary, no fixed K)
  point_blobs         pose / faces: unnamed points, one per disc (colour separates touching discs only),
                      capped at one prediction per GT point
  depth_grey          relative depth: grey value in [0, 1] plus the predicted-invalid mask (exact black)
  normals_unit        surface normals: unit vectors plus the predicted-invalid mask (exact black)
  luma01              saliency / edges: BT.601 luma in [0, 1] (as metrics_v2)
  inpaint_composite   inpainting: prediction inside the hole, the given input outside (the inpainting convention)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BLACK_MAX = 24                 # max(R, G, B) <= 24 is black: void / background / no point (the repo's luminance floor)
SEG_DELTA_E = 15.0             # CIELAB distance within which a colour joins an existing cluster
SEG_CORE_WIDTH = 3             # side of the square used to open a cluster: blends narrower than this have no core
SEG_CORE_MIN_PX = 16           # a cluster seeds a region only if its opened core keeps this many pixels
SEG_MIN_FRAGMENT_PX = 20       # connected fragments smaller than this are refilled from their neighbours
POINT_MIN_BLOB_PX = 4          # a disc fragment smaller than this is not a point
EIGHT = np.ones((3, 3), bool)


# ----------------------------------------------------------------------------- colour helpers
def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIE L*a*b* (D65), float64. Pure numpy, so no OpenCV version can change a label map."""
    c = np.asarray(rgb_u8, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def is_black(rgb_u8: np.ndarray) -> np.ndarray:
    return np.asarray(rgb_u8).max(-1) <= BLACK_MAX


# ----------------------------------------------------------------------------- segmentation
def _colour_clusters(pred_u8: np.ndarray, delta_e: float) -> np.ndarray:
    """-> raw label map: 0 = black, 1..M = greedy CIELAB clusters of the non-black colours (frequency-ordered
    seeds; a colour joins the nearest existing centre within delta_e, else opens a new centre)."""
    a = np.asarray(pred_u8, np.uint8)
    fg = ~is_black(a)
    raw = np.zeros(a.shape[:2], np.int32)
    if not fg.any():
        return raw
    cols, inv, cnt = np.unique(a[fg].reshape(-1, 3), axis=0, return_inverse=True, return_counts=True)
    lab = rgb_to_lab(cols)
    centres = np.empty((0, 3)); assign = np.empty(len(cols), np.int64)
    for i in np.lexsort((np.arange(len(cols)), -cnt)):          # by count, ties by colour order: deterministic
        if len(centres):
            d = np.sqrt(((centres - lab[i]) ** 2).sum(-1)); j = int(d.argmin())
            if d[j] <= delta_e:
                assign[i] = j
                continue
        centres = np.vstack([centres, lab[i]]); assign[i] = len(centres) - 1
    raw[fg] = assign[inv.reshape(-1)] + 1
    return raw


def _fill_from(lab: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Pixels where `keep` is False take the label of the nearest pixel where it is True (Euclidean)."""
    if keep.all() or not keep.any():
        return lab
    _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
    return lab[iy, ix]


def seg_label_map(pred_u8: np.ndarray, *, delta_e: float = SEG_DELTA_E, core_width: int = SEG_CORE_WIDTH,
                  core_min_px: int = SEG_CORE_MIN_PX, min_fragment_px: int = SEG_MIN_FRAGMENT_PX) -> np.ndarray:
    """Predicted RGB -> int32 label map: 0 = unassigned (black), 1..M = predicted segments (one per colour cluster).

    1. black (max channel <= 24) is unassigned: void / background in the coding;
    2. the other colours are clustered in CIELAB (delta_e), with no vocabulary and no fixed number of segments;
    3. a cluster SEEDS a region if its morphological opening (core_width square) keeps >= core_min_px pixels; thin
       structures, i.e. the blends along a boundary between two colours, have no core;
    4. every other non-black pixel takes the label of the nearest core pixel, which restores real segments to their
       full extent and splits a blend between its two neighbours;
    5. connected fragments smaller than min_fragment_px are refilled the same way from the other labels.
    One id per colour cluster: a semantic class may be multi-region and an occluded instance is one instance; the
    labels are relabelled 1..M in order of first appearance (row-major), so the output does not depend on colours.
    Controls (cross_cutting/seg_labelmap_check_v4.json): recoloured GT 0.960 / 0.955 (semantic mIoU / PQ), blurred +
    noisy recoloured GT 0.881 / 0.868 against an oracle reference of 0.875 / 0.855."""
    raw = _colour_clusters(pred_u8, delta_e)
    fg = raw > 0
    if not fg.any():
        return raw
    seeds = np.zeros(raw.shape, np.int32)
    st = np.ones((core_width, core_width), bool)
    for k in np.unique(raw[fg]):
        core = ndimage.binary_opening(raw == k, structure=st)
        if int(core.sum()) >= core_min_px:
            seeds[core] = k
    if not (seeds > 0).any():                                  # nothing has a core (e.g. only speckle): raw clusters
        seeds = raw.copy()
    lab = np.where(fg, _fill_from(seeds, seeds > 0), 0)
    small = np.zeros(lab.shape, bool)
    for k in np.unique(lab[fg]):
        cc, n = ndimage.label(lab == k, structure=EIGHT)
        if n:
            sz = ndimage.sum(np.ones_like(cc), cc, index=np.arange(1, n + 1))
            small |= np.isin(cc, np.nonzero(sz < min_fragment_px)[0] + 1)
    if small.any() and (fg & ~small).any():
        lab = np.where(fg, _fill_from(lab, fg & ~small & (lab > 0)), 0)
    return relabel_first_appearance(lab)


def relabel_first_appearance(lab: np.ndarray) -> np.ndarray:
    """Labels > 0 renumbered 1..M in row-major order of first appearance; 0 stays 0."""
    flat = lab.reshape(-1)
    vals, first = np.unique(flat, return_index=True)
    order = [v for _, v in sorted(zip(first, vals)) if v > 0]
    lut = np.zeros(int(flat.max()) + 1 if flat.size else 1, np.int32)
    for i, v in enumerate(order, 1):
        lut[v] = i
    return lut[lab].astype(np.int32)


# ----------------------------------------------------------------------------- points
def point_blobs(pred_u8: np.ndarray, vocab_rgb: np.ndarray, *, min_blob_px: int = POINT_MIN_BLOB_PX,
                max_points: int | None = None) -> np.ndarray:
    """Predicted RGB -> (N, 2) float array of unnamed points (x, y) in tile pixels.

    Every non-black pixel is assigned to its nearest colour of `vocab_rgb` (the episode's coded vocabulary), with NO
    rejection radius; connected components (8-connectivity) of each colour of at least min_blob_px pixels are
    candidate points, at their pixel centroid. The colour only SEPARATES touching discs (a colour-blind labelling
    merges them: GT self-score 0.436 on faces against 0.998, cross_cutting/keypoint_extraction_check_v1.json); it
    never names a point.
    ONE PREDICTION PER GT POINT: with `max_points` = the number of scored GT points, only the max_points largest
    candidates are kept (ties in a fixed order). Identity-free PCK / NME are recall measures, so without the cap a
    prediction with blobs everywhere wins: copy-the-photo scored PCK@0.2 0.827 against TA 0.546 on bench_v2_val pose.
    With the cap copy falls to chance (pose 0.143 vs uniform points 0.116; faces NME 0.332 vs 0.349) and the GT
    itself stays perfect (cross_cutting/point_cap_check_v1.json). Pixel indices are the coordinates (x = column,
    y = row), the convention of the renderer's disc centres (round(x), round(y))."""
    a = np.asarray(pred_u8, np.uint8)
    fg = ~is_black(a)
    if not fg.any():
        return np.zeros((0, 2))
    voc = np.asarray(vocab_rgb, np.int64).reshape(-1, 3)
    px = a[fg].astype(np.int64)
    idx = np.empty(px.shape[0], np.int64)
    for s in range(0, px.shape[0], 16384):
        idx[s:s + 16384] = ((px[s:s + 16384, None, :] - voc[None]) ** 2).sum(-1).argmin(1)
    lab = np.full(a.shape[:2], -1, np.int64); lab[fg] = idx
    pts, areas = [], []
    rows, cols = np.indices(a.shape[:2])
    for c in np.unique(idx):
        cc, n = ndimage.label(lab == c, structure=EIGHT)
        if not n:
            continue
        ix = np.arange(1, n + 1)
        sizes = ndimage.sum(np.ones_like(cc), cc, index=ix)
        cy = ndimage.mean(rows, cc, index=ix); cx = ndimage.mean(cols, cc, index=ix)
        for sz, y, x in zip(sizes, cy, cx):
            if sz >= min_blob_px:
                pts.append((float(x), float(y))); areas.append(float(sz))
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if max_points is not None and len(pts) > max_points:
        keep = np.argsort(-np.asarray(areas), kind="stable")[:max_points]
        pts = pts[np.sort(keep)]
    return pts


# ----------------------------------------------------------------------------- dense fields
def depth_grey(pred_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (mean(R, G, B) / 255 as float64, predicted-invalid mask = exact black (all three channels 0)).
    Code 1 (RGB 1, 1, 1) is the far end of a valid prediction, so the invalid test must be exact."""
    a = np.asarray(pred_u8)
    return a.astype(np.float64).mean(-1) / 255.0, (a == 0).all(-1)


def decode_normals_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """The renderer's RGB = trunc((n + 1) / 2 * 255) inverted at the bin centre: n = (rgb + 0.5) / 127.5 - 1
    (not normalised). The truncation-centred decode halves the quantisation error (0.17 deg vs 0.34 deg mean)."""
    return (np.asarray(rgb_u8, np.float64) + 0.5) / 127.5 - 1.0


def normals_unit(pred_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (unit normals from decode_normals_rgb, predicted-invalid mask = exact black, detected on the raw bytes
    BEFORE decoding). The norm is floored at 1e-12."""
    a = np.asarray(pred_u8)
    n = decode_normals_rgb(a)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-12), (a == 0).all(-1)


def luma01(rgb_u8: np.ndarray) -> np.ndarray:
    """BT.601 luma in [0, 1], as metrics_v2._luma01 (the saliency and edge decoders)."""
    a = np.asarray(rgb_u8, np.float64) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def inpaint_composite(pred_u8: np.ndarray, input_u8: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """The inpainting convention: the prediction inside the hole, the given (known) pixels outside."""
    h = np.asarray(hole, bool)[..., None]
    return np.where(h, np.asarray(pred_u8, np.uint8), np.asarray(input_u8, np.uint8)).astype(np.uint8)
