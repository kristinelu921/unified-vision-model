"""eval v2 metrics, v1: the standard metrics, class- and identity-agnostic where the task has labels.

Inputs are already post-processed (postprocess_v1) and the ground truth is in standard format (gt_v1). Every function
scores ONE image; dataset-level pooled variants (ODS, pooled normals) take per-image counts.
Definitions and the standards they follow: docs/eval_v2_2026-10-06/EVAL_V2.md (written with the literature check).

  segmentation  seg_overlap: Hungarian-matched mIoU + pixel accuracy (the unsupervised-segmentation protocol, per
                image); panoptic_quality: class-agnostic PQ / SQ / RQ with void handling (Kirillov et al. 2019);
                f1_iou: F1 at IoU thresholds (".5:.95" = the COCO grid)
  points        match_points: one-to-one assignment on distance with outliers at the gate; pck; nme
  depth         depth_midas: least-squares scale + shift in inverse depth, then depth-space AbsRel / delta / RMSE
  normals       normal_angles / normal_metrics: angular error, predicted-invalid charged invalid_deg (180)
  edges         edge_counts: per-threshold one-to-one maximum matching within a pixel tolerance; ois / ods / ap
  appearance    psnr, ssim_rgb, psnr_y, chroma_psnr, ab_psnr, ciede2000, lpips
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from jflux2.fasteval import metrics_v2 as M2
from jflux2.fasteval import metrics_v3 as M3
from jflux2.fasteval.eval_v2 import postprocess_v1 as PP

_trapz = getattr(np, "trapezoid", None) or np.trapz          # numpy >= 2 renamed trapz


# ============================================================================ segmentation
def _contingency(gt_lab: np.ndarray, pred_lab: np.ndarray, valid: np.ndarray):
    """-> (inter [G+1, P+1], gt area [G+1], pred area [P+1]) over `valid` pixels; index 0 = unlabelled."""
    g = np.asarray(gt_lab, np.int64)[valid]; p = np.asarray(pred_lab, np.int64)[valid]
    G, P = int(g.max(initial=0)), int(p.max(initial=0))
    inter = np.bincount(g * (P + 1) + p, minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1)
    return inter, inter.sum(1), inter.sum(0)


def seg_overlap(gt_lab: np.ndarray, pred_lab: np.ndarray, void: np.ndarray | None = None) -> dict:
    """Hungarian-matched mIoU and pixel accuracy, per image (class-agnostic).

    GT segments = the labels > 0 of gt_lab; predicted segments = the labels > 0 of pred_lab (0 = unassigned). Void
    pixels (`void`, e.g. the semantic GT's black) are excluded from every count. One-to-one assignment maximising the
    total IoU; mIoU = mean over GT segments (an unmatched GT segment scores 0); pixel accuracy = share of non-void
    pixels whose predicted segment is matched to their GT segment."""
    valid = np.ones(np.shape(gt_lab), bool) if void is None else ~np.asarray(void, bool)
    inter, ag, ap = _contingency(gt_lab, pred_lab, valid)
    gids = [g for g in range(1, inter.shape[0]) if ag[g] > 0]
    pids = [p for p in range(1, inter.shape[1]) if ap[p] > 0]
    out = {"n_gt": len(gids), "n_pred": len(pids)}
    if not gids:
        return out | {"miou": float("nan"), "pixel_acc": float("nan"), "fw_iou": float("nan")}
    if not pids:
        return out | {"miou": 0.0, "pixel_acc": 0.0, "fw_iou": 0.0}
    I = inter[np.ix_(gids, pids)].astype(np.float64)
    U = ag[gids][:, None] + ap[pids][None, :] - I
    iou = np.where(U > 0, I / np.maximum(U, 1), 0.0)
    ri, ci = linear_sum_assignment(-iou)
    per = np.zeros(len(gids)); per[ri] = iou[ri, ci]
    correct = float(I[ri, ci].sum())
    w = ag[gids].astype(np.float64)
    return out | {"miou": float(per.mean()), "pixel_acc": correct / float(valid.sum()),
                  "fw_iou": float((per * w).sum() / w.sum())}


def panoptic_quality(gt_lab: np.ndarray, pred_lab: np.ndarray, void: np.ndarray | None = None,
                     iou_thresh: float = 0.5) -> dict:
    """Class-agnostic PQ / SQ / RQ (Kirillov et al. 2019) with every segment of one class.

    Matching: a (GT, predicted) pair is a true positive when IoU > iou_thresh (>= 0.5 makes it unique). IoU is
    computed with GT-void pixels removed from the predicted segment. An unmatched predicted segment is a false
    positive unless more than half of its pixels are GT-void. An unmatched GT segment is a false negative."""
    shape = np.shape(gt_lab)
    void = np.zeros(shape, bool) if void is None else np.asarray(void, bool)
    allpx = np.ones(shape, bool)
    inter, ag, ap_all = _contingency(np.where(void, 0, gt_lab), pred_lab, allpx)   # GT-void pixels -> GT label 0
    ap_void = np.bincount(np.asarray(pred_lab, np.int64)[void], minlength=inter.shape[1]) if void.any() else \
        np.zeros(inter.shape[1], np.int64)
    ap_nonvoid = ap_all - ap_void[:inter.shape[1]]
    gids = [g for g in range(1, inter.shape[0]) if ag[g] > 0]
    pids = [p for p in range(1, inter.shape[1]) if ap_all[p] > 0]
    tp_iou, matched_p, matched_g = [], set(), set()
    for g in gids:
        for p in pids:
            i = float(inter[g, p])
            if i <= 0:
                continue
            u = float(ag[g] + ap_nonvoid[p] - i)
            iou = i / u if u > 0 else 0.0
            if iou > iou_thresh:
                tp_iou.append(iou); matched_p.add(p); matched_g.add(g)
    fp = sum(1 for p in pids if p not in matched_p and ap_void[p] <= 0.5 * ap_all[p])
    fn = sum(1 for g in gids if g not in matched_g)
    tp = len(tp_iou)
    denom = tp + 0.5 * fp + 0.5 * fn
    sq = float(np.mean(tp_iou)) if tp else 0.0
    rq = tp / denom if denom > 0 else float("nan")
    pq = (sum(tp_iou) / denom) if denom > 0 else float("nan")
    return {"pq": float(pq), "sq": sq, "rq": float(rq), "tp": tp, "fp": int(fp), "fn": int(fn),
            "tp_iou_sum": float(sum(tp_iou))}


IOU_GRID = tuple(np.round(np.arange(0.50, 0.96, 0.05), 2))


def f1_iou(gt_lab: np.ndarray, pred_lab: np.ndarray, void: np.ndarray | None = None,
           thresholds=IOU_GRID) -> dict:
    """F1 = RQ at each IoU threshold (unique matching, as panoptic_quality); the instance summary without scores."""
    f = {float(t): panoptic_quality(gt_lab, pred_lab, void, iou_thresh=float(t))["rq"] for t in thresholds}
    return {"f1_50": f[0.5], "f1_50_95": float(np.nanmean(list(f.values()))), "by_threshold": f}


# ============================================================================ points
def match_points(gt_xy: np.ndarray, pred_xy: np.ndarray, gate: float):
    """One-to-one assignment minimising the total distance, with every pair beyond `gate` treated as an outlier
    (cost capped at the gate, so no pair is forced past it). -> (pairs [(gt_i, pred_j, dist)], unmatched_gt,
    unmatched_pred)."""
    g = np.asarray(gt_xy, np.float64).reshape(-1, 2); p = np.asarray(pred_xy, np.float64).reshape(-1, 2)
    if len(g) == 0 or len(p) == 0:
        return [], list(range(len(g))), list(range(len(p)))
    d = np.sqrt(((g[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    ri, ci = linear_sum_assignment(np.minimum(d, gate))
    pairs = [(int(i), int(j), float(d[i, j])) for i, j in zip(ri, ci) if d[i, j] < gate]
    mg = {i for i, _, _ in pairs}; mp = {j for _, j, _ in pairs}
    return pairs, [i for i in range(len(g)) if i not in mg], [j for j in range(len(p)) if j not in mp]


def point_metrics(gt_xy: np.ndarray, pred_xy: np.ndarray, norm: float, *, alpha: float, gate_frac: float = 0.5,
                  auc_max: float | None = None) -> dict:
    """Identity-agnostic point scores for one image, normalised by `norm` (px).

    pck        share of GT points with a matched prediction within alpha * norm (an unmatched GT point is a miss)
    nme        mean over ALL GT points of (matched distance / norm), an unmatched GT point charged the gate (gate_frac)
    nme_matched, coverage (matched share), n_false_pos (unmatched predictions)
    auc        area under the CED / PCK curve over [0, auc_max] of the normalised error, divided by auc_max
               (auc_max defaults to alpha)."""
    g = np.asarray(gt_xy, np.float64).reshape(-1, 2)
    if len(g) == 0 or not (norm > 0):
        return {"pck": float("nan"), "nme": float("nan"), "nme_matched": float("nan"), "coverage": float("nan"),
                "n_false_pos": 0, "auc": float("nan"), "n_gt": len(g)}
    gate = gate_frac * norm
    pairs, miss, extra = match_points(g, pred_xy, gate)
    err = np.full(len(g), gate_frac)                       # normalised error per GT point; misses at the gate
    for i, _, d in pairs:
        err[i] = d / norm
    hit = np.array([err[i] <= alpha and i not in miss for i in range(len(g))])
    amax = alpha if auc_max is None else auc_max
    xs = np.linspace(0.0, amax, 101)
    curve = np.array([np.mean([(err[i] <= x) and (i not in miss) for i in range(len(g))]) for x in xs])
    auc = float(_trapz(curve, xs) / amax)
    return {"pck": float(hit.mean()), "nme": float(err.mean()),
            "nme_matched": float(np.mean([d / norm for _, _, d in pairs])) if pairs else float("nan"),
            "coverage": len(pairs) / len(g), "n_false_pos": len(extra), "auc": auc, "n_gt": len(g)}


# ============================================================================ relative depth
def depth_midas(pred_grey: np.ndarray, pred_invalid: np.ndarray, gt_inv: np.ndarray, gt_valid: np.ndarray, *,
                min_inv: float) -> dict:
    """The MiDaS protocol on one image (Ranftl et al. 2020): least-squares scale + shift of the prediction to the GT
    INVERSE depth on pixels valid in both, scale constrained >= 0 (an inverted convention is not rescued); the
    aligned inverse depth is floored at `min_inv` (the depth cap) and inverted; AbsRel, delta < 1.25^k and RMSE in
    depth space over every GT-valid pixel. Predicted-invalid pixels take the cap (the far end) and are counted.
    gt_valid must exclude GT pixels whose true value is unknown (invalid, clipped ends)."""
    gv = np.asarray(gt_valid, bool); pv = gv & ~np.asarray(pred_invalid, bool)
    out = {"n_valid": int(gv.sum()), "coverage": float(pv.sum() / max(gv.sum(), 1))}
    if gv.sum() == 0:
        return out | {k: float("nan") for k in ("delta1", "delta2", "delta3", "abs_rel", "rmse", "scale", "shift")}
    x = np.asarray(pred_grey, np.float64)[pv]; y = np.asarray(gt_inv, np.float64)[pv]
    if x.size >= 2 and np.ptp(x) > 0:
        A = np.stack([x, np.ones_like(x)], 1)
        s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    else:
        s, t = 0.0, float(y.mean()) if y.size else float(min_inv)
    if s < 0:
        s, t = 0.0, float(y.mean())
    aligned = np.full(gv.sum(), float(min_inv))
    aligned[pv[gv]] = s * np.asarray(pred_grey, np.float64)[pv] + t
    aligned = np.maximum(aligned, min_inv)
    dp = 1.0 / aligned; dg = 1.0 / np.asarray(gt_inv, np.float64)[gv]
    ratio = np.maximum(dp / dg, dg / dp)
    return out | {"delta1": float(np.mean(ratio < 1.25)), "delta2": float(np.mean(ratio < 1.25 ** 2)),
                  "delta3": float(np.mean(ratio < 1.25 ** 3)), "abs_rel": float(np.mean(np.abs(dp - dg) / dg)),
                  "rmse": float(np.sqrt(np.mean((dp - dg) ** 2))), "scale": float(s), "shift": float(t)}


# ============================================================================ surface normals
NORMAL_THRESHOLDS = (5.0, 7.5, 11.25, 22.5, 30.0)


def normal_angles(pred_unit: np.ndarray, pred_invalid: np.ndarray, gt_unit: np.ndarray, gt_valid: np.ndarray, *,
                  invalid_deg: float = 180.0) -> np.ndarray:
    """Angular error (deg) on GT-valid pixels; a predicted-invalid (black) pixel is charged invalid_deg."""
    gv = np.asarray(gt_valid, bool)
    dot = np.clip((np.asarray(pred_unit) * np.asarray(gt_unit)).sum(-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    ang = np.where(np.asarray(pred_invalid, bool), float(invalid_deg), ang)
    return ang[gv]


def normal_metrics(ang: np.ndarray, coverage: float | None = None) -> dict:
    a = np.asarray(ang, np.float64)
    if a.size == 0:
        return {"mean_deg": float("nan"), "median_deg": float("nan"), "rmse_deg": float("nan"), "n": 0,
                **{f"within_{t:g}": float("nan") for t in NORMAL_THRESHOLDS}}
    out = {"mean_deg": float(a.mean()), "median_deg": float(np.median(a)), "rmse_deg": float(np.sqrt((a * a).mean())),
           "n": int(a.size), **{f"within_{t:g}": float((a < t).mean()) for t in NORMAL_THRESHOLDS}}
    if coverage is not None:
        out["coverage"] = float(coverage)
    return out


# ============================================================================ edges
EDGE_THRESHOLDS_99 = tuple(np.round(np.linspace(0.01, 0.99, 99), 2))


def _max_matching(t_pts: np.ndarray, p_pts: np.ndarray, tol: float) -> int:
    """Size of a maximum one-to-one matching between GT and predicted edge pixels within `tol` (Hopcroft-Karp)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.spatial import cKDTree
    if len(t_pts) == 0 or len(p_pts) == 0:
        return 0
    nb = cKDTree(t_pts).query_ball_tree(cKDTree(p_pts), r=tol)
    rows = np.repeat(np.arange(len(t_pts)), [len(x) for x in nb])
    if rows.size == 0:
        return 0
    cols = np.concatenate([np.asarray(x, np.int64) for x in nb if len(x)])
    g = csr_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(len(t_pts), len(p_pts)))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


def edge_counts(pred01: np.ndarray, gt_thin: np.ndarray, tol_px: float, thresholds=EDGE_THRESHOLDS_99) -> np.ndarray:
    """Per threshold: (matched predicted px, predicted px, matched GT px, GT px), the BSDS accumulators, with the
    prediction binarised at the threshold and thinned (Zhang-Suen), and a ONE-TO-ONE maximum matching within tol_px
    (so matched predicted = matched GT). -> int array [len(thresholds), 4]."""
    p = np.asarray(pred01, np.float64)
    t_pts = np.argwhere(np.asarray(gt_thin, bool)); n_t = len(t_pts)
    out = np.zeros((len(thresholds), 4), np.int64)
    for k, th in enumerate(thresholds):
        pb = M2.zhang_suen_thin(p >= th)
        p_pts = np.argwhere(pb)
        m = _max_matching(t_pts, p_pts, tol_px)
        out[k] = (m, len(p_pts), m, n_t)
    return out


def _f(prec, rec):
    return 0.0 if prec + rec <= 0 else 2 * prec * rec / (prec + rec)


def ois(counts: np.ndarray) -> dict:
    """The image's best F over thresholds (OIS for one image) with its precision / recall."""
    best = (0.0, 0.0, 0.0, -1)
    for k, (mp, np_, mt, nt) in enumerate(np.asarray(counts)):
        pr = mp / np_ if np_ else (1.0 if nt == 0 else 0.0)
        rc = mt / nt if nt else 0.0
        f = _f(pr, rc)
        if f > best[0]:
            best = (f, pr, rc, k)
    return {"f": float(best[0]), "precision": float(best[1]), "recall": float(best[2]), "k": int(best[3])}


def ods_ap(all_counts: list) -> dict:
    """Dataset level: ODS = best F of the pooled counts over a common threshold; AP = area under the pooled PR curve
    (recall-ordered trapezoid, as the BSDS toolbox's interpolation)."""
    c = np.sum(np.asarray(all_counts), 0)
    prec = np.where(c[:, 1] > 0, c[:, 0] / np.maximum(c[:, 1], 1), 1.0)
    rec = np.where(c[:, 3] > 0, c[:, 2] / np.maximum(c[:, 3], 1), 0.0)
    f = np.array([_f(p, r) for p, r in zip(prec, rec)])
    order = np.argsort(rec)
    ap = float(_trapz(prec[order], rec[order]))
    return {"ods": float(f.max()), "ods_k": int(f.argmax()), "ap": ap}


# ============================================================================ appearance
def psnr(pred_u8, targ_u8, mask: np.ndarray | None = None) -> float:
    p = np.asarray(pred_u8, np.float64); t = np.asarray(targ_u8, np.float64)
    if mask is not None:
        m = np.asarray(mask, bool); p, t = p[m], t[m]
    mse = float(np.mean((p - t) ** 2)) if p.size else float("nan")
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim_rgb(pred_u8, targ_u8) -> float:
    """Wang et al. 2004 SSIM per RGB channel (11 x 11 Gaussian, sigma 1.5, valid region), averaged over channels."""
    p = np.asarray(pred_u8, np.float64) / 255.0; t = np.asarray(targ_u8, np.float64) / 255.0
    return float(np.mean([M2.ssim_gaussian(p[..., c], t[..., c]) for c in range(3)]))


def psnr_y(pred_u8, targ_u8, crop: int = 0) -> float:
    """PSNR on BT.601 Y (the SR convention), with `crop` border pixels removed."""
    yp = M2._y_bt601(pred_u8) * 255.0; yt = M2._y_bt601(targ_u8) * 255.0
    if crop:
        yp, yt = yp[crop:-crop, crop:-crop], yt[crop:-crop, crop:-crop]
    mse = float(np.mean((yp - yt) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def chroma_psnr(pred_u8, targ_u8) -> float:
    """metrics_v3's colorization composite: the GT luma with the predicted chroma, PSNR against the GT."""
    return float(M3.appearance_metrics(pred_u8, targ_u8, task="colorization")["psnr_chroma"])


def ab_psnr(pred_u8, targ_u8) -> float:
    """PSNR of the CIELAB a*, b* channels (peak 255, the convention of colorization work that reports it)."""
    a = PP.rgb_to_lab(pred_u8)[..., 1:]; b = PP.rgb_to_lab(targ_u8)[..., 1:]
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), elementwise over (..., 3) L*a*b* arrays."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360; h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, h1p); h2p = np.where((a2p == 0) & (b2 == 0), 0.0, h2p)
    dLp = L2 - L1; dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0.0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm = (L1 + L2) / 2; Cpm = (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hsum,
                   np.where(np.abs(h1p - h2p) <= 180, hsum / 2,
                            np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc = 1 + 0.045 * Cpm; Sh = 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def ciede2000_mean(pred_u8, targ_u8) -> float:
    return float(np.mean(ciede2000(PP.rgb_to_lab(pred_u8), PP.rgb_to_lab(targ_u8))))


def lpips(pred_u8, targ_u8) -> float:
    """The official LPIPS (v0.1, AlexNet), metrics_v3.lpips_alex; NaN where torch / the package is unavailable."""
    return float(M3.lpips_alex(np.asarray(pred_u8, np.uint8), np.asarray(targ_u8, np.uint8)))


def finite_or_nan(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


# ============================================================================ official-protocol additions (2026-10-06)
# Folded in from the literature / official-protocol check (docs/eval_v2_2026-10-06/standards/standards_v1.json).

def guo_hall_thin(mask: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Guo & Hall (1989) parallel thinning, the algorithm of MATLAB's bwmorph(..., 'thin', Inf) and
    skimage.morphology.thin, which the BSDS toolbox (edgesEvalImg) applies to the binarised PREDICTION."""
    b = np.asarray(mask, bool).copy()
    for _ in range(max_iter):
        changed = False
        for it in (0, 1):
            P = np.pad(b, 1).astype(np.int32)
            p1, p2, p3 = P[:-2, :-2], P[:-2, 1:-1], P[:-2, 2:]
            p8, p4 = P[1:-1, :-2], P[1:-1, 2:]
            p7, p6, p5 = P[2:, :-2], P[2:, 1:-1], P[2:, 2:]
            C = ((~p2.astype(bool)) & (p3 | p4).astype(bool)).astype(np.int32) + \
                ((~p4.astype(bool)) & (p5 | p6).astype(bool)).astype(np.int32) + \
                ((~p6.astype(bool)) & (p7 | p8).astype(bool)).astype(np.int32) + \
                ((~p8.astype(bool)) & (p1 | p2).astype(bool)).astype(np.int32)
            N1 = (p1 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
            N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p1)
            N = np.minimum(N1, N2)
            if it == 0:
                m = ((p2 | p3 | (1 - p5)) & p4)
            else:
                m = ((p6 | p7 | (1 - p1)) & p8)
            delete = b & (C == 1) & (N >= 2) & (N <= 3) & (m == 0)
            if delete.any():
                b[delete] = False
                changed = True
        if not changed:
            break
    return b


def edge_counts_v2(pred01: np.ndarray, gt_thin: np.ndarray, tol_px: float, thresholds=EDGE_THRESHOLDS_99) -> np.ndarray:
    """edge_counts with the BSDS toolbox's thinning of the prediction (Guo-Hall, bwmorph 'thin') at each threshold."""
    p = np.asarray(pred01, np.float64)
    t_pts = np.argwhere(np.asarray(gt_thin, bool)); n_t = len(t_pts)
    out = np.zeros((len(thresholds), 4), np.int64)
    prev_key, prev = None, None
    for k, th in enumerate(thresholds):
        pb_raw = p >= th
        key = pb_raw.tobytes()
        if key == prev_key:                                       # identical binarisation: reuse
            out[k] = prev; continue
        pb = guo_hall_thin(pb_raw)
        p_pts = np.argwhere(pb)
        m = _max_matching(t_pts, p_pts, tol_px)
        out[k] = prev = (m, len(p_pts), m, n_t); prev_key = key
    return out


def bsds_dataset_scores(all_counts: list) -> dict:
    """The BSDS toolbox's dataset scores from per-image counts [n_img][n_thr][4] (matched pred, pred, matched GT, GT):
    ODS = best F over thresholds of the counts pooled over images; OIS = ONE F from the counts pooled at each image's
    own best threshold (not a mean of per-image F); AP = precision interpolated at 100 recall points / 100 (the
    toolbox's convention, its acknowledged /100 kept for comparability)."""
    c = np.asarray(all_counts, np.float64)
    if c.size == 0:
        return {"ods": float("nan"), "ois": float("nan"), "ap": float("nan")}
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
    order = np.argsort(R)
    rs, ps = R[order], P[order]
    grid = np.linspace(0, 1, 101)
    pi = np.interp(grid, rs, ps, left=ps[0] if len(ps) else 0.0, right=0.0)
    pi[grid > rs.max()] = 0.0
    return {"ods": float(F.max()), "ods_threshold_index": int(F.argmax()), "ois": float(ois_),
            "ap": float(pi.sum() / 100.0)}


# ---------------------------------------------------------------------------- saliency (PySODMetrics conventions)
def saliency_prepare(pred_u8: np.ndarray, target_u8: np.ndarray):
    """-> (prediction min-max normalised to [0, 1] per image, GT binary = target > 128): PySODMetrics prepare_data."""
    p = PP.luma01(pred_u8)
    lo, hi = float(p.min()), float(p.max())
    p = (p - lo) / (hi - lo) if hi > lo else np.zeros_like(p)
    return p, np.asarray(target_u8)[..., 0] > 128


def _e_measure_binary(fm: np.ndarray, gt: np.ndarray) -> float:
    fm = fm.astype(np.float64); g = gt.astype(np.float64)
    n = fm.size
    if g.sum() == 0:
        enh = 1.0 - fm
    elif g.sum() == n:
        enh = fm
    else:
        pf = fm - fm.mean(); pg = g - g.mean()
        align = 2.0 * pf * pg / (pf * pf + pg * pg + 1e-12)
        enh = (align + 1.0) ** 2 / 4.0
    return float(enh.sum() / (n - 1 + 1e-12))


def saliency_curves(p01: np.ndarray, gt: np.ndarray, beta_sq: float = 0.3) -> dict:
    """Per image: the 256-threshold F-beta curve and E-measure curve (thresholds 0..255 / 255 on the normalised
    prediction, PySODMetrics' np.linspace(0, 256, 257) bins), adaptive F-beta (threshold min(2 mean, 1)), MAE,
    S-measure (alpha .5). Dataset max / mean F-beta and mean E-measure come from curves AVERAGED over images first
    (eval_v2_readout), the PySODMetrics / CODToolbox order."""
    q = np.clip(np.rint(p01 * 255.0), 0, 255).astype(np.int64)
    n_gt = float(gt.sum())
    h_all = np.bincount(q.ravel(), minlength=256); h_pos = np.bincount(q[gt].ravel(), minlength=256)
    pred_pos = np.cumsum(h_all[::-1])[::-1].astype(np.float64); tp = np.cumsum(h_pos[::-1])[::-1].astype(np.float64)
    prec = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1), 0.0); rec = tp / max(n_gt, 1.0)
    f = (1 + beta_sq) * prec * rec / np.maximum(beta_sq * prec + rec, 1e-12)
    e = np.array([_e_measure_binary(q >= k, gt) for k in range(256)])
    th = min(2.0 * float(p01.mean()), 1.0)
    pb = p01 >= th; tpa = float((pb & gt).sum())
    pa = tpa / max(float(pb.sum()), 1.0); ra = tpa / max(n_gt, 1.0)
    adaptive = (1 + beta_sq) * pa * ra / max(beta_sq * pa + ra, 1e-12)
    return {"max_fbeta": float(f.max()), "mean_fbeta": float(f.mean()), "adaptive_fbeta": float(adaptive),
            "mae": float(np.abs(p01 - gt).mean()), "s_measure": float(M2._s_measure(p01, gt)),
            "mean_emeasure": float(e.mean()), "max_emeasure": float(e.max()),
            "_f_curve": f.tolist(), "_e_curve": e.tolist()}


# ---------------------------------------------------------------------------- appearance, dataset conventions
def ssim_y(pred_u8, targ_u8, crop: int = 0) -> float:
    """SSIM on BT.601 Y (MATLAB rgb2ycbcr), border crop: the SR convention (BasicSR calculate_ssim, test_y_channel)."""
    yp = M2._y_bt601(pred_u8); yt = M2._y_bt601(targ_u8)
    if crop:
        yp, yt = yp[crop:-crop, crop:-crop], yt[crop:-crop, crop:-crop]
    return float(M2.ssim_gaussian(yp, yt))


def psnr_rgb_crop(pred_u8, targ_u8, crop: int) -> float:
    p = np.asarray(pred_u8, np.float64)[crop:-crop, crop:-crop]; t = np.asarray(targ_u8, np.float64)[crop:-crop, crop:-crop]
    mse = float(np.mean((p - t) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def _ssim_map_channel(x, y, sigma=1.5, truncate=3.5):
    from scipy.ndimage import gaussian_filter
    f = lambda a: gaussian_filter(a, sigma=sigma, truncate=truncate, mode="reflect")
    mx, my = f(x), f(y)
    vx, vy, vxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def realblur_ecc(pred_u8, targ_u8) -> dict:
    """RealBlur's official evaluation (rimchang/RealBlur evaluation_RealBlur_ecc.py, reproduced by Restormer):
    least-squares intensity scaling of the deblurred image, an ECC homography (100 iterations, Gaussian filter 5)
    warping it into the GT frame, then PSNR (data range 1) and Gaussian SSIM over the warp's valid mask, SSIM cropped
    by the window radius. ECC failures are reported, not hidden."""
    import cv2
    z = np.asarray(pred_u8, np.float32) / 255.0; x = np.asarray(targ_u8, np.float32) / 255.0
    zs = (float(np.sum(x * z)) / max(float(np.sum(z * z)), 1e-12)) * z
    warp = np.eye(3, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0)
    try:
        _, warp = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY),
                                       warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
        ok = True
    except cv2.error:
        ok = False
    h, w = x.shape[:2]
    zr = cv2.warpPerspective(zs, warp, (w, h), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    cr = cv2.warpPerspective(np.ones_like(zs, dtype=np.float32), warp, (w, h),
                             flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    zr = zr * cr; xr = x * cr
    err = float(np.sum((xr - zr) ** 2, dtype=np.float64) / max(float(np.sum(cr)), 1.0))
    psnr_ = float("inf") if err == 0 else float(10 * np.log10(1.0 / err))
    smap = np.stack([_ssim_map_channel(xr[..., c].astype(np.float64), zr[..., c].astype(np.float64)) for c in range(3)], -1)
    smap = smap * cr
    r = int(3.5 * 1.5 + 0.5)
    s = smap[r:-r, r:-r, :].sum((0, 1)) / np.maximum(cr[r:-r, r:-r, :].sum((0, 1)), 1.0)
    return {"psnr_ecc": psnr_, "ssim_ecc": float(np.mean(s)), "ecc_ok": bool(ok)}


def colorfulness(rgb_u8) -> float:
    """Hasler & Suesstrunk (2003) colourfulness, the metric DDColor / BigColor report."""
    a = np.asarray(rgb_u8, np.float64)
    rg = a[..., 0] - a[..., 1]; yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def auc_ab(pred_u8, targ_u8, max_dist: float = 150.0) -> float:
    """Zhang, Isola & Efros (2016) colorization AuC: share of pixels whose ab (CIELAB) distance to the GT is within a
    threshold, swept over [0, 150], area under the curve normalised to [0, 1]."""
    a = PP.rgb_to_lab(pred_u8)[..., 1:]; b = PP.rgb_to_lab(targ_u8)[..., 1:]
    d = np.sqrt(((a - b) ** 2).sum(-1)).ravel()
    ts = np.linspace(0.0, max_dist, 151)
    curve = np.array([(d <= t).mean() for t in ts])
    return float(_trapz(curve, ts) / max_dist)
