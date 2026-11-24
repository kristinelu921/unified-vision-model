"""eval v2 scoring, v1: one prediction of one query -> its eval v2 metrics (the task's primary and secondaries).

`score_prediction(task, dataset, pred_u8, ctx)` is the single entry point used by tools/eval_v2_score_v1.py. `ctx`
carries the query's standard-format GT (from gt_v1) and what the post-processors need (the episode vocabulary, the
query input, the inpainting hole). The protocol constants (primaries, per-dataset normalisers, tolerances, ranges)
live in protocol_v1 and are imported, never redefined here. Keys starting with "_" are per-image accumulators for
the dataset-level official forms computed by tools/eval_v2_readout_v1.py (BSDS ODS / OIS / AP, PySOD curves, pooled
PQ, pooled normals).
"""
from __future__ import annotations

import numpy as np

from jflux2.fasteval.eval_v2 import metrics_v1 as M
from jflux2.fasteval.eval_v2 import postprocess_v1 as PP
from jflux2.fasteval.eval_v2 import protocol_v1 as P

APPEARANCE = ("colorization", "inpainting", "super_resolution", "denoising", "deblurring")


def _seg(task, pred, ctx):
    """Semantic: void = the GT's black (class 0). Instance: void = the source's void / unlabelled pixels from the
    sidecar (VOC 255 band, NYU class 0); the GT background is a real (scored) region, never void."""
    lab = PP.seg_label_map(pred)
    gt, void = ctx["gt_label"], ctx.get("gt_void")
    o = M.seg_overlap(gt, lab, void)
    q = M.panoptic_quality(gt, lab, void)
    den = q["tp"] + q["fp"] + q["fn"]
    out = {"miou": o["miou"], "fw_iou": o["fw_iou"], "pq": q["pq"], "sq": q["sq"], "rq": q["rq"],
           "dsb_ap_50": (q["tp"] / den) if den else float("nan"),            # DSB / StarDist "AP" = TP/(TP+FP+FN)
           "n_gt": o["n_gt"], "n_pred": o["n_pred"],
           "_tp": q["tp"], "_fp": q["fp"], "_fn": q["fn"], "_tp_iou_sum": q["tp_iou_sum"]}
    if task == "semantic_segmentation":
        out["pixel_acc"] = o["pixel_acc"]
        out["ade_score"] = 0.5 * (o["pixel_acc"] + o["miou"])                 # ADE20K's (pixel acc + mIoU) / 2
    else:
        f = M.f1_iou(gt, lab, void)
        out["f1_50"] = f["f1_50"]; out["f1_50_95"] = f["f1_50_95"]
    return out


def _saliency(pred, ctx):
    p01, gt = M.saliency_prepare(pred, ctx["target"])
    return M.saliency_curves(p01, gt)


def _edges(dataset, pred, ctx):
    tol = P.edge_tolerance_px(dataset, ctx["target"].shape)
    counts = M.edge_counts_v2(PP.luma01(pred), ctx["gt_edge_thin"], tol)
    o = M.ois(counts)
    return {"ois_f": o["f"], "ois_precision": o["precision"], "ois_recall": o["recall"], "tol_px": tol,
            "_edge_counts": counts.tolist()}


def _depth(dataset, pred, ctx):
    grey, pinv = PP.depth_grey(pred)
    scored = ctx["gt_scored"] & P.depth_eval_region(dataset, ctx["target"].shape)
    rng = P.DEPTH_RANGE_M.get(dataset)
    if rng:
        with np.errstate(divide="ignore"):
            d = 1.0 / np.asarray(ctx["gt_inv"], np.float64)
        scored = scored & (d >= rng[0]) & (d <= rng[1])
    r = M.depth_midas(grey, pinv, ctx["gt_inv"], scored, min_inv=P.depth_min_inv(dataset, ctx))
    r["n_scored"] = r.pop("n_valid", None)
    return r


def _normals(pred, ctx):
    unit, pinv = PP.normals_unit(pred)
    ang = M.normal_angles(unit, pinv, ctx["gt_unit"], ctx["gt_valid"], invalid_deg=P.NORMALS_INVALID_DEG)
    gv = ctx["gt_valid"]
    out = M.normal_metrics(ang, coverage=float(1.0 - (pinv & gv).sum() / max(gv.sum(), 1)))
    out["_hist_01deg"] = np.histogram(ang, bins=1800, range=(0.0, 180.0))[0].tolist()   # pooled (DSINE-style) stats
    out["_sum"] = float(ang.sum()); out["_sumsq"] = float((ang * ang).sum()); out["_n"] = int(ang.size)
    out["_n_within"] = [int((ang < t).sum()) for t in M.NORMAL_THRESHOLDS]
    return out


def _points(task, dataset, pred, ctx):
    gxy, rec = ctx["gt_xy"], ctx["gt_rec"]
    n_gt = int(len(gxy))
    allp = PP.point_blobs(pred, ctx["vocab_rgb"])
    pts = PP.point_blobs(pred, ctx["vocab_rgb"], max_points=n_gt)          # one prediction per GT point
    out = {"n_pred_points": int(len(allp)), "n_gt_points": n_gt}
    nme_name = P.point_nme_name(task, dataset)
    for name, (norm_kind, alpha) in P.point_scores(task, dataset).items():
        norm = P.point_norm(norm_kind, rec, dataset)
        m = M.point_metrics(gxy, pts, norm, alpha=alpha, auc_max=P.POINT_AUC_MAX.get(norm_kind))
        out[name] = m["pck"]
        if name == nme_name:
            out.update({"nme": m["nme"], "nme_matched": m["nme_matched"], "coverage": m["coverage"],
                        "n_false_pos": m["n_false_pos"], "norm_px": norm, "_hits": m["pck"] * n_gt})
            if dataset == "flic_test":                                   # FLIC's reported joints (elbows, wrists)
                out["pck_torso_0.2_elbows_wrists"] = _subset_pck(gxy, pts, rec, norm, alpha, P.FLIC_SUBSET)
    return out


def _subset_pck(gxy, pts, rec, norm, alpha, slots):
    """Hit rate on the GT points of the given slots, from the SAME one-to-one matching of all points."""
    scored = [p for p in rec["points"] if p.get("drawn") and p.get("inside_tile") and p.get("visible_in_raster")]
    idx = [i for i, p in enumerate(scored) if int(p["slot"]) in slots]
    if not idx:
        return float("nan")
    pairs, _, _ = M.match_points(gxy, pts, 0.5 * norm)
    hit = {i for i, _, d in pairs if d <= alpha * norm}
    return float(np.mean([i in hit for i in idx]))


def _appearance(task, dataset, pred, ctx):
    t = ctx["target"]; p = np.asarray(pred, np.uint8)
    out = {}
    if task == "inpainting":                      # the inpainting convention: score the COMPOSITE, whole image
        hole = ctx["hole"]
        comp = PP.inpaint_composite(p, ctx["input"], hole)
        out.update({"lpips": M.lpips(comp, t), "psnr": M.psnr(comp, t), "ssim": M.ssim_rgb(comp, t),
                    "l1": float(np.abs(comp.astype(np.float64) - t).mean() / 255.0),
                    "hole_psnr": M.psnr(p, t, mask=hole),
                    "hole_mae": float(np.abs(p[hole].astype(np.float64) - t[hole]).mean() / 255.0),
                    "outside_leak_mae": float(np.abs(p[~hole].astype(np.float64) - ctx["input"][~hole]).mean() / 255.0),
                    "lpips_uncomposited": M.lpips(p, t)})
        return out
    out.update({"lpips": M.lpips(p, t), "psnr": M.psnr(p, t), "ssim": M.ssim_rgb(p, t)})
    if task == "colorization":
        out.update({"chroma_psnr": M.chroma_psnr(p, t), "ab_psnr": M.ab_psnr(p, t), "ciede2000": M.ciede2000_mean(p, t),
                    "colorfulness": M.colorfulness(p), "delta_colorfulness": abs(M.colorfulness(p) - M.colorfulness(t)),
                    "auc_ab": M.auc_ab(p, t)})
    if task == "super_resolution":                # the SR convention: Y channel, crop = scale
        c = P.PSNR_Y_CROP["super_resolution"]
        out.update({"psnr_y": M.psnr_y(p, t, crop=c), "ssim_y": M.ssim_y(p, t, crop=c)})
        if dataset in ("div2k", "div2k_val"):    # DIV2K / NTIRE 2017: RGB, crop 6 + scale
            out["psnr_rgb_ntire"] = M.psnr_rgb_crop(p, t, P.DIV2K_NTIRE_CROP)
    if task == "deblurring" and dataset in P.REALBLUR_DATASETS:   # RealBlur's own ECC-aligned PSNR / SSIM
        out.update(M.realblur_ecc(p, t))
    return out


def score_prediction(task: str, dataset: str, pred_u8: np.ndarray, ctx: dict) -> dict:
    pred = np.asarray(pred_u8, np.uint8)
    if pred.shape != ctx["target"].shape:
        raise ValueError(f"prediction shape {pred.shape} != target {ctx['target'].shape}")
    if task in ("semantic_segmentation", "instance_segmentation"):
        return _seg(task, pred, ctx)
    if task == "saliency_detection":
        return _saliency(pred, ctx)
    if task == "edge_detection":
        return _edges(dataset, pred, ctx)
    if task == "relative_depth":
        return _depth(dataset, pred, ctx)
    if task == "surface_normals":
        return _normals(pred, ctx)
    if task in ("human_pose_2d", "face_landmarks"):
        return _points(task, dataset, pred, ctx)
    if task in APPEARANCE:
        return _appearance(task, dataset, pred, ctx)
    raise ValueError(f"eval v2 has no protocol for task {task}")
