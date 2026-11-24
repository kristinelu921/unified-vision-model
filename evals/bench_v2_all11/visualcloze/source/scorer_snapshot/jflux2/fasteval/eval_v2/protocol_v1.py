"""eval v2 protocol constants, v1: the metric of record per task, margins, and each DATASET's official conventions.

User decisions (2026-10-06): the metrics of record switch to these primaries; standard normalisers; predicted-invalid
normals are charged 180 deg; segmentation is class-agnostic and points identity-agnostic; each dataset's OFFICIAL
evaluation protocol is followed where it applies to the 256 x 256 tiles ("follow to some extend, if applicable to our
scenario"). Sources for every rule: docs/eval_v2_2026-10-06/standards/standards_v1.json; the rules and their
deviations are listed in docs/eval_v2_2026-10-06/EVAL_V2.md. Frozen (never edited) once the pre-registered
re-scoring amendment cites its hash.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- metric of record per task (per query)
PRIMARY = {                                   # task -> (metric key, lower is better)
    "semantic_segmentation": ("miou", False),       # Hungarian-matched mIoU, per image, void excluded
    "instance_segmentation": ("pq", False),         # class-agnostic PQ, per image, source void excluded
    "saliency_detection": ("max_fbeta", False),     # max F-beta (beta^2 .3) of the image's own curve
    "edge_detection": ("ois_f", False),             # the image's best F over 99 thresholds (one-to-one matching)
    "relative_depth": ("delta1", False),            # delta < 1.25 after scale + shift in inverse depth
    "surface_normals": ("mean_deg", True),          # mean angular error, predicted-invalid = 180 deg
    "human_pose_2d": ("pck_torso_0.2", False),      # PCK@0.2 of the dataset's torso, identity-free
    "face_landmarks": ("nme", True),                # NME of the dataset's normaliser, identity-free
    "colorization": ("lpips", True),
    "inpainting": ("lpips", True),                  # on the composite (inpainting convention)
    "super_resolution": ("lpips", True),
    "denoising": ("lpips", True),
    "deblurring": ("lpips", True),
}


def margin(task: str, copy_mean: float) -> float:
    """The registered margin rule, unchanged: LPIPS .05; every other metric max(5 % of |copy mean|, .02)."""
    if PRIMARY[task][0] == "lpips":
        return 0.05
    return max(0.05 * abs(copy_mean), 0.02)


# --------------------------------------------------------------------------- normals
NORMALS_INVALID_DEG = 180.0                   # user decision 3 (no precedent either way: bench_v2's anti-abstention rule)

# --------------------------------------------------------------------------- edges
EDGE_TOL_FRAC = {"bsds500": 0.0075, "bsds500_test": 0.0075,     # BSDS benchmark maxDist
                 "nyudv2_test_hed": 0.011}                       # NYUD (HED 4.3, RCF 4.2): larger images


def edge_tolerance_px(dataset: str, shape) -> float:
    """maxDist x the tile diagonal (the toolbox applies it to the evaluated image's own diagonal)."""
    return EDGE_TOL_FRAC.get(dataset, 0.0075) * math.hypot(shape[0], shape[1])


# --------------------------------------------------------------------------- depth
DEPTH_RANGE_M = {"nyu_depth_v2": (1e-3, 10.0), "nyuv2_test": (1e-3, 10.0),    # NYUv2 evaluation range
                 "diode": (0.6, 350.0), "diode_val": (0.6, 350.0)}             # DIODE sensor range
NYU_DEPTH_DATASETS = ("nyu_depth_v2", "nyuv2_test")


def depth_eval_region(dataset: str, shape) -> np.ndarray:
    """NYUv2: the Eigen crop (rows 45:471, cols 41:601 of the 480 x 640 frame) mapped into the tile. The tile is the
    whole frame's height (short side 480 -> 256) and its columns are frame columns 80..560 (inside 41..601), so the
    crop keeps tile rows 24..251 and every column; it removes the frame's white border (constant white rows 0 and 255
    measured on the test pack) where the GT depth is filled, not measured. Other datasets: the whole tile."""
    m = np.ones(shape[:2], bool)
    if dataset in NYU_DEPTH_DATASETS:
        m[:24] = False; m[252:] = False
    return m


def depth_min_inv(dataset: str, ctx: dict) -> float:
    """The cap on predicted depth as an inverse-depth floor: the image's own far clip lo (no scored GT pixel is
    farther; a fixed metric cap cannot apply to percentile-coded GT), and never below 1 / the dataset's max range."""
    lo = float(ctx["lo"])
    rng = DEPTH_RANGE_M.get(dataset)
    return max(lo, 1.0 / rng[1]) if rng else lo


# --------------------------------------------------------------------------- points
FACE_NORMALISER = {"300w": "inter_ocular", "300w_test": "inter_ocular",   # 300-W: outer eye corners 36 / 45
                   "cofw68_test": "inter_pupil"}                           # COFW-68's own benchmark (Ghiasi & Fowlkes)
POSE_TORSO = {"lsp_test": (13, 2),             # LSP: left shoulder - right hip (Sapp & Taskar; bearpaw/eval_pose)
              "flic_test": (3, 12),            # FLIC: left hip - right shoulder (MODEC 6.2)
              "mpii": (13, 2)}                 # MPII has no torso standard (its own is PCKh); LSP's diagonal is used
FLIC_SUBSET = (10, 11, 14, 15)                 # FLIC's reported joints: wrists and elbows (Tompson et al. 2014)
FACE_FR_THRESHOLDS = (0.08, 0.10)              # 300-W (MDM) and the later 0.10
POINT_AUC_MAX = {"inter_ocular": 0.1, "inter_pupil": 0.1, "torso": 0.5, "head_segment": 0.5}


def point_scores(task: str, dataset: str) -> dict:
    """name -> (normaliser kind, alpha). The FIRST entry's normaliser also defines the reported NME."""
    if task == "face_landmarks":
        k = FACE_NORMALISER.get(dataset, "inter_ocular")
        return {f"pck_{k}_0.1": (k, 0.10), f"pck_{k}_0.08": (k, 0.08)}
    out = {"pck_torso_0.2": ("torso", 0.20), "pck_torso_0.1": ("torso", 0.10)}
    if dataset in ("mpii", "lsp_test"):        # head joints annotated: PCKh with the head SEGMENT (MPII's head box
        out["pckh_0.5_segment"] = ("head_segment", 0.50)    # needs mpii_gt_val.mat, not available: a proxy)
    return out


def point_nme_name(task: str, dataset: str) -> str:
    return next(iter(point_scores(task, dataset)))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_norm(kind: str, rec: dict, dataset: str = "") -> float:
    """Normaliser in tile pixels from the record's EXACT annotated points (gt_v1.point_xy; drawn or not)."""
    from jflux2.fasteval.eval_v2.gt_v1 import point_xy
    if kind == "inter_ocular":                     # iBUG 36 / 45: the outer eye corners
        a, b = point_xy(rec, 36), point_xy(rec, 45)
        return _dist(a, b) if a and b else float("nan")
    if kind == "inter_pupil":                      # the eye centres: means of iBUG 36-41 and 42-47
        r = [point_xy(rec, i) for i in range(36, 42)]; l_ = [point_xy(rec, i) for i in range(42, 48)]
        if not all(r) or not all(l_):
            return float("nan")
        cr = (np.mean([p[0] for p in r]), np.mean([p[1] for p in r]))
        cl = (np.mean([p[0] for p in l_]), np.mean([p[1] for p in l_]))
        return _dist(cr, cl)
    if kind == "torso":
        a, b = POSE_TORSO.get(dataset, (13, 2))
        pa, pb = point_xy(rec, a), point_xy(rec, b)
        return _dist(pa, pb) if pa and pb else float("nan")
    if kind == "head_segment":                     # upper_neck 8 - head_top 9 (LSP: neck - head top)
        a, b = point_xy(rec, 8), point_xy(rec, 9)
        return _dist(a, b) if a and b else float("nan")
    raise ValueError(f"unknown normaliser {kind}")


# --------------------------------------------------------------------------- appearance, per-dataset conventions
SR_SCALE = 4
PSNR_Y_CROP = {"super_resolution": SR_SCALE}  # SR: Y channel, border crop = scale (BasicSR / the classical protocol)
DIV2K_NTIRE_CROP = 6 + SR_SCALE               # DIV2K (NTIRE 2017, EDSR 4.4): RGB, crop 6 + scale (moderate confidence)
REALBLUR_DATASETS = ("realblur_j_test",)      # RealBlur's own ECC-aligned PSNR / SSIM
