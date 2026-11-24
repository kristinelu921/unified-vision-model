"""eval v2 add-on, v1 (2026-10-07): the OFFICIAL BSDS500 multi-annotator edge counts.

The frozen eval v2 (f8219a3d) scores BSDS edges against ONE majority GT (gt_v1.edge_gt: Guo-Hall thin of the
agreement tile > 0.5). The BSDS benchmark (pdollar/edges edgesEvalImg.m; Arbelaez et al., PAMI 2011) instead, at each
threshold, matches the thinned prediction to EACH annotator's boundary map g_a separately (correspondPixels within
maxDist) and accumulates
    matched predicted = |predicted pixels matched to AT LEAST ONE annotator|,  predicted = |predicted pixels|,
    matched GT       = sum_a |matched pixels of g_a|,                         GT        = sum_a |g_a|.
User decision 2026-10-07 (ledger 48690da6): BSDS500 edges switch to this rule. The per-annotator thin GT in tile
space is the sidecar mt_eval_packs/bench_v2_gt_v1_bsds/ (tools/build_gt_sidecar_bsds_v1.py, verified byte-for-byte
against the published tiles).

`edge_counts_multi` returns the same int [n_thr, 4] layout as metrics_v1.edge_counts_v2, so metrics_v1.ois and
metrics_v1.bsds_dataset_scores apply unchanged. With one annotator it equals edge_counts_v2 exactly.

Documented deviation: correspondPixels is a min-cost assignment (CSA, distance-weighted, with outlier cost); this
module, like metrics_v1, uses a maximum-CARDINALITY one-to-one matching (Hopcroft-Karp). The matched-GT counts are
therefore maximal; WHICH predicted pixels are matched is one of the maximum matchings, so the union in the precision
numerator can depend on that choice (quantified in docs/eval_v2_2026-10-06/bsds_v1/).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from jflux2.fasteval.eval_v2 import metrics_v1 as M
from jflux2.fasteval.eval_v2.gt_v1 import PACKS_ROOT, _load_json

# Default = the logical location. Since 2026-10-07 the sidecar lives on GCS only
# (gs://kmh-gcp-us-west4/data/xtiange/mt_eval_packs/bench_v2_gt_v1_bsds/); tools/eval_v2_score_bsds_v1.py stages it,
# verifies it against docs/eval_v2_2026-10-06/bsds_v1/gt_bsds_manifest_v1.json and calls set_sidecar_root().
SIDECAR_ROOT_BSDS = PACKS_ROOT / "bench_v2_gt_v1_bsds"
SIDECAR_NAME = "gt_bsds_annotators_v1"
FORMAT = "bench_v2_gt.bsds_annotators.v1"


# ----------------------------------------------------------------------------- matching
def _matched_pred(t_pts: np.ndarray, t_tree, p_pts: np.ndarray, p_tree, tol: float) -> np.ndarray:
    """Indices into p_pts of the predicted pixels in a maximum one-to-one matching to t_pts within `tol` (the same
    graph and Hopcroft-Karp call as metrics_v1._max_matching, returning the matching instead of its size)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if len(t_pts) == 0 or len(p_pts) == 0:
        return np.zeros(0, np.int64)
    nb = t_tree.query_ball_tree(p_tree, r=tol)
    lens = np.fromiter((len(x) for x in nb), np.int64, len(nb))
    if lens.sum() == 0:
        return np.zeros(0, np.int64)
    rows = np.repeat(np.arange(len(t_pts)), lens)
    cols = np.concatenate([np.asarray(x, np.int64) for x in nb if len(x)])
    g = csr_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(len(t_pts), len(p_pts)))
    m = maximum_bipartite_matching(g, perm_type="column")           # per GT row: matched pred column or -1
    return m[m >= 0].astype(np.int64)


def edge_counts_multi(pred01: np.ndarray, annotators_thin, tol_px: float,
                      thresholds=M.EDGE_THRESHOLDS_99, pred_order_seed: int | None = None) -> np.ndarray:
    """Per threshold: (|union_a matched predicted px|, predicted px, sum_a matched GT px, sum_a GT px) -> int
    [len(thresholds), 4]. The prediction is binarised (>= t) and Guo-Hall thinned as in edge_counts_v2; identical
    binarisations at consecutive thresholds are reused. `pred_order_seed` (diagnostic only; default None = raster
    order, the metric of record) permutes the predicted pixels before matching, to measure how much the union
    depends on which maximum matching Hopcroft-Karp returns."""
    from scipy.spatial import cKDTree
    p = np.asarray(pred01, np.float64)
    gts = [np.argwhere(np.asarray(g, bool)) for g in annotators_thin]
    trees = [cKDTree(t) if len(t) else None for t in gts]
    n_gt = int(sum(len(t) for t in gts))
    out = np.zeros((len(thresholds), 4), np.int64)
    prev_key, prev = None, None
    for k, th in enumerate(thresholds):
        pb_raw = p >= th
        key = pb_raw.tobytes()
        if key == prev_key:
            out[k] = prev; continue
        p_pts = np.argwhere(M.guo_hall_thin(pb_raw))
        if pred_order_seed is not None:
            p_pts = p_pts[np.random.default_rng(pred_order_seed + k).permutation(len(p_pts))]
        p_tree = cKDTree(p_pts) if len(p_pts) else None
        hit = np.zeros(len(p_pts), bool)
        m_gt = 0
        for t, tt in zip(gts, trees):
            if tt is None or p_tree is None:
                continue
            mp = _matched_pred(t, tt, p_pts, p_tree, tol_px)
            hit[mp] = True
            m_gt += len(mp)
        out[k] = prev = (int(hit.sum()), len(p_pts), m_gt, n_gt); prev_key = key
    return out


# ----------------------------------------------------------------------------- sidecar loader
def sidecar_dir(pack_rel_: str) -> Path:
    """bench_v2_val -> <root>/bench_v2_val/edge_detection; a test pack_rel -> <root>/<pack_rel> (gt_v1's layout)."""
    return SIDECAR_ROOT_BSDS / pack_rel_ / "edge_detection" if pack_rel_ == "bench_v2_val" \
        else SIDECAR_ROOT_BSDS / pack_rel_


@lru_cache(maxsize=None)
def _annotator_sidecar(pack_rel_: str, pack_id: str | None):
    d = sidecar_dir(pack_rel_)
    m = _load_json(str(d / f"{SIDECAR_NAME}.json"))
    if m.get("format") != FORMAT:
        raise ValueError(f"{d}: format {m.get('format')} != {FORMAT}")
    if pack_id is not None and m.get("pack_id") not in (None, pack_id):
        raise ValueError(f"sidecar pack_id {m.get('pack_id')} != {pack_id}")
    bad = [r["record_id"] for r in m["records"] if not (r.get("verified") and r.get("max_abs_diff") == 0)]
    if bad:
        raise ValueError(f"{d}: {len(bad)} unverified records")
    with np.load(d / f"{SIDECAR_NAME}.npz") as z:
        thin, n_ann = z["thin"], z["n_annotators"]
    if thin.shape[0] != len(m["records"]) or [int(x) for x in n_ann] != [r["n_annotators"] for r in m["records"]]:
        raise ValueError(f"{d}: npz rows / annotator counts disagree with the json")
    return thin, n_ann, {r["record_id"]: i for i, r in enumerate(m["records"])}


def bsds_annotators(pack_rel_: str, record_id: str, pack_id: str | None = None) -> list:
    """The record's per-annotator thin tile-space GT: a list of n_annotators bool [256, 256] arrays."""
    thin, n_ann, order = _annotator_sidecar(pack_rel_, pack_id)
    k = order[record_id]
    return [np.asarray(thin[k, a], bool) for a in range(int(n_ann[k]))]


def set_sidecar_root(root) -> None:
    """Point the loader at a staged, verified copy of the bench_v2_gt_v1_bsds release (clears the caches)."""
    global SIDECAR_ROOT_BSDS
    SIDECAR_ROOT_BSDS = Path(root)
    _annotator_sidecar.cache_clear()
    _load_json.cache_clear()


def has_bsds_annotators(pack_rel_: str) -> bool:
    return (sidecar_dir(pack_rel_) / f"{SIDECAR_NAME}.json").exists()
