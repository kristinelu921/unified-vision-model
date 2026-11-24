"""BSDS multi-annotator edge counts, v2: v1 (edges_bsds_v1) with a matching algorithm of guaranteed complexity.

WHY. scipy.sparse.csgraph.maximum_bipartite_matching hits pathological cases on edge-correspondence graphs: on
bench_v2_val bsds500:226043 (GT tile as the prediction, threshold 0.36, annotator 4; graph 3,376 x 3,143, 14,711
edges) one call took 141 s, while every other call took milliseconds. Splitting the graph into connected components
does not help (the case lives inside one component). networkx's Hopcroft-Karp (O(E sqrt V)) returns a matching of the
same, unique maximum size in 0.11 s. The maximum-matching SIZE is unique, so every count that depends only on sizes
(matched GT pixels per annotator; the majority metric's matched counts) is identical to v1 / metrics_v1. The union of
matched predicted pixels across annotators depends on WHICH maximum matching is returned (documented in v1; the
pred_order_seed diagnostic measures it), so v2's precision numerator may differ from v1's within that sensitivity.

API: as v1 (edge_counts_multi, bsds_annotators, set_sidecar_root, has_bsds_annotators), plus max_matching_size, a
drop-in for metrics_v1._max_matching (identical return value).
"""
from __future__ import annotations

import numpy as np

from jflux2.fasteval.eval_v2 import edges_bsds_v1 as V1
from jflux2.fasteval.eval_v2 import metrics_v1 as M

bsds_annotators = V1.bsds_annotators
set_sidecar_root = V1.set_sidecar_root
has_bsds_annotators = V1.has_bsds_annotators


def _hk_matching(t_pts: np.ndarray, p_pts: np.ndarray, tol: float, t_tree=None, p_tree=None) -> np.ndarray:
    """-> int array of the predicted indices matched in a maximum one-to-one matching of t_pts to p_pts within tol
    (Hopcroft-Karp, networkx; deterministic for fixed inputs)."""
    import networkx as nx
    from scipy.spatial import cKDTree
    if len(t_pts) == 0 or len(p_pts) == 0:
        return np.zeros(0, np.int64)
    t_tree = t_tree if t_tree is not None else cKDTree(t_pts)
    p_tree = p_tree if p_tree is not None else cKDTree(p_pts)
    nb = t_tree.query_ball_tree(p_tree, r=tol)
    nt = len(t_pts)
    G = nx.Graph()
    G.add_nodes_from(range(nt))
    G.add_nodes_from(range(nt, nt + len(p_pts)))
    G.add_edges_from((i, nt + int(j)) for i, js in enumerate(nb) for j in js)
    m = nx.bipartite.hopcroft_karp_matching(G, top_nodes=range(nt))
    return np.asarray(sorted(v - nt for k, v in m.items() if k < nt), np.int64)


def max_matching_size(t_pts: np.ndarray, p_pts: np.ndarray, tol: float) -> int:
    """Drop-in for metrics_v1._max_matching: the size of a maximum one-to-one matching within tol (unique)."""
    return int(len(_hk_matching(np.asarray(t_pts), np.asarray(p_pts), tol)))


def edge_counts_multi(pred01: np.ndarray, annotators_thin, tol_px: float,
                      thresholds=M.EDGE_THRESHOLDS_99, pred_order_seed: int | None = None) -> np.ndarray:
    """v1's counts (|union_a matched predicted px|, predicted px, sum_a matched GT px, sum_a GT px) per threshold,
    with the Hopcroft-Karp matching."""
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
            out[k] = prev
            continue
        p_pts = np.argwhere(M.guo_hall_thin(pb_raw))
        if pred_order_seed is not None:
            p_pts = p_pts[np.random.default_rng(pred_order_seed + k).permutation(len(p_pts))]
        p_tree = cKDTree(p_pts) if len(p_pts) else None
        hit = np.zeros(len(p_pts), bool)
        m_gt = 0
        for t, tt in zip(gts, trees):
            if tt is None or p_tree is None:
                continue
            mp = _hk_matching(t, p_pts, tol_px, tt, p_tree)
            hit[mp] = True
            m_gt += len(mp)
        out[k] = prev = (int(hit.sum()), len(p_pts), m_gt, n_gt)
        prev_key = key
    return out
