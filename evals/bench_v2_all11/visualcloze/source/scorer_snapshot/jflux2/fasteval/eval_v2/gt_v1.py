"""eval v2 ground truth, v1: GT in standard format from the bench_v2 target tiles and the bench_v2_gt_v1 sidecars.

Tile-derived (lossless; the tiles are flat-coded rasters, nearest-resized for the palette tasks):
  semantic_gt   int32 class-agnostic label map (palette index; 0 = void) and the void mask
  instance_gt   int32 instance map (palette index = instance rank; 0 = background)
  saliency_gt   binary foreground (mask[..., 0] > 128, PySODMetrics)
  edge_gt       soft strength (mask / 255) and the thinned binary GT (soft > 0.5, Guo-Hall like the predictions)
  normals_gt    unit normals (truncation-centred decode) and validity (| |n| - 1 | < 0.35; black = no measurement)
Sidecar-derived (mt_eval_packs/bench_v2_gt_v1/<pack_rel>[/<task>]/...; built by tools/build_gt_sidecar_*_v1.py from
the sources, verified against the tiles):
  void_mask     instance packs: the source's void / unlabelled pixels (VOC 255 band, NYU class 0), background in tiles
  depth_gt      true inverse depth from the record's endpoints (lo, hi): inv = lo + (code - 1) / 254 * (hi - lo) for
                codes 2..254; code 1 / 255 (clipped ends) and 0 (invalid) have no known value and are not scored
  hole_mask     the exact inpainting hole
  points        exact tile coordinates of every annotated point, with drawn / visible flags
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from jflux2.dataloaders import multi_task_loader_v4 as CV4
from jflux2.fasteval import metrics as M1
from jflux2.fasteval import metrics_v2 as M2

PACKS_ROOT = Path("/kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_packs")
SIDECAR_ROOT = PACKS_ROOT / "bench_v2_gt_v1"


# ----------------------------------------------------------------------------- tile-derived
def palette_index_map(tile_u8: np.ndarray) -> np.ndarray:
    """Exact palette index per pixel (DISTINCT_PALETTE); raises if a pixel is not a palette colour."""
    pm = CV4._posmap(CV4._packed(np.asarray(tile_u8, np.int64)))
    if (pm < 0).any():
        raise ValueError(f"{int((pm < 0).sum())} target pixels are not palette colours")
    return pm.astype(np.int32)


def semantic_gt(tile_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = palette_index_map(tile_u8)
    return lab, lab == 0


def instance_gt(tile_u8: np.ndarray) -> np.ndarray:
    return palette_index_map(tile_u8)


def saliency_gt(tile_u8: np.ndarray) -> np.ndarray:
    """Binary GT = target > 128 (PySODMetrics / CODToolbox); the data card's >= 128 differs only at exactly 128."""
    return np.asarray(tile_u8)[..., 0] > 128


def edge_gt(tile_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (soft strength, thin binary GT). The tiles carry a 1-px-dilated, resized edge map, so the GT is binarised at
    0.5 and thinned with the SAME Guo-Hall thinning as the predictions (the BSDS toolbox's GT is already thin)."""
    from jflux2.fasteval.eval_v2.metrics_v1 import guo_hall_thin
    soft = np.asarray(tile_u8, np.float64)[..., 0] / 255.0
    return soft, guo_hall_thin(soft > 0.5)


def normals_gt(tile_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """-> (unit normals, validity). Decoded at the truncation bin centre (postprocess_v1.decode_normals_rgb) like the
    prediction; valid where | |n| - 1 | < 0.35 (black, no measurement, decodes to norm ~1.73)."""
    from jflux2.fasteval.eval_v2.postprocess_v1 import decode_normals_rgb
    n = decode_normals_rgb(tile_u8)
    norm = np.linalg.norm(n, axis=-1)
    return n / np.maximum(norm, 1e-12)[..., None], np.abs(norm - 1.0) < 0.35


def depth_gt(tile_u8: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (true inverse depth [1/m] where known, scored mask = codes 2..254, any-valid mask = codes >= 1)."""
    code = np.asarray(tile_u8)[..., 0].astype(np.int64)
    inv = lo + (code - 1).astype(np.float64) / 254.0 * (hi - lo)
    return inv, (code >= 2) & (code <= 254), code >= 1


# ----------------------------------------------------------------------------- sidecars
def pack_rel(pack_dir: str | Path) -> str:
    """'bench_v2_val' or 'bench_v2_test/<task>/<dataset>' for a pack directory under PACKS_ROOT."""
    return str(Path(pack_dir).resolve().relative_to(PACKS_ROOT.resolve()))


def sidecar_dir(pack_rel_: str, task: str) -> Path:
    return SIDECAR_ROOT / pack_rel_ / task if pack_rel_ == "bench_v2_val" else SIDECAR_ROOT / pack_rel_


@lru_cache(maxsize=None)
def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _records_by_id(meta: dict, pack_id: str | None) -> dict:
    if pack_id is not None and meta.get("pack_id") not in (None, pack_id):
        raise ValueError(f"sidecar pack_id {meta.get('pack_id')} != {pack_id}")
    return {r["record_id"]: r for r in meta["records"]}


@lru_cache(maxsize=None)
def depth_endpoints(pack_rel_: str, pack_id: str | None = None) -> dict:
    f = sidecar_dir(pack_rel_, "relative_depth") / "gt_depth_endpoints_v1.json"
    m = _load_json(str(f))
    recs = _records_by_id(m, pack_id)
    bad = [k for k, r in recs.items() if not (r.get("verified_tile_identical") and r.get("verified_reencode"))]
    if bad:
        raise ValueError(f"{f}: {len(bad)} unverified records")
    return recs


@lru_cache(maxsize=None)
def _hole_masks(pack_rel_: str, pack_id: str | None):
    d = sidecar_dir(pack_rel_, "inpainting")
    m = _load_json(str(d / "gt_inpaint_mask_v1.json"))
    holes = np.load(d / "gt_inpaint_mask_v1.npz")["hole"]
    recs = _records_by_id(m, pack_id)
    order = {r["record_id"]: i for i, r in enumerate(m["records"])}
    if any(not r.get("verified") for r in recs.values()):
        raise ValueError(f"{d}: unverified hole masks")
    return holes, order


def hole_mask(pack_rel_: str, record_id: str, pack_id: str | None = None) -> np.ndarray:
    holes, order = _hole_masks(pack_rel_, pack_id)
    return np.asarray(holes[order[record_id]], bool)


@lru_cache(maxsize=None)
def _void_masks(pack_rel_: str, task: str, pack_id: str | None):
    d = sidecar_dir(pack_rel_, task)
    f = d / "gt_void_mask_v1.json"
    if not f.exists():
        return None, None
    m = _load_json(str(f))
    recs = _records_by_id(m, pack_id)
    if any(not r.get("verified") for r in recs.values()):
        raise ValueError(f"{d}: unverified void masks")
    return np.load(d / "gt_void_mask_v1.npz")["void"], {r["record_id"]: i for i, r in enumerate(m["records"])}


def void_mask(pack_rel_: str, task: str, record_id: str, pack_id: str | None = None):
    """The source annotation's void / unlabelled pixels that the tile renders as background (VOC's 255 band; NYU's
    class 0), or None where the pack has no void sidecar. Official protocols exclude these pixels."""
    masks, order = _void_masks(pack_rel_, task, pack_id)
    if masks is None:
        return None
    return np.asarray(masks[order[record_id]], bool)


@lru_cache(maxsize=None)
def points_records(pack_rel_: str, task: str, pack_id: str | None = None) -> dict:
    f = sidecar_dir(pack_rel_, task) / "gt_points_v1.json"
    m = _load_json(str(f))
    return _records_by_id(m, pack_id)


def scored_points(rec: dict) -> np.ndarray:
    """(N, 2) exact (x, y) of the points a raster target can show: drawn, inside the tile, and visible in the raster
    (a fully overwritten disc is a representation limit, reported, not scored)."""
    pts = [(p["x"], p["y"]) for p in rec["points"] if p.get("drawn") and p.get("inside_tile")
           and p.get("visible_in_raster")]
    return np.asarray(pts, np.float64).reshape(-1, 2)


def point_xy(rec: dict, slot: int, person: int = 0):
    """Exact (x, y) of an annotated point (drawn or not), or None."""
    for p in rec["points"]:
        if int(p["slot"]) == int(slot) and int(p.get("person", 0)) == int(person) and p.get("x") is not None:
            return float(p["x"]), float(p["y"])
    return None
