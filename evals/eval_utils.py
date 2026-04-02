from __future__ import annotations

import os
from typing import Iterator, Sequence

import numpy as np


def walk_files(root: str, *, exts: Sequence[str]) -> Iterator[str]:
    exts_l = tuple(e.lower() for e in exts)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts_l):
                print(os.path.join(dirpath, fn))
                yield os.path.join(dirpath, fn)


def relpath_posix(path: str, start: str) -> str:
    return os.path.relpath(path, start=start).replace(os.sep, "/")


def depth_u16_to_m(depth_u16: np.ndarray, *, scale: float) -> np.ndarray:
    x = np.asarray(depth_u16)
    if x.dtype != np.uint16:
        x = x.astype(np.uint16, copy=False)
    return (x.astype(np.float32) / float(scale)).astype(np.float32, copy=False)


def finite_mask(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    return np.isfinite(d) & (d > 0)


def diode_valid_from_depth_mask(depth: np.ndarray, mask_npy: np.ndarray) -> np.ndarray:
    """DIODE ``*_depth_mask.npy``: ``(m != 0) & finite_mask(depth[..., 0])`` (eval_diode TFDS)."""
    a = np.asarray(depth)
    d0 = np.squeeze(a[..., 0] if a.ndim >= 3 else a)
    m = np.asarray(mask_npy).squeeze()
    if d0.shape != m.shape:
        raise ValueError(f"depth shape {d0.shape} != mask shape {m.shape}")
    return (m != 0) & finite_mask(d0)


def _to_2d_depth(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = np.mean(a, axis=-1)
    if a.ndim != 2:
        raise ValueError(f"Expected depth with 2D-compatible shape, got {a.shape}")
    return a.astype(np.float32, copy=False)


def _depth_range_valid(d: np.ndarray, lo: float, hi: float) -> np.ndarray:
    v = np.isfinite(d) & (d > lo)
    if hi is not None and np.isfinite(hi):
        v &= d < hi
    return v


_KITTI_ROI_FRAC = {
    "garg": ((0.40810811, 0.99189189), (0.03594771, 0.96405229)),
    "eigen": ((0.3324324, 0.91351351), (0.0359477, 0.96405229)),
}


def _kitti_eval_roi_mask(h: int, w: int, crop: str | None) -> np.ndarray:
    if crop is None:
        return np.ones((h, w), dtype=bool)
    if crop not in _KITTI_ROI_FRAC:
        raise ValueError(f"kitti_valid_mask_crop must be None, 'garg', or 'eigen', got {crop!r}")
    (y0, y1), (x0, x1) = _KITTI_ROI_FRAC[crop]
    out = np.zeros((h, w), dtype=bool)
    out[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
    return out


def _nyu_eigen_roi_mask(h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=bool)
    out[45:471, 41:601] = True
    return out


def get_valid_mask(
    depth: np.ndarray,
    dataset_name: str | None,
    *,
    eigen_valid_mask: bool | None = None,
    kitti_valid_mask_crop: str | None = None,
    external_mask: np.ndarray | None = None,
    d3_dataset_ranges: dict[str, tuple[float, float]] | None = None,
    d3_default_all_valid: bool = True,
) -> np.ndarray:
    """Valid (H,W) bool mask. For raw KITTI/NYU/ScanNet PNGs, run :func:`decode_benchmark_depth_to_metric` first. DIODE: pass ``external_mask`` (*_depth_mask.npy)."""
    from evals.configs.benchmark_dataset_configs import (
        DIODE_CFG,
        ETH3D_CFG,
        KITTI_CFG,
        NYU_CFG,
        SCANNET_CFG,
    )

    d = _to_2d_depth(depth)
    h, w = d.shape
    name = (dataset_name or "").strip().lower()
    name = {"nyu_v2": "nyu", "nyu2": "nyu", "d3_vkitti": "vkitti"}.get(name, name)

    d3_ranges = d3_dataset_ranges or {
        "vkitti": (1e-5, 80.0),
        "hypersim": (1e-5, 65.0),
        "coco": (1e-5, 80.0),
    }
    if name in d3_ranges:
        lo, hi = d3_ranges[name]
        return _depth_range_valid(d, lo, hi)

    if name == "diode":
        lo, hi = float(DIODE_CFG["min_depth"]), float(DIODE_CFG["max_depth"])
        base = _depth_range_valid(d, lo, hi)
        if external_mask is not None:
            base &= diode_valid_from_depth_mask(depth, external_mask)
        return base

    if name == "eth3d":
        return _depth_range_valid(d, float(ETH3D_CFG["min_depth"]), float("inf"))

    if name == "kitti":
        lo, hi = float(KITTI_CFG["min_depth"]), float(KITTI_CFG["max_depth"])
        crop = kitti_valid_mask_crop if kitti_valid_mask_crop is not None else KITTI_CFG.get("valid_mask_crop")
        return _depth_range_valid(d, lo, hi) & _kitti_eval_roi_mask(h, w, crop)

    if name == "nyu":
        lo, hi = float(NYU_CFG["min_depth"]), float(NYU_CFG["max_depth"])
        use_e = NYU_CFG["eigen_valid_mask"] if eigen_valid_mask is None else bool(eigen_valid_mask)
        v = _depth_range_valid(d, lo, hi)
        return v & _nyu_eigen_roi_mask(h, w) if use_e else v

    if name == "scannet":
        return _depth_range_valid(d, float(SCANNET_CFG["min_depth"]), float(SCANNET_CFG["max_depth"]))

    if d3_default_all_valid:
        return np.ones((h, w), dtype=bool)
    return np.isfinite(d) & (d > 0)


def valid_mask_for_dataset(
    depth: np.ndarray,
    dataset_type: str | None = None,
    dataset_ranges: dict[str, tuple[float, float]] | None = None,
    default_all_valid: bool = True,
) -> np.ndarray:
    """D3 TFDS: vkitti / hypersim / coco (override ranges via ``dataset_ranges``)."""
    return get_valid_mask(
        depth,
        dataset_type,
        d3_dataset_ranges=dataset_ranges,
        d3_default_all_valid=default_all_valid,
    )


def _squeeze_depth_hw3_equal(d: np.ndarray) -> np.ndarray:
    """Match ``BaseDataset._read_depth_file``: one channel if RGB planes are identical."""
    d = np.asarray(d, dtype=np.float32)
    if d.ndim == 3 and d.shape[2] == 3:
        if np.allclose(d[..., 0], d[..., 1]) and np.allclose(d[..., 0], d[..., 2]):
            return d[..., 0]
    return d


def decode_benchmark_depth_to_metric(
    depth: np.ndarray,
    dataset_name: str | None,
    *,
    is_exr: bool = False,
) -> np.ndarray:
    """
    Metric depth in meters, using the same rules as ``evals/configs/*._read_depth_file``:

    - **kitti**: PNG / uint-style → ``/ 256`` (unless ``is_exr``)
    - **nyu**, **scannet**: mm-style PNG → ``/ 1000`` (unless ``is_exr``)
    - **eth3d**: already float32 plane; ``±inf`` → ``0`` (matches ETH3D loader)
    - **diode** (npy), **D3** sources: no extra scale (already metric in pipeline)

    Call this on raw rasters before :func:`get_valid_mask` / alignment when needed.
    """
    name = (dataset_name or "").strip().lower()
    name = {"nyu_v2": "nyu", "nyu2": "nyu"}.get(name, name)
    d = _squeeze_depth_hw3_equal(depth)
    if is_exr:
        return np.asarray(d, dtype=np.float32)
    if name == "kitti":
        return np.asarray(d, dtype=np.float32) / 256.0
    if name in ("nyu", "scannet"):
        return np.asarray(d, dtype=np.float32) / 1000.0
    if name == "eth3d":
        d = np.asarray(d, dtype=np.float32)
        d[np.isinf(d)] = 0.0
        return d
    return np.asarray(d, dtype=np.float32)


def decode_eth3d_depth_binary(
    data: bytes,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    """
    ETH3D float32 blob → ``(H, W)``, ``inf`` → ``0`` (see ``ETH3DDataset._read_depth_file``).
    Default ``H, W`` from ``benchmark_dataset_configs.ETH3D_CFG`` if omitted.
    """
    from evals.configs.benchmark_dataset_configs import ETH3D_CFG

    h = int(ETH3D_CFG["depth_binary_height"]) if height is None else height
    w = int(ETH3D_CFG["depth_binary_width"]) if width is None else width
    depth = np.frombuffer(data, dtype=np.float32).copy()
    depth[np.isinf(depth)] = 0.0
    return depth.reshape((h, w))


def decode_tf_text(x) -> str:
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def depth_to_vis_2d(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.ndim == 3:
            vm = vm[..., 0]
        v = vm.astype(bool)
    else:
        v = np.isfinite(d) & (d > 0)
    if not np.any(v):
        return np.zeros_like(d, dtype=np.float32)
    vals = d[v]
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    vmax = vmax if vmax > vmin else vmin + 1e-6
    out = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    out = np.where(v, out, 0.0).astype(np.float32)
    return np.repeat(out[..., np.newaxis], 3, axis=-1)


def _mask_2d(vm: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    a = np.asarray(vm)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != ref_shape:
        raise ValueError(f"Mask shape mismatch: mask {a.shape} vs depth {ref_shape}")
    return a.astype(bool)


def _valid_pixels(pred2d, gt2d, valid_mask, require_positive_gt: bool) -> np.ndarray:
    if valid_mask is None:
        v = np.isfinite(pred2d) & np.isfinite(gt2d)
    else:
        v = _mask_2d(valid_mask, pred2d.shape) & np.isfinite(pred2d) & np.isfinite(gt2d)
    if require_positive_gt:
        v &= gt2d > 0
    return v


def align_least_squares(
    depth_pred,
    depth_gt,
    valid_mask=None,
    *,
    require_positive_gt: bool = True,
):
    """LS fit pred_aligned = scale * pred + shift; optional ``valid_mask`` from ``get_valid_mask``."""
    pred2d = _to_2d_depth(depth_pred)
    gt2d = _to_2d_depth(depth_gt)
    if pred2d.shape != gt2d.shape:
        raise ValueError(f"Shape mismatch: pred {pred2d.shape} vs gt {gt2d.shape}")
    v = _valid_pixels(pred2d, gt2d, valid_mask, require_positive_gt)
    if not np.any(v):
        return pred2d.copy()
    x = pred2d[v].reshape(-1, 1)
    y = gt2d[v].reshape(-1, 1)
    A = np.hstack([x, np.ones_like(x)])
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0].ravel()
    return (scale * pred2d + shift).astype(np.float32, copy=False)


def decode_normals_png(normals_img: np.ndarray) -> np.ndarray:
    return ((normals_img / 255.0) * 2.0 - 1.0).astype(np.float32)


def get_accuracy(
    depth_pred, depth_gt, valid_mask=None, *, require_positive_gt: bool = True
) -> tuple[float, float]:
    pred2d = _to_2d_depth(depth_pred)
    gt2d = _to_2d_depth(depth_gt)
    if pred2d.shape != gt2d.shape:
        raise ValueError(f"Shape mismatch: pred {pred2d.shape} vs gt {gt2d.shape}")
    v = _valid_pixels(pred2d, gt2d, valid_mask, require_positive_gt)
    if not np.any(v):
        return float("nan"), float("nan")
    predv = np.clip(pred2d[v], 1e-6, None)
    gtv = np.clip(gt2d[v], 1e-6, None)
    absrel = float(np.mean(np.abs(predv - gtv) / gtv))
    ratio = np.maximum(predv / gtv, gtv / predv)
    return absrel, float(np.mean(ratio < 1.25))
