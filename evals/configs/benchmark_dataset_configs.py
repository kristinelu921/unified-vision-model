"""
Dictionary configs for GenPercept-style benchmark datasets (DIODE, ETH3D, KITTI, NYU, ScanNet).

Each entry holds BaseDataset kwargs plus dataset-specific fields used by the corresponding
subclass in the same folder. Override any key by passing kwargs to the Dataset constructor.

Depth scaling / ETH3D binary decode for numpy eval mirrors ``evals/configs/*._read_depth_file``;
see ``evals.eval_utils.decode_benchmark_depth_to_metric`` and ``decode_eth3d_depth_binary``.
"""

from __future__ import annotations

from typing import Any

# --- DIODE --------------------------------------------------------------------

DIODE_CFG: dict[str, Any] = {
    "min_depth": 0.6,
    "max_depth": 350,
    "has_filled_depth": False,
    "name_mode": "id",
}

# --- ETH3D --------------------------------------------------------------------

ETH3D_CFG: dict[str, Any] = {
    "min_depth": 1e-5,
    "max_depth": float("inf"),
    "has_filled_depth": False,
    "name_mode": "id",
    "depth_binary_height": 4032,
    "depth_binary_width": 6048,
}

# --- KITTI --------------------------------------------------------------------

KITTI_CFG: dict[str, Any] = {
    "min_depth": 1e-5,
    "max_depth": 80,
    "has_filled_depth": False,
    "name_mode": "id",
    "kitti_bm_crop": True,
    "valid_mask_crop": None,
}

# --- NYU ----------------------------------------------------------------------

NYU_CFG: dict[str, Any] = {
    "min_depth": 1e-3,
    "max_depth": 10.0,
    "has_filled_depth": True,
    "name_mode": "rgb_id",
    "eigen_valid_mask": True,
}

# --- ScanNet ------------------------------------------------------------------

SCANNET_CFG: dict[str, Any] = {
    "min_depth": 1e-3,
    "max_depth": 10,
    "has_filled_depth": False,
    "name_mode": "id",
}

BENCHMARK_DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "diode": DIODE_CFG,
    "eth3d": ETH3D_CFG,
    "kitti": KITTI_CFG,
    "nyu": NYU_CFG,
    "scannet": SCANNET_CFG,
}


def resolve_name_mode(key: str):
    from .base_dataset import PerceptionFileNameMode

    mapping = {
        "id": PerceptionFileNameMode.id,
        "rgb_id": PerceptionFileNameMode.rgb_id,
        "i_d_rgb": PerceptionFileNameMode.i_d_rgb,
        "rgb_i_d": PerceptionFileNameMode.rgb_i_d,
    }
    if key not in mapping:
        raise ValueError(f"Unknown name_mode: {key!r}")
    return mapping[key]
