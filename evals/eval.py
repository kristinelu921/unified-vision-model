"""
Run AbsRel + delta1 (sigma) using ``evals/eval_utils`` on each GenPercept TFDS benchmark.

Depth in these TFDS datasets is already metric (meters); masks follow each builder.

Default ``--identity`` sets prediction = ground truth (sanity: AbsRel ~ 0, delta1 ~ 1).

Usage (from repo root, TFDS already built under ``DATA_DIR``)::

  python -m evals.eval --data-dir /path/to/tfds_output --datasets all --num-batches 2

"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import tensorflow_datasets as tfds

# Register custom TFDS builders so ``tfds.load("eval_*", ...)`` resolves.
from evals.tfds import eval_diode  # noqa: F401
from evals.tfds import eval_eth3d  # noqa: F401
from evals.tfds import eval_kitti  # noqa: F401
from evals.tfds import eval_nyu  # noqa: F401
from evals.tfds import eval_scannet  # noqa: F401

from evals.eval_utils import align_least_squares, get_accuracy, get_valid_mask

# tfds builder name -> get_valid_mask dataset_name + default split
_DATASETS: dict[str, dict[str, Any]] = {
    "kitti": {"tfds": "eval_kitti", "split": "test", "mask_name": "kitti"},
    "nyu": {"tfds": "eval_nyu", "split": "test", "mask_name": "nyu"},
    "diode": {"tfds": "eval_diode", "split": "test", "mask_name": "diode"},
    "eth3d": {"tfds": "eval_eth3_d", "split": "test", "mask_name": "eth3d"},
    "scannet": {"tfds": "eval_scan_net", "split": "test", "mask_name": "scannet"},
}


def _combine_mask(
    gt: np.ndarray,
    mask_name: str,
    vm_tfds_hw: np.ndarray,
    *,
    kitti_crop: str | None,
    nyu_eigen: bool,
) -> np.ndarray:
    """``(H,W)`` bool: ``eval_utils.get_valid_mask`` ∩ TFDS ``valid_mask``."""
    if mask_name == "diode":
        vm = get_valid_mask(gt, "diode", external_mask=vm_tfds_hw)
        return vm
    kw: dict = {}
    if mask_name == "kitti" and kitti_crop is not None:
        kw["kitti_valid_mask_crop"] = kitti_crop
    if mask_name == "nyu":
        kw["eigen_valid_mask"] = nyu_eigen
    vm = get_valid_mask(gt, mask_name, **kw)
    return vm & vm_tfds_hw.astype(bool)


def _run_one_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    vm: np.ndarray,
) -> tuple[float, float]:
    pred_a = align_least_squares(pred, gt, valid_mask=vm)
    return get_accuracy(pred_a, gt, valid_mask=vm)


def run_benchmark(
    key: str,
    data_dir: str,
    *,
    split: str | None,
    num_batches: int,
    batch_size: int,
    identity: bool,
    kitti_crop: str | None,
    nyu_eigen: bool,
) -> tuple[list[float], list[float]]:
    cfg = _DATASETS[key]
    sp = split or cfg["split"]
    mask_name = cfg["mask_name"]

    load_kw: dict = {"split": sp, "data_dir": data_dir}
    ds = tfds.load(cfg["tfds"], **load_kw)
    ds = ds.batch(batch_size).take(num_batches)

    absrels: list[float] = []
    deltas: list[float] = []

    for batch in ds:
        depth = batch["depth"].numpy()
        vm_tfds = batch["valid_mask"].numpy()
        b = depth.shape[0]
        for i in range(b):
            gt = depth[i]
            pred = np.copy(gt)
            if not identity:
                pass  # future: pred = model prediction for this sample
            vm_hw = vm_tfds[i, ..., 0]
            vm = _combine_mask(
                gt,
                mask_name,
                vm_hw,
                kitti_crop=kitti_crop,
                nyu_eigen=nyu_eigen,
            )
            if not np.any(vm):
                continue
            ar, d1 = _run_one_sample(pred, gt, vm)
            if np.isfinite(ar):
                absrels.append(ar)
            if np.isfinite(d1):
                deltas.append(d1)

    return absrels, deltas


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TFDS benchmark metrics via eval_utils")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory passed to tfds.load(..., data_dir=) for built datasets",
    )
    p.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated: kitti,nyu,diode,eth3d,scannet (default: all)",
    )
    p.add_argument("--split", default=None, help="Override split (default per dataset)")
    p.add_argument("--num-batches", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--no-identity",
        action="store_false",
        dest="identity",
        default=True,
        help="Reserved for real predictions; currently still uses GT as pred",
    )
    p.add_argument(
        "--kitti-crop",
        default=None,
        choices=["garg", "eigen"],
        help="KITTI eval ROI inside get_valid_mask (optional)",
    )
    p.add_argument(
        "--nyu-no-eigen",
        action="store_true",
        help="Disable Eigen crop for NYU (default: Eigen crop on)",
    )
    args = p.parse_args(argv)

    keys = list(_DATASETS.keys()) if args.datasets.strip().lower() == "all" else [
        k.strip().lower() for k in args.datasets.split(",") if k.strip()
    ]
    for k in keys:
        if k not in _DATASETS:
            print(f"Unknown dataset: {k!r}. Choose from {list(_DATASETS)}", file=sys.stderr)
            return 1

    nyu_eigen = not args.nyu_no_eigen

    for key in keys:
        print(f"--- {key} ({_DATASETS[key]['tfds']}) ---")
        try:
            absrels, deltas = run_benchmark(
                key,
                args.data_dir,
                split=args.split,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                identity=args.identity,
                kitti_crop=args.kitti_crop,
                nyu_eigen=nyu_eigen,
            )
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            continue
        if not absrels:
            print("  no valid samples (check data_dir / split / env roots for builders)")
            continue
        print(
            f"  samples: {len(absrels)}  "
            f"mean AbsRel: {float(np.mean(absrels)):.6f}  "
            f"mean delta1: {float(np.mean(deltas)):.6f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
