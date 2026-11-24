#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_BUCKET_ROOT = "gs://kmh-gcp-asia-northeast1-b/kristine/lvm"
DEFAULT_OUT_DIR = "/mnt/klum/tfds_first5_ground_truth"
LVIS_PERSON_SPLIT = "train_cat_793"


def _require_tfds():
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise SystemExit(
            "Missing tensorflow_datasets. Install TensorFlow + TFDS in this env first."
        ) from exc
    return tfds


def _builder_dir(bucket_root: str, name: str) -> str:
    return f"{bucket_root.rstrip('/')}/{name}/{name.removesuffix('_tfds')}/1.0.0"


def _lvis_builder_dir(bucket_root: str) -> str:
    return (
        f"{bucket_root.rstrip('/')}/"
        "lvis_instance_category_tfds/lvis_instance_category_sharded/2.0.0"
    )


def _as_uint8_image(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _decode_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode()
        return item
    if arr.size == 1:
        item = arr.reshape(-1)[0].item()
        if isinstance(item, bytes):
            return item.decode()
        return item
    return arr.tolist()


def _metadata(example: dict[str, Any]) -> dict[str, Any]:
    raw = example.get("metadata", {})
    if not isinstance(raw, dict):
        return {}
    return {key: _decode_scalar(value) for key, value in raw.items()}


def _save_png(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_uint8_image(value)).save(path)
    return str(path)


def _save_depth_metric(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    np.save(path, arr)
    return str(path)


def _lvis_mask(example: dict[str, Any]) -> np.ndarray:
    masks = np.asarray(example["annotations"]["mask_raster"])
    if masks.ndim == 3:
        masks = masks[..., None]
    mask = np.any(masks > 0, axis=(0, -1)).astype(np.uint8) * 255
    return np.repeat(mask[..., None], 3, axis=-1)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2) + "\n")


def _iter_first(builder_path: str, split: str, count: int):
    tfds = _require_tfds()
    builder = tfds.builder_from_directory(builder_path)
    total = builder.info.splits[split].num_examples
    n = min(count, total)
    for idx in range(n):
        ds = builder.as_dataset(split=f"{split}[{idx}:{idx + 1}]", shuffle_files=False)
        yield idx, next(iter(tfds.as_numpy(ds)))


def export_depth(bucket_root: str, out_dir: Path, count: int) -> list[dict[str, Any]]:
    builder_path = _builder_dir(bucket_root, "d3_depth_300k_tfds")
    rows: list[dict[str, Any]] = []
    for idx, example in _iter_first(builder_path, "train", count):
        sample_dir = out_dir / "depth_300k" / f"{idx:05d}"
        row = {
            "dataset": "d3_depth_300k",
            "split": "train",
            "index": idx,
            "builder_path": builder_path,
            "image": _save_png(sample_dir / "image.png", example["image"]),
            "depth_ground_truth": _save_png(sample_dir / "depth_ground_truth.png", example["mask"]),
            "depth_metric_npy": _save_depth_metric(
                sample_dir / "depth_metric.npy", example["depth_metric"]
            ),
            "metadata": _metadata(example),
        }
        rows.append(row)
        print(f"depth_300k[{idx}] -> {sample_dir}", flush=True)
    return rows


def export_normal(bucket_root: str, out_dir: Path, count: int) -> list[dict[str, Any]]:
    builder_path = _builder_dir(bucket_root, "d3_normal_300k_tfds")
    rows: list[dict[str, Any]] = []
    for idx, example in _iter_first(builder_path, "train", count):
        sample_dir = out_dir / "normal_300k" / f"{idx:05d}"
        row = {
            "dataset": "d3_normal_300k",
            "split": "train",
            "index": idx,
            "builder_path": builder_path,
            "image": _save_png(sample_dir / "image.png", example["image"]),
            "surface_normal_ground_truth": _save_png(
                sample_dir / "surface_normal_ground_truth.png", example["mask"]
            ),
            "metadata": _metadata(example),
        }
        rows.append(row)
        print(f"normal_300k[{idx}] -> {sample_dir}", flush=True)
    return rows


def export_person(bucket_root: str, out_dir: Path, count: int) -> list[dict[str, Any]]:
    builder_path = _lvis_builder_dir(bucket_root)
    rows: list[dict[str, Any]] = []
    for idx, example in _iter_first(builder_path, LVIS_PERSON_SPLIT, count):
        sample_dir = out_dir / "lvis_person_segmentation" / f"{idx:05d}"
        mask = _lvis_mask(example)
        row = {
            "dataset": "lvis_instance_category_sharded",
            "category": "person",
            "split": LVIS_PERSON_SPLIT,
            "index": idx,
            "builder_path": builder_path,
            "image": _save_png(sample_dir / "image.png", example["image"]),
            "segmentation_ground_truth": _save_png(
                sample_dir / "segmentation_ground_truth.png", mask
            ),
            "metadata": _metadata(example),
        }
        rows.append(row)
        print(f"lvis person[{idx}] -> {sample_dir}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the first N RGB/ground-truth pairs from depth_300k, "
            "normal_300k, and LVIS person segmentation TFDS builders."
        )
    )
    parser.add_argument("--bucket-root", default=DEFAULT_BUCKET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(export_depth(args.bucket_root, args.out_dir, args.count))
    rows.extend(export_normal(args.bucket_root, args.out_dir, args.count))
    rows.extend(export_person(args.bucket_root, args.out_dir, args.count))

    manifest = args.out_dir / "manifest.json"
    _write_manifest(manifest, rows)
    print(f"Wrote {len(rows)} samples and manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
