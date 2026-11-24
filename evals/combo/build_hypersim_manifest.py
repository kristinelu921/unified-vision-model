#!/usr/bin/env python3
"""Build a combo-eval manifest from a small Hypersim scene subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SEMANTIC_ID_TO_NAME = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floormat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "showercurtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

DEFAULT_EXCLUDE_CATEGORIES = {
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "otherstructure",
}


def read_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["dataset"][()]


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def majority_semantic_name(semantic: np.ndarray, mask: np.ndarray) -> tuple[int, str]:
    values, counts = np.unique(semantic[mask], return_counts=True)
    valid = values >= 0
    if not np.any(valid):
        return -1, "unknown"
    values = values[valid]
    counts = counts[valid]
    semantic_id = int(values[int(np.argmax(counts))])
    return semantic_id, SEMANTIC_ID_TO_NAME.get(semantic_id, f"semantic_{semantic_id}")


def build_manifest(
    subset_dir: Path,
    out_dir: Path,
    scene: str,
    camera: str,
    min_area: int,
    max_frames: int,
    primary_category: str | None,
    secondary_category: str | None,
) -> dict:
    scene_dir = subset_dir / scene
    image_dir = scene_dir / "images" / f"scene_{camera}_final_preview"
    geometry_dir = scene_dir / "images" / f"scene_{camera}_geometry_hdf5"
    mask_dir = out_dir / "masks" / scene / camera
    mask_dir.mkdir(parents=True, exist_ok=True)

    records = []
    image_paths = sorted(image_dir.glob("frame.*.color.jpg"))
    if max_frames > 0:
        image_paths = image_paths[:max_frames]

    for image_path in image_paths:
        frame = image_path.name.split(".")[1]
        semantic_path = geometry_dir / f"frame.{frame}.semantic.hdf5"
        instance_path = geometry_dir / f"frame.{frame}.semantic_instance.hdf5"
        depth_path = geometry_dir / f"frame.{frame}.depth_meters.hdf5"
        normal_path = geometry_dir / f"frame.{frame}.normal_cam.hdf5"
        if not all(p.is_file() for p in [semantic_path, instance_path, depth_path, normal_path]):
            continue

        semantic = read_hdf5(semantic_path)
        semantic_instance = read_hdf5(instance_path)
        instances = []
        for instance_id in sorted(int(x) for x in np.unique(semantic_instance) if x >= 0):
            mask = semantic_instance == instance_id
            area = int(mask.sum())
            if area < min_area:
                continue
            semantic_id, category = majority_semantic_name(semantic, mask)
            if category in DEFAULT_EXCLUDE_CATEGORIES:
                continue
            mask_path = mask_dir / f"frame.{frame}.instance_{instance_id:04d}.png"
            Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
            instances.append(
                {
                    "id": f"{scene}_{camera}_{frame}_instance_{instance_id}",
                    "instance_id": instance_id,
                    "semantic_id": semantic_id,
                    "category": category,
                    "area": area,
                    "mask": str(mask_path),
                    "bbox": bbox_from_mask(mask),
                }
            )

        if primary_category or secondary_category:
            required = {x for x in [primary_category, secondary_category] if x}
            available = {item["category"] for item in instances}
            if not required.issubset(available):
                continue

            def sort_key(item: dict) -> tuple[int, int]:
                if item["category"] == primary_category:
                    return (0, -item["area"])
                if item["category"] == secondary_category:
                    return (1, -item["area"])
                return (2, -item["area"])

            instances.sort(key=sort_key)
        else:
            instances.sort(key=lambda item: item["area"], reverse=True)
        if len({item["category"] for item in instances}) < 2:
            continue
        records.append(
            {
                "id": f"{scene}_{camera}_{frame}",
                "scene": scene,
                "camera": camera,
                "frame": int(frame),
                "image": str(image_path),
                "depth": str(depth_path),
                "depth_key": "dataset",
                "normal": str(normal_path),
                "normal_key": "dataset",
                "semantic": str(semantic_path),
                "semantic_instance": str(instance_path),
                "instances": instances,
            }
        )

    return {
        "source": "Hypersim",
        "subset_dir": str(subset_dir),
        "scene": scene,
        "camera": camera,
        "min_area": min_area,
        "primary_category": primary_category,
        "secondary_category": secondary_category,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=Path("/mnt/klum/hypersim_subset"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/klum/combo_hypersim"))
    parser.add_argument("--scene", default="ai_001_001")
    parser.add_argument("--camera", default="cam_00")
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--primary-category")
    parser.add_argument("--secondary-category")
    parser.add_argument("--manifest-name", default="combo_manifest.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        subset_dir=args.subset_dir,
        out_dir=args.out_dir,
        scene=args.scene,
        camera=args.camera,
        min_area=args.min_area,
        max_frames=args.max_frames,
        primary_category=args.primary_category,
        secondary_category=args.secondary_category,
    )
    manifest_path = args.out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['records'])} records to {manifest_path}")


if __name__ == "__main__":
    main()
