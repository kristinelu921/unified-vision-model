#!/usr/bin/env python3
"""Scan downloaded Hypersim subsets for recurring category pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from build_hypersim_manifest import DEFAULT_EXCLUDE_CATEGORIES, SEMANTIC_ID_TO_NAME


def read_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["dataset"][()]


def frame_categories(semantic_path: Path, instance_path: Path, min_area: int) -> set[str]:
    semantic = read_hdf5(semantic_path)
    semantic_instance = read_hdf5(instance_path)
    categories = set()
    for instance_id in sorted(int(x) for x in np.unique(semantic_instance) if x >= 0):
        mask = semantic_instance == instance_id
        area = int(mask.sum())
        if area < min_area:
            continue
        values, counts = np.unique(semantic[mask], return_counts=True)
        valid = values >= 0
        if not np.any(valid):
            continue
        values = values[valid]
        counts = counts[valid]
        semantic_id = int(values[int(np.argmax(counts))])
        category = SEMANTIC_ID_TO_NAME.get(semantic_id, f"semantic_{semantic_id}")
        if category in DEFAULT_EXCLUDE_CATEGORIES:
            continue
        categories.add(category)
    return categories


def scan_roots(roots: list[Path], min_area: int) -> dict:
    pair_records: Counter[tuple[str, str]] = Counter()
    pair_scenes: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_scene_cameras: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    category_records: Counter[str] = Counter()
    scene_count = set()
    frame_count = 0

    seen_frames = set()
    for root in roots:
        for semantic_path in sorted(root.glob("*/images/scene_*_geometry_hdf5/frame.*.semantic.hdf5")):
            scene = semantic_path.parts[-4]
            geometry_dir = semantic_path.parent.name
            camera = geometry_dir.removeprefix("scene_").removesuffix("_geometry_hdf5")
            frame = semantic_path.name.split(".")[1]
            key = (scene, camera, frame)
            if key in seen_frames:
                continue
            seen_frames.add(key)
            instance_path = semantic_path.with_name(f"frame.{frame}.semantic_instance.hdf5")
            depth_path = semantic_path.with_name(f"frame.{frame}.depth_meters.hdf5")
            normal_path = semantic_path.with_name(f"frame.{frame}.normal_cam.hdf5")
            image_path = semantic_path.parents[1] / f"scene_{camera}_final_preview" / f"frame.{frame}.color.jpg"
            if not all(path.is_file() for path in [instance_path, depth_path, normal_path, image_path]):
                continue
            categories = frame_categories(semantic_path, instance_path, min_area)
            if len(categories) < 2:
                continue
            frame_count += 1
            scene_count.add(scene)
            for category in categories:
                category_records[category] += 1
            for pair in combinations(sorted(categories), 2):
                pair_records[pair] += 1
                pair_scenes[pair].add(scene)
                pair_scene_cameras[pair].add((scene, camera))

    ranked = sorted(
        pair_records,
        key=lambda pair: (len(pair_scenes[pair]), len(pair_scene_cameras[pair]), pair_records[pair], pair),
        reverse=True,
    )
    return {
        "roots": [str(root) for root in roots],
        "min_area": min_area,
        "scene_count": len(scene_count),
        "frame_count": frame_count,
        "top_categories": category_records.most_common(40),
        "top_pairs": [
            {
                "pair": list(pair),
                "records": pair_records[pair],
                "scenes": len(pair_scenes[pair]),
                "scene_cameras": len(pair_scene_cameras[pair]),
                "scene_names": sorted(pair_scenes[pair])[:20],
            }
            for pair in ranked[:100]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=sorted(Path("/mnt/klum").glob("hypersim_subset*")),
    )
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/hypersim_duo_scan.json"))
    args = parser.parse_args()

    result = scan_roots(args.roots, args.min_area)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["scene_count", "frame_count", "top_pairs"]}, indent=2)[:12000])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
