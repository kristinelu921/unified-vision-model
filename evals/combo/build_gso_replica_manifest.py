#!/usr/bin/env python3
"""Build a normalized combo manifest from Omnidata Replica-GSO debug data."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


STEM_RE = re.compile(r"^(point_\d+_view_\d+)_domain_semantic\.png$")


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def decode_semantic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sem = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint16)
    class_map = sem[..., 0] * 256 + sem[..., 1]
    instance_map = sem[..., 2]
    return class_map, instance_map


def collect_instances(root: Path, scene: str, min_area: int) -> dict[int, list[dict[str, Any]]]:
    sem_dir = root / "semantic" / "replica_gso" / scene
    rgb_dir = root / "rgb" / "replica_gso" / scene
    normal_dir = root / "normal" / "replica_gso" / scene
    depth_dir = root / "depth_euclidean" / "replica_gso" / scene

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sem_path in sorted(sem_dir.glob("*_domain_semantic.png")):
        match = STEM_RE.match(sem_path.name)
        if not match:
            continue
        stem = match.group(1)
        rgb_path = rgb_dir / f"{stem}_domain_rgb.png"
        normal_path = normal_dir / f"{stem}_domain_normal.png"
        depth_path = depth_dir / f"{stem}_domain_depth_euclidean.png"
        if not (rgb_path.exists() and normal_path.exists() and depth_path.exists()):
            continue

        class_map, instance_map = decode_semantic(sem_path)
        for class_id in np.unique(class_map):
            class_id = int(class_id)
            if class_id < 102:
                continue
            class_mask = class_map == class_id
            for instance_id in np.unique(instance_map[class_mask]):
                instance_id = int(instance_id)
                mask = class_mask & (instance_map == instance_id)
                area = int(mask.sum())
                if area < min_area:
                    continue
                by_class[class_id].append(
                    {
                        "stem": stem,
                        "semantic": str(sem_path),
                        "image": str(rgb_path),
                        "normal": str(normal_path),
                        "depth": str(depth_path),
                        "class_id": class_id,
                        "instance_id": instance_id,
                        "area": area,
                        "bbox": bbox_from_mask(mask),
                    }
                )
    return by_class


def collect_frames(root: Path, scene: str, min_area: int) -> list[dict[str, Any]]:
    sem_dir = root / "semantic" / "replica_gso" / scene
    rgb_dir = root / "rgb" / "replica_gso" / scene
    normal_dir = root / "normal" / "replica_gso" / scene
    depth_dir = root / "depth_euclidean" / "replica_gso" / scene

    frames = []
    for sem_path in sorted(sem_dir.glob("*_domain_semantic.png")):
        match = STEM_RE.match(sem_path.name)
        if not match:
            continue
        stem = match.group(1)
        rgb_path = rgb_dir / f"{stem}_domain_rgb.png"
        normal_path = normal_dir / f"{stem}_domain_normal.png"
        depth_path = depth_dir / f"{stem}_domain_depth_euclidean.png"
        if not (rgb_path.exists() and normal_path.exists() and depth_path.exists()):
            continue

        class_map, instance_map = decode_semantic(sem_path)
        instances = []
        for class_id in np.unique(class_map):
            class_id = int(class_id)
            if class_id < 102:
                continue
            class_mask = class_map == class_id
            for instance_id in np.unique(instance_map[class_mask]):
                instance_id = int(instance_id)
                mask = class_mask & (instance_map == instance_id)
                area = int(mask.sum())
                if area < min_area:
                    continue
                instances.append(
                    {
                        "stem": stem,
                        "semantic": str(sem_path),
                        "image": str(rgb_path),
                        "normal": str(normal_path),
                        "depth": str(depth_path),
                        "class_id": class_id,
                        "instance_id": instance_id,
                        "area": area,
                        "bbox": bbox_from_mask(mask),
                    }
                )
        category_count = len({item["class_id"] for item in instances})
        if len(instances) >= 2 and category_count >= 2:
            frames.append(
                {
                    "stem": stem,
                    "image": str(rgb_path),
                    "normal": str(normal_path),
                    "depth": str(depth_path),
                    "instances": sorted(instances, key=lambda item: -item["area"]),
                    "category_count": category_count,
                    "total_area": sum(item["area"] for item in instances),
                }
            )
    return frames


def write_mask(item: dict[str, Any], mask_dir: Path) -> Path:
    class_id = item["class_id"]
    instance_id = item["instance_id"]
    mask_path = mask_dir / f"{item['stem']}_class_{class_id}_inst_{instance_id}_mask.png"
    if mask_path.exists():
        return mask_path

    class_map, instance_map = decode_semantic(Path(item["semantic"]))
    mask = ((class_map == class_id) & (instance_map == instance_id)).astype(np.uint8) * 255
    Image.fromarray(mask).save(mask_path)
    return mask_path


def build_single_object_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out
    mask_dir = out_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    by_class = collect_instances(args.root, args.scene, args.min_area)
    classes = sorted(by_class, key=lambda class_id: len(by_class[class_id]), reverse=True)[: args.classes]
    records = []
    summary = []
    for class_id in classes:
        items = sorted(
            by_class[class_id],
            key=lambda item: (-item["area"], item["stem"], item["instance_id"]),
        )[: args.per_class]
        summary.append(
            {
                "class_id": class_id,
                "count_available": len(by_class[class_id]),
                "count_selected": len(items),
            }
        )
        for item in items:
            mask_path = write_mask(item, mask_dir)
            stem = item["stem"]
            instance_id = item["instance_id"]
            records.append(
                {
                    "id": f"gso_replica_{args.scene}_{stem}_class_{class_id}_inst_{instance_id}",
                    "dataset": "omnidata_replica_gso_debug",
                    "scene": args.scene,
                    "camera": stem.split("_view_")[1],
                    "frame": stem.split("_view_")[0].replace("point_", ""),
                    "image": item["image"],
                    "depth": item["depth"],
                    "normal": item["normal"],
                    "instances": [
                        {
                            "id": f"class_{class_id}_inst_{instance_id}",
                            "category": f"gso_{class_id}",
                            "mask": str(mask_path),
                            "bbox": item["bbox"],
                            "area": item["area"],
                        }
                    ],
                }
            )

    return {
        "dataset": "omnidata_replica_gso_debug",
        "scene": args.scene,
        "semantic_encoding": (
            "class_id = 256 * red + green; instance_id = blue; "
            "GSO objects filtered as class_id >= 102"
        ),
        "records": records,
        "class_summary": summary,
    }


def build_multi_instance_manifest(args: argparse.Namespace) -> dict[str, Any]:
    out_root = args.out
    mask_dir = out_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(args.root, args.scene, args.min_area)
    frames = sorted(frames, key=lambda frame: (-frame["category_count"], -frame["total_area"], frame["stem"]))
    records = []
    class_counts: dict[int, int] = defaultdict(int)
    for frame in frames[: args.records]:
        instances = []
        for item in frame["instances"][: args.max_instances]:
            class_id = item["class_id"]
            instance_id = item["instance_id"]
            class_counts[class_id] += 1
            instances.append(
                {
                    "id": f"class_{class_id}_inst_{instance_id}",
                    "category": f"gso_{class_id}",
                    "mask": str(write_mask(item, mask_dir)),
                    "bbox": item["bbox"],
                    "area": item["area"],
                }
            )
        stem = frame["stem"]
        records.append(
            {
                "id": f"gso_replica_{args.scene}_{stem}",
                "dataset": "omnidata_replica_gso_debug",
                "scene": args.scene,
                "camera": stem.split("_view_")[1],
                "frame": stem.split("_view_")[0].replace("point_", ""),
                "image": frame["image"],
                "depth": frame["depth"],
                "normal": frame["normal"],
                "instances": instances,
            }
        )

    return {
        "dataset": "omnidata_replica_gso_debug",
        "scene": args.scene,
        "semantic_encoding": (
            "class_id = 256 * red + green; instance_id = blue; "
            "GSO objects filtered as class_id >= 102"
        ),
        "records": records,
        "frame_count_available": len(frames),
        "class_counts_selected": [
            {"class_id": class_id, "count": count}
            for class_id, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/klum/omnidata_gso_replica_debug"))
    parser.add_argument("--scene", default="frl_apartment_0-3")
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/omnidata_gso_replica_debug_single_object"))
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--records", type=int, default=60)
    parser.add_argument("--max-instances", type=int, default=8)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--manifest-name", default="manifest_top6_10each.json")
    parser.add_argument("--mode", choices=["single_object", "multi_instance"], default="single_object")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "multi_instance":
        manifest = build_multi_instance_manifest(args)
    else:
        manifest = build_single_object_manifest(args)
    manifest_path = args.out / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    details_key = "class_summary" if args.mode == "single_object" else "class_counts_selected"
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "record_count": len(manifest["records"]),
                "details": manifest[details_key],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
