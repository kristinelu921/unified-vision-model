#!/usr/bin/env python3
"""Find recurring object duos in Omnidata Replica-GSO semantic annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from build_gso_replica_manifest import collect_frames, write_mask


METADATA_DIR = Path(
    "/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310/lib/python3.10/"
    "site-packages/omnidata_tools/dataset/starter_dataset/component_datasets/replica_gso"
)


def load_gso_names(metadata_dir: Path) -> tuple[list[str], list[str]]:
    return (
        json.loads((metadata_dir / "classes.json").read_text(encoding="utf-8")),
        json.loads((metadata_dir / "list_of_objects.json").read_text(encoding="utf-8")),
    )


def describe_class(class_id: int, classes: list[str], objects: list[str]) -> dict[str, Any]:
    index = class_id - 102
    if index < 0 or index >= len(objects):
        return {"class_id": class_id, "object_category": "", "object_name": ""}
    return {"class_id": class_id, "object_category": classes[index], "object_name": objects[index]}


def class_instance_by_id(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
    best = {}
    for item in frame["instances"]:
        class_id = item["class_id"]
        if class_id not in best or item["area"] > best[class_id]["area"]:
            best[class_id] = item
    return best


def build_duo_manifest(args: argparse.Namespace) -> dict[str, Any]:
    classes, objects = load_gso_names(args.metadata_dir)
    frames = collect_frames(args.root, args.scene, args.min_area)

    pair_counts: Counter[tuple[int, int]] = Counter()
    frames_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        class_ids = sorted({item["class_id"] for item in frame["instances"]})
        for pair in combinations(class_ids, 2):
            pair_counts[pair] += 1
            frames_by_pair[pair].append(frame)

    selected_pairs = [pair for pair, count in pair_counts.most_common(args.duos) if count >= args.min_records]
    mask_dir = args.out / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    duo_summaries = []
    all_records = []

    for pair in selected_pairs:
        pair_frames = sorted(
            frames_by_pair[pair],
            key=lambda frame: (
                -class_instance_by_id(frame)[pair[0]]["area"] - class_instance_by_id(frame)[pair[1]]["area"],
                frame["stem"],
            ),
        )[: args.records_per_duo]
        duo_records = []
        for frame in pair_frames:
            by_id = class_instance_by_id(frame)
            instances = []
            for class_id in pair:
                item = by_id[class_id]
                instance_id = item["instance_id"]
                instances.append(
                    {
                        "id": f"class_{class_id}_inst_{instance_id}",
                        "category": f"gso_{class_id}",
                        "mask": str(write_mask(item, mask_dir)),
                        "bbox": item["bbox"],
                        "area": item["area"],
                        **describe_class(class_id, classes, objects),
                    }
                )
            stem = frame["stem"]
            record = {
                "id": f"gso_replica_{args.scene}_{stem}_duo_{pair[0]}_{pair[1]}",
                "dataset": "omnidata_replica_gso_debug",
                "scene": args.scene,
                "camera": stem.split("_view_")[1],
                "frame": stem.split("_view_")[0].replace("point_", ""),
                "image": frame["image"],
                "depth": frame["depth"],
                "normal": frame["normal"],
                "duo": [f"gso_{pair[0]}", f"gso_{pair[1]}"],
                "instances": instances,
            }
            duo_records.append(record)
            all_records.append(record)
        duo_summaries.append(
            {
                "duo": [describe_class(pair[0], classes, objects), describe_class(pair[1], classes, objects)],
                "available_records": pair_counts[pair],
                "selected_records": len(duo_records),
            }
        )

    return {
        "dataset": "omnidata_replica_gso_debug_duos",
        "scene": args.scene,
        "semantic_encoding": (
            "class_id = 256 * red + green; instance_id = blue; "
            "GSO objects filtered as class_id >= 102"
        ),
        "records": all_records,
        "duo_summaries": duo_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/klum/omnidata_gso_replica_debug"))
    parser.add_argument("--scene", default="frl_apartment_0-3")
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/omnidata_gso_replica_duos"))
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--duos", type=int, default=8)
    parser.add_argument("--records-per-duo", type=int, default=60)
    parser.add_argument("--min-records", type=int, default=9)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--manifest-name", default="manifest_top_duos.json")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = build_duo_manifest(args)
    manifest_path = args.out / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "record_count": len(manifest["records"]),
                "duos": manifest["duo_summaries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
