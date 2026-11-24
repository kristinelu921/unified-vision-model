#!/usr/bin/env python3
"""Build scene-diverse, category-consistent Omnidata Replica-GSO duo manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from build_gso_replica_duos import METADATA_DIR, class_instance_by_id, describe_class, load_gso_names
from build_gso_replica_manifest import collect_frames, write_mask


def available_scenes(root: Path) -> list[str]:
    rgb_root = root / "rgb" / "replica_gso"
    return sorted(path.name for path in rgb_root.iterdir() if path.is_dir())


def collect_scene_frames(root: Path, scenes: list[str], min_area: int) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for scene in scenes:
        frames = collect_frames(root, scene, min_area)
        if frames:
            out[scene] = frames
            print(f"{scene}: {len(frames)} candidate frames")
        else:
            print(f"{scene}: 0 candidate frames")
    return out


def find_scene_diverse_pairs(
    frames_by_scene: dict[str, list[dict[str, Any]]],
    min_records_per_scene: int,
    min_scenes: int,
) -> list[tuple[tuple[int, int], dict[str, int]]]:
    pair_scene_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for scene, frames in frames_by_scene.items():
        for frame in frames:
            class_ids = sorted({item["class_id"] for item in frame["instances"]})
            for pair in combinations(class_ids, 2):
                pair_scene_counts[pair][scene] += 1

    candidates = []
    for pair, scene_counts in pair_scene_counts.items():
        valid = {scene: count for scene, count in scene_counts.items() if count >= min_records_per_scene}
        if len(valid) >= min_scenes:
            candidates.append((pair, valid))
    candidates.sort(key=lambda item: (len(item[1]), sum(item[1].values()), item[0]), reverse=True)
    return candidates


def build_manifest_for_pair(
    *,
    root: Path,
    out_dir: Path,
    frames_by_scene: dict[str, list[dict[str, Any]]],
    pair: tuple[int, int],
    scene_counts: dict[str, int],
    records_per_scene: int,
    classes: list[str],
    objects: list[str],
) -> dict[str, Any]:
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    records = []
    per_scene = []
    for scene in sorted(scene_counts):
        matching_frames = []
        for frame in frames_by_scene[scene]:
            by_id = class_instance_by_id(frame)
            if pair[0] in by_id and pair[1] in by_id:
                matching_frames.append(frame)
        matching_frames.sort(
            key=lambda frame: (
                -class_instance_by_id(frame)[pair[0]]["area"]
                - class_instance_by_id(frame)[pair[1]]["area"],
                frame["stem"],
            )
        )
        selected = matching_frames[:records_per_scene]
        per_scene.append({"scene": scene, "available": len(matching_frames), "selected": len(selected)})
        for frame in selected:
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
            records.append(
                {
                    "id": f"gso_replica_{scene}_{stem}_duo_{pair[0]}_{pair[1]}",
                    "dataset": "omnidata_replica_gso_multiscene_duos",
                    "scene": scene,
                    "camera": stem.split("_view_")[1],
                    "frame": stem.split("_view_")[0].replace("point_", ""),
                    "image": frame["image"],
                    "depth": frame["depth"],
                    "normal": frame["normal"],
                    "duo": [f"gso_{pair[0]}", f"gso_{pair[1]}"],
                    "instances": instances,
                }
            )
    return {
        "dataset": "omnidata_replica_gso_multiscene_duos",
        "root": str(root),
        "primary_category": f"gso_{pair[0]}",
        "secondary_category": f"gso_{pair[1]}",
        "duo": [describe_class(pair[0], classes, objects), describe_class(pair[1], classes, objects)],
        "records": records,
        "per_scene": per_scene,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/klum/omnidata_replica_gso_multiscene"))
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/omnidata_replica_gso_multiscene_duos"))
    parser.add_argument("--metadata-dir", type=Path, default=METADATA_DIR)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--min-records-per-scene", type=int, default=9)
    parser.add_argument("--min-scenes", type=int, default=4)
    parser.add_argument("--records-per-scene", type=int, default=12)
    parser.add_argument("--duos", type=int, default=8)
    args = parser.parse_args()

    scenes = args.scenes or available_scenes(args.root)
    classes, objects = load_gso_names(args.metadata_dir)
    frames_by_scene = collect_scene_frames(args.root, scenes, args.min_area)
    candidates = find_scene_diverse_pairs(frames_by_scene, args.min_records_per_scene, args.min_scenes)
    args.out.mkdir(parents=True, exist_ok=True)

    summary_paths = []
    summaries = []
    for pair, scene_counts in candidates[: args.duos]:
        name = f"gso_{pair[0]}_gso_{pair[1]}"
        duo_dir = args.out / name
        duo_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest_for_pair(
            root=args.root,
            out_dir=duo_dir,
            frames_by_scene=frames_by_scene,
            pair=pair,
            scene_counts=scene_counts,
            records_per_scene=args.records_per_scene,
            classes=classes,
            objects=objects,
        )
        manifest_path = duo_dir / "combo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary = {
            "dataset_name": f"omnidata_replica_gso_multiscene_{name}",
            "scene_type": "replica_gso_multiscene",
            "primary_category": manifest["primary_category"],
            "secondary_category": manifest["secondary_category"],
            "record_count": len(manifest["records"]),
            "scene_count": len({record["scene"] for record in manifest["records"]}),
            "full_manifest": str(manifest_path),
            "full_pairs_dir": str(duo_dir / "samples_24"),
            "full_pairs_manifest": str(duo_dir / "samples_24" / "manifest.jsonl"),
            "raw_subset_dir": str(args.root),
            "tasks_file": str(Path("evals/combo/selected_24_tasks.json").resolve()),
            "duo": manifest["duo"],
            "per_scene": manifest["per_scene"],
        }
        summary_path = duo_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary_paths.append(summary_path)
        summaries.append(summary)
        print(json.dumps({"duo": name, "records": summary["record_count"], "scenes": summary["scene_count"]}))

    (args.out / "summary_paths.txt").write_text(
        ",".join(str(path) for path in summary_paths) + "\n", encoding="utf-8"
    )
    (args.out / "pair_scan_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'summary_paths.txt'}")


if __name__ == "__main__":
    main()
