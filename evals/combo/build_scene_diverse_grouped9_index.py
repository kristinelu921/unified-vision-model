#!/usr/bin/env python3
"""Build grouped-9 JSONL indexes with scene-diverse groups."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def rows_for_task(samples_manifest: Path) -> dict[str, list[dict]]:
    rows_by_task: dict[str, list[dict]] = defaultdict(list)
    with samples_manifest.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows_by_task[row["task_id"]].append(row)
    return rows_by_task


def scene_diverse_groups(rows: list[dict], records_by_id: dict[str, dict], group_size: int):
    buckets: dict[str, deque[dict]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: item["record_id"]):
        scene = str(records_by_id[row["record_id"]]["scene"])
        buckets[scene].append(row)

    scenes = sorted(buckets, key=lambda scene: (-len(buckets[scene]), scene))
    while True:
        group = []
        active_scenes = list(scenes)
        while len(group) < group_size and active_scenes:
            next_scenes = []
            for scene in active_scenes:
                bucket = buckets[scene]
                if bucket and len(group) < group_size:
                    group.append(bucket.popleft())
                if bucket:
                    next_scenes.append(scene)
            active_scenes = next_scenes
        if len(group) < group_size:
            break
        yield group
        scenes = sorted(
            [scene for scene in scenes if buckets[scene]],
            key=lambda scene: (-len(buckets[scene]), scene),
        )


def build_index(summary_paths: list[Path], out_dir: Path, group_size: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_manifest = out_dir / "manifest.jsonl"
    split_summaries = []
    with all_manifest.open("w", encoding="utf-8") as all_f:
        for summary_path in summary_paths:
            summary = load_json(summary_path)
            combo_manifest = load_json(Path(summary["full_manifest"]))
            records_by_id = {record["id"]: record for record in combo_manifest["records"]}
            combo_root = Path(summary["full_manifest"]).resolve().parent
            rows_by_task = rows_for_task(Path(summary["full_pairs_manifest"]))
            duo = f"{summary['primary_category']}_{summary['secondary_category']}"

            for task_id, rows in sorted(rows_by_task.items()):
                split_name = f"{duo}__task_{task_id}"
                split_path = out_dir / f"{split_name}.jsonl"
                group_count = 0
                scene_counts = []
                with split_path.open("w", encoding="utf-8") as split_f:
                    for group in scene_diverse_groups(rows, records_by_id, group_size):
                        items = []
                        scenes = []
                        for row in group:
                            record = records_by_id[row["record_id"]]
                            scenes.append(str(record["scene"]))
                            primary = [
                                item
                                for item in record["instances"]
                                if item["category"] == summary["primary_category"]
                            ]
                            secondary = [
                                item
                                for item in record["instances"]
                                if item["category"] == summary["secondary_category"]
                            ]
                            items.append(
                                {
                                    "record_id": row["record_id"],
                                    "scene": record["scene"],
                                    "base": row["base"],
                                    "transformed": row["transformed"],
                                    "roles": row["roles"],
                                    "role_groups": row["role_groups"],
                                    "primary_masks": [item["mask"] for item in primary],
                                    "secondary_masks": [item["mask"] for item in secondary],
                                    "primary_instances": primary,
                                    "secondary_instances": secondary,
                                    "annotation": str(Path(row["transformed"]).with_name(task_id + "_meta.json")),
                                }
                            )
                        entry = {
                            "group_id": f"{split_name}:{group_count:05d}",
                            "split_name": split_name,
                            "duo": [summary["primary_category"], summary["secondary_category"]],
                            "task_id": task_id,
                            "task_description": group[0].get("description", ""),
                            "group_size": group_size,
                            "scene_count": len(set(scenes)),
                            "scenes": scenes,
                            "items": items,
                        }
                        line = json.dumps(entry, separators=(",", ":")) + "\n"
                        split_f.write(line)
                        all_f.write(line)
                        scene_counts.append(len(set(scenes)))
                        group_count += 1
                split_summaries.append(
                    {
                        "split_name": split_name,
                        "duo": duo,
                        "task_id": task_id,
                        "groups": group_count,
                        "min_scene_count_per_group": min(scene_counts) if scene_counts else 0,
                        "max_scene_count_per_group": max(scene_counts) if scene_counts else 0,
                        "path": str(split_path),
                        "combo_root": str(combo_root),
                    }
                )
    return {"grouped_manifest": str(all_manifest), "splits": split_summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-paths", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=9)
    args = parser.parse_args()

    summary_paths = [Path(item) for item in args.summary_paths.read_text().strip().split(",") if item]
    summary = build_index(summary_paths, args.out, args.group_size)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "splits": len(summary["splits"]),
                "groups": sum(item["groups"] for item in summary["splits"]),
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
