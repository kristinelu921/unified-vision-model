"""Build a strict grouped-9 UrbanSyn index for TFDS generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CROP_ORIGINAL_XYXY = (512, 0, 1536, 1024)
TARGET_SIZE = 256
MIN_BOX_AREA = TARGET_SIZE * TARGET_SIZE / 50


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _bbox_area_xyxy(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _retained_fraction(original_xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = original_xyxy
    cx1, cy1, cx2, cy2 = CROP_ORIGINAL_XYXY
    clipped = (max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2))
    original_area = _bbox_area_xyxy(original_xyxy)
    if original_area <= 0:
        return 0.0
    return _bbox_area_xyxy(clipped) / float(original_area)


def _original_boxes_by_color(full_root: Path, frame: str) -> dict[tuple[int, int, int, int], dict]:
    rows = _load_json(full_root / "bbox2d" / f"bbox2d_{str(frame).zfill(4)}.json")
    out = {}
    for row in rows:
        box = row["bbox"]
        out[tuple(row["color"])] = {
            "bbox_xyxy_original": [box["xMin"], box["yMin"], box["xMax"], box["yMax"]],
            "occlusion_percentage": row.get("occlusion_percentage"),
        }
    return out


def _enrich_instance(instance: dict, original_lookup: dict[tuple[int, int, int, int], dict]) -> dict | None:
    source_color = tuple(instance["source_color"])
    original = original_lookup.get(source_color)
    if original is None:
        return None
    enriched = dict(instance)
    original_box = tuple(original["bbox_xyxy_original"])
    enriched["bbox_xyxy_original"] = list(original_box)
    enriched["retained_bbox_fraction"] = _retained_fraction(original_box)
    enriched["occlusion_percentage"] = original.get("occlusion_percentage")
    return enriched


def _valid_role(role: dict, record: dict, original_lookup: dict[tuple[int, int, int, int], dict]) -> tuple[bool, dict | None, str | None]:
    instance = next((item for item in record["instances"] if item["id"] == role["id"]), None)
    if instance is None:
        return False, None, "cropped_out"
    enriched = _enrich_instance(instance, original_lookup)
    if enriched is None:
        return False, None, "missing_original_bbox"
    if float(enriched["retained_bbox_fraction"]) < 0.5:
        return False, enriched, "retained_lt_half"
    if int(enriched["bbox_area"]) < MIN_BOX_AREA:
        return False, enriched, "small_box"
    return True, enriched, None


def _strict_rows(pair_dir: Path, full_root: Path, combo: dict) -> tuple[list[dict], Counter]:
    records_by_id = {record["id"]: record for record in combo["records"]}
    drops: Counter = Counter()
    kept = []
    original_cache: dict[str, dict] = {}
    for row in _read_jsonl(pair_dir / "samples_24_256" / "manifest.jsonl"):
        record = records_by_id[row["record_id"]]
        frame = str(record["frame"]).zfill(4)
        if frame not in original_cache:
            original_cache[frame] = _original_boxes_by_color(full_root, frame)
        primary_ok, primary_instance, primary_reason = _valid_role(
            row["roles"]["primary"], record, original_cache[frame]
        )
        secondary_ok, secondary_instance, secondary_reason = _valid_role(
            row["roles"]["secondary"], record, original_cache[frame]
        )
        if not primary_ok or not secondary_ok:
            drops[primary_reason or secondary_reason or "invalid"] += 1
            continue
        kept.append(
            {
                "record_id": row["record_id"],
                "task_id": row["task_id"],
                "description": row.get("description", ""),
                "roles": row["roles"],
                "role_groups": row["role_groups"],
                "base": row["base"],
                "transformed": row["transformed"],
                "scene": record["scene"],
                "camera": record["camera"],
                "frame": str(record["frame"]),
                "primary_instance": primary_instance,
                "secondary_instance": secondary_instance,
                "primary_mask": primary_instance["mask"],
                "secondary_mask": secondary_instance["mask"],
            }
        )
    return kept, drops


def _group_rows(rows: list[dict], group_size: int = 9) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row["scene"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (int(row["frame"]), row["record_id"]))
    groups = []
    while len(buckets) >= group_size:
        scenes = sorted(buckets, key=lambda scene: (-len(buckets[scene]), scene))[:group_size]
        group = [buckets[scene].pop(0) for scene in scenes]
        groups.append(group)
        buckets = {scene: bucket for scene, bucket in buckets.items() if bucket}
    return groups


def build_index(source_root: Path, full_root: Path, out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "UrbanSyn",
        "source_root": str(source_root),
        "full_root": str(full_root),
        "target_size": TARGET_SIZE,
        "min_box_area": MIN_BOX_AREA,
        "min_retained_bbox_fraction": 0.5,
        "splits": [],
        "drop_reasons": {},
    }
    total_groups = 0
    total_rows = 0
    total_drops: Counter = Counter()
    for pair_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        combo_path = pair_dir / "combo_manifest_256.json"
        if not combo_path.is_file():
            continue
        combo = _load_json(combo_path)
        kept, drops = _strict_rows(pair_dir, full_root, combo)
        total_drops.update(drops)
        by_task: dict[str, list[dict]] = {}
        for row in kept:
            by_task.setdefault(row["task_id"], []).append(row)
        duo = _sanitize(pair_dir.name)
        for task_id, task_rows in sorted(by_task.items()):
            groups = _group_rows(task_rows)
            if not groups:
                continue
            split_name = f"{duo}__task_{_sanitize(task_id)}"
            split_path = out_root / f"{split_name}.jsonl"
            with split_path.open("w", encoding="utf-8") as f:
                for group_index, group in enumerate(groups):
                    item = {
                        "group_id": f"{split_name}:{group_index:05d}",
                        "split_name": split_name,
                        "duo": [combo["primary_category"], combo["secondary_category"]],
                        "task_id": task_id,
                        "task_description": group[0].get("description", ""),
                        "scene_count": len({row["scene"] for row in group}),
                        "group_size": len(group),
                        "items": group,
                    }
                    f.write(json.dumps(item, separators=(",", ":")) + "\n")
            summary["splits"].append(
                {
                    "split_name": split_name,
                    "pair": pair_dir.name,
                    "task_id": task_id,
                    "path": str(split_path),
                    "groups": len(groups),
                    "rows": len(groups) * 9,
                }
            )
            total_groups += len(groups)
            total_rows += len(groups) * 9
    summary["groups"] = total_groups
    summary["rows"] = total_rows
    summary["active_splits"] = len(summary["splits"])
    summary["drop_reasons"] = dict(total_drops)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("/mnt/klum/lvm/diverse_dataset/urbansyn/selected_pairs_24task"))
    parser.add_argument("--full-root", type=Path, default=Path("/mnt/klum/lvm/diverse_dataset/urbansyn/full"))
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/klum/lvm/diverse_dataset/urban_syn_tfds_index"))
    args = parser.parse_args()
    summary = build_index(args.source_root, args.full_root, args.out_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
