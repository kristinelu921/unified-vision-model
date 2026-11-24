#!/usr/bin/env python3
"""Build single-primary-object manifests from existing combo manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


DEFAULT_SOURCE_MANIFESTS = [
    Path("/mnt/klum/combo_hypersim_scaled/combo_manifest.json"),
    Path("/mnt/klum/combo_hypersim_livingroom_sofa_table/combo_manifest.json"),
    Path("/mnt/klum/combo_hypersim_bedroom_bed_nightstand/combo_manifest.json"),
    Path("/mnt/klum/combo_hypersim_kitchen_fridge_cabinet/combo_manifest.json"),
    Path("/mnt/klum/combo_hypersim_dining_chair_table/combo_manifest.json"),
    Path("/mnt/klum/combo_hypersim_restaurant_chair_table/combo_manifest.json"),
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_records(source_manifests: list[Path]) -> dict[str, dict]:
    records = {}
    for manifest_path in source_manifests:
        manifest = read_json(manifest_path)
        for record in manifest["records"]:
            records.setdefault(record["id"], record)
    return records


def record_for_category(record: dict, category: str) -> dict | None:
    matching = [inst for inst in record["instances"] if inst.get("category") == category]
    if not matching:
        return None
    out = deepcopy(record)
    out["instances"].sort(
        key=lambda inst: (
            0 if inst.get("category") == category else 1,
            -int(inst.get("area", 0)),
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/klum/combo_hypersim_single_object"))
    parser.add_argument("--min-records", type=int, default=60)
    parser.add_argument("--max-records-per-class", type=int, default=0)
    parser.add_argument(
        "--categories",
        nargs="*",
        default=["chair", "table", "lamp", "shelves", "picture", "blinds"],
    )
    parser.add_argument("--source-manifests", nargs="*", type=Path, default=DEFAULT_SOURCE_MANIFESTS)
    args = parser.parse_args()

    records_by_id = collect_records(args.source_manifests)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in records_by_id.values():
        seen = {inst.get("category") for inst in record["instances"]}
        for category in seen:
            if category:
                prepared = record_for_category(record, category)
                if prepared is not None:
                    by_category[category].append(prepared)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": "Hypersim existing combo manifests",
        "task_count": 10,
        "min_records": args.min_records,
        "classes": [],
    }
    for category in args.categories:
        records = sorted(by_category.get(category, []), key=lambda record: record["id"])
        if args.max_records_per_class > 0:
            records = records[: args.max_records_per_class]
        if len(records) < args.min_records:
            print(f"Skipping {category}: only {len(records)} records")
            continue

        class_dir = args.out_root / category
        class_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source": "Hypersim",
            "record_count": len(records),
            "primary_category": category,
            "secondary_category": None,
            "records": records,
        }
        manifest_path = class_dir / "combo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary["classes"].append(
            {
                "category": category,
                "record_count": len(records),
                "manifest": str(manifest_path),
                "pairs_dir": str(class_dir / "samples_10"),
                "pairs_manifest": str(class_dir / "samples_10" / "manifest.jsonl"),
                "pair_count": len(records) * 10,
            }
        )
        print(f"Wrote {len(records)} {category} records to {manifest_path}")

    summary_path = args.out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
