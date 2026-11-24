#!/usr/bin/env python3
"""Download remaining Replica-GSO scenes one scene at a time."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DOMAINS = ["rgb", "depth_euclidean", "normal", "semantic"]
SPLIT_CSV = Path(
    "/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310/lib/python3.10/"
    "site-packages/omnidata_tools/dataset/starter_dataset/component_datasets/"
    "replica_gso/train_val_test_gso.csv"
)
DOWNLOADER = Path(__file__).with_name("download_omnidata_replica_gso_scenes.py")


def load_scenes(split_csv: Path) -> list[str]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        return [row["id"].strip() for row in csv.DictReader(f)]


def is_complete(root: Path, scene: str) -> bool:
    return all((root / domain / "replica_gso" / scene).is_dir() for domain in DOMAINS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("/mnt/klum/omnidata_replica_gso_multiscene"))
    parser.add_argument(
        "--compressed",
        type=Path,
        default=Path("/mnt/klum/omnidata_replica_gso_multiscene_compressed"),
    )
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-complete", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    scenes = load_scenes(args.split_csv)
    if args.skip_complete:
        scenes = [scene for scene in scenes if not is_complete(args.dest, scene)]
    if args.limit:
        scenes = scenes[: args.limit]

    print(f"scenes_to_download={len(scenes)}", flush=True)
    for idx, scene in enumerate(scenes, 1):
        print(f"=== {idx}/{len(scenes)} {scene} ===", flush=True)
        cmd = [
            sys.executable,
            str(DOWNLOADER),
            "--dest",
            str(args.dest),
            "--compressed",
            str(args.compressed),
            "--scenes",
            scene,
            "--connections",
            str(args.connections),
            "--keep-compressed",
        ]
        subprocess.run(cmd, check=True)

    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
