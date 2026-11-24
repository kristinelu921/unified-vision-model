#!/usr/bin/env python3
"""Download, build, and merge a larger multi-scene Hypersim combo dataset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from build_hypersim_manifest import build_manifest


def load_scene_cameras(csv_path: Path, scene_type: str, max_scene_cameras: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Scene type"].strip() != scene_type:
                continue
            animation = row["Animation"].strip()
            parts = animation.split("_")
            if len(parts) < 5:
                continue
            scene = "_".join(parts[:3])
            camera = "_".join(parts[3:5])
            pairs.append((scene, camera))
    if max_scene_cameras > 0:
        pairs = pairs[:max_scene_cameras]
    return pairs


def download_scene_camera(
    downloader: Path,
    subset_dir: Path,
    scene: str,
    camera: str,
    frames_per_camera: int,
    retries: int,
) -> None:
    commands = [
        [
            sys.executable,
            str(downloader),
            "--directory",
            str(subset_dir),
            "--scene",
            scene,
            "--contains",
            "_detail",
            "--silent",
        ]
    ]

    for frame_idx in range(frames_per_camera):
        frame = f"{frame_idx:04d}"
        contains_specs = [
            [f"scene_{camera}_final_preview", f"frame.{frame}.color.jpg"],
            [f"scene_{camera}_geometry_hdf5", f"frame.{frame}.depth_meters.hdf5"],
            [f"scene_{camera}_geometry_hdf5", f"frame.{frame}.normal_cam.hdf5"],
            [f"scene_{camera}_geometry_hdf5", f"frame.{frame}.semantic_instance.hdf5"],
            [f"scene_{camera}_geometry_hdf5", f"frame.{frame}.semantic.hdf5"],
        ]
        for contains in contains_specs:
            commands.append(
                [
                    sys.executable,
                    str(downloader),
                    "--directory",
                    str(subset_dir),
                    "--scene",
                    scene,
                    "--contains",
                    *contains,
                    "--silent",
                ]
            )

    for cmd in commands:
        for attempt in range(retries + 1):
            try:
                subprocess.run(cmd, check=True)
                break
            except subprocess.CalledProcessError:
                if attempt >= retries:
                    print(f"Failed after {retries + 1} attempts: {' '.join(cmd)}", file=sys.stderr)
                    raise
                time.sleep(2.0 * (attempt + 1))


def merge_manifests(manifests: list[dict], subset_dir: Path) -> dict:
    records = []
    for manifest in manifests:
        records.extend(manifest["records"])
    records.sort(key=lambda record: record["id"])
    return {
        "source": "Hypersim",
        "subset_dir": str(subset_dir),
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, default=Path("/mnt/klum/hypersim_subset_scaled"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/klum/combo_hypersim_scaled"))
    parser.add_argument(
        "--camera-trajectories-csv",
        type=Path,
        default=Path("/mnt/klum/hypersim_tools/evermotion_dataset/analysis/metadata_camera_trajectories.csv"),
    )
    parser.add_argument(
        "--downloader",
        type=Path,
        default=Path("/mnt/klum/hypersim_tools/contrib/99991/download.py"),
    )
    parser.add_argument("--scene-type", default="Bathroom")
    parser.add_argument("--max-scene-cameras", type=int, default=8)
    parser.add_argument("--frames-per-camera", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--primary-category", default="bathtub")
    parser.add_argument("--secondary-category", default="sink")
    parser.add_argument("--manifest-name", default="combo_manifest.json")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--download-retries", type=int, default=2)
    parser.add_argument("--skip-failed-downloads", action="store_true")
    args = parser.parse_args()

    args.subset_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene_cameras = load_scene_cameras(
        csv_path=args.camera_trajectories_csv,
        scene_type=args.scene_type,
        max_scene_cameras=args.max_scene_cameras,
    )

    manifests = []
    for scene, camera in scene_cameras:
        if not args.skip_download:
            try:
                download_scene_camera(
                    downloader=args.downloader,
                    subset_dir=args.subset_dir,
                    scene=scene,
                    camera=camera,
                    frames_per_camera=args.frames_per_camera,
                    retries=args.download_retries,
                )
            except subprocess.CalledProcessError:
                if args.skip_failed_downloads:
                    continue
                raise

        scene_out_dir = args.out_dir / f"{scene}_{camera}"
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            subset_dir=args.subset_dir,
            out_dir=scene_out_dir,
            scene=scene,
            camera=camera,
            min_area=args.min_area,
            max_frames=args.frames_per_camera,
            primary_category=args.primary_category,
            secondary_category=args.secondary_category,
        )
        if manifest["records"]:
            manifests.append(manifest)

    merged = merge_manifests(manifests, args.subset_dir)
    merged["scene_cameras"] = [f"{scene}_{camera}" for scene, camera in scene_cameras]
    merged["scene_type"] = args.scene_type
    merged["frames_per_camera"] = args.frames_per_camera
    merged["primary_category"] = args.primary_category
    merged["secondary_category"] = args.secondary_category

    manifest_path = args.out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(merged['records'])} records to {manifest_path}")


if __name__ == "__main__":
    main()
