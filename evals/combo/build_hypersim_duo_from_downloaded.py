#!/usr/bin/env python3
"""Build duo manifests from all already-downloaded Hypersim subset roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_hypersim_collection import merge_manifests
from build_hypersim_manifest import build_manifest


def available_scene_cameras(roots: list[Path]) -> list[tuple[Path, str, str]]:
    seen = set()
    out = []
    for root in roots:
        for geometry_dir in sorted(root.glob("*/images/scene_*_geometry_hdf5")):
            scene = geometry_dir.parts[-3]
            camera = geometry_dir.name.removeprefix("scene_").removesuffix("_geometry_hdf5")
            key = (scene, camera)
            if key in seen:
                continue
            seen.add(key)
            out.append((root, scene, camera))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=sorted(Path("/mnt/klum").glob("hypersim_subset*")),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/klum/combo_hypersim_all_downloaded_chair_table"))
    parser.add_argument("--primary-category", default="chair")
    parser.add_argument("--secondary-category", default="table")
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--manifest-name", default="combo_manifest.json")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    scene_cameras = []
    for root, scene, camera in available_scene_cameras(args.roots):
        scene_out_dir = args.out_dir / f"{scene}_{camera}"
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            subset_dir=root,
            out_dir=scene_out_dir,
            scene=scene,
            camera=camera,
            min_area=args.min_area,
            max_frames=args.max_frames,
            primary_category=args.primary_category,
            secondary_category=args.secondary_category,
        )
        if manifest["records"]:
            manifests.append(manifest)
            scene_cameras.append(f"{scene}_{camera}")

    merged = merge_manifests(manifests, args.roots[0] if len(args.roots) == 1 else Path("/mnt/klum"))
    merged["scene_cameras"] = scene_cameras
    merged["scene_type"] = "all_downloaded"
    merged["frames_per_camera"] = args.max_frames
    merged["primary_category"] = args.primary_category
    merged["secondary_category"] = args.secondary_category
    merged["subset_roots"] = [str(root) for root in args.roots]

    manifest_path = args.out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records": len(merged["records"]),
                "scene_cameras": len(scene_cameras),
                "scenes": len({item.rsplit("_cam_", 1)[0] for item in scene_cameras}),
                "primary_category": args.primary_category,
                "secondary_category": args.secondary_category,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
