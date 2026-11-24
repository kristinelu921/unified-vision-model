#!/usr/bin/env python3
"""Build expanded Hypersim duo manifests, transformations, and summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from build_hypersim_collection import merge_manifests
from build_hypersim_duo_from_downloaded import available_scene_cameras
from build_hypersim_manifest import build_manifest


DEFAULT_PAIRS = [
    ("chair", "lamp"),
    ("lamp", "table"),
    ("desk", "lamp"),
    ("cabinet", "chair"),
    ("chair", "table"),
]


def default_roots() -> list[Path]:
    roots = []
    for pattern in ("hypersim_subset*", "hypersim_more_*", "hypersim_probe_*"):
        roots.extend(Path("/mnt/klum").glob(pattern))
    return sorted({root.resolve() for root in roots if root.is_dir()})


def pair_name(primary: str, secondary: str) -> str:
    return f"{primary}_{secondary}"


def build_pair_manifest(
    *,
    roots: list[Path],
    out_dir: Path,
    primary: str,
    secondary: str,
    min_area: int,
    max_frames: int,
) -> dict:
    manifests = []
    scene_cameras = []
    for root, scene, camera in available_scene_cameras(roots):
        scene_out_dir = out_dir / f"{scene}_{camera}"
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            subset_dir=root,
            out_dir=scene_out_dir,
            scene=scene,
            camera=camera,
            min_area=min_area,
            max_frames=max_frames,
            primary_category=primary,
            secondary_category=secondary,
        )
        if manifest["records"]:
            manifests.append(manifest)
            scene_cameras.append(f"{scene}_{camera}")

    merged = merge_manifests(manifests, Path("/mnt/klum"))
    merged["scene_cameras"] = scene_cameras
    merged["scene_type"] = "expanded"
    merged["frames_per_camera"] = max_frames
    merged["primary_category"] = primary
    merged["secondary_category"] = secondary
    merged["subset_roots"] = [str(root) for root in roots]
    return merged


def count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, default=default_roots())
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/klum/combo_hypersim_expanded_duos"))
    parser.add_argument("--summary-dir", type=Path, default=Path("evals/combo"))
    parser.add_argument("--tasks", type=Path, default=Path("evals/combo/selected_16_tasks.json"))
    parser.add_argument("--min-area", type=int, default=1000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-transform-errors", action="store_true")
    parser.add_argument("--generate-jobs", type=int, default=1)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{primary}:{secondary}" for primary, secondary in DEFAULT_PAIRS],
        help="Duo pairs as primary:secondary.",
    )
    args = parser.parse_args()

    roots = [root.resolve() for root in args.roots if root.is_dir()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = []
    for pair in args.pairs:
        primary, secondary = pair.split(":", 1)
        name = pair_name(primary, secondary)
        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = build_pair_manifest(
            roots=roots,
            out_dir=out_dir,
            primary=primary,
            secondary=secondary,
            min_area=args.min_area,
            max_frames=args.max_frames,
        )
        manifest_path = out_dir / "combo_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        samples_dir = out_dir / "samples_20"
        samples_manifest = samples_dir / "manifest.jsonl"
        if not args.skip_generate:
            cmd = [
                    sys.executable,
                    "evals/combo/generate_combo_pairs.py",
                    "--manifest",
                    str(manifest_path),
                    "--tasks",
                    str(args.tasks),
                    "--out",
                    str(samples_dir),
                ]
            if args.skip_transform_errors:
                cmd.append("--skip-errors")
            if args.generate_jobs > 1:
                cmd.extend(["--jobs", str(args.generate_jobs)])
            subprocess.run(cmd, check=True)

        pair_count = count_jsonl(samples_manifest) if samples_manifest.is_file() else 0
        summary = {
            "dataset_name": f"hypersim_combo_expanded_{name}",
            "scene_type": "expanded",
            "primary_category": primary,
            "secondary_category": secondary,
            "task_count": 20,
            "pair_count": pair_count,
            "record_count": len(manifest["records"]),
            "scene_count": len({record["scene"] for record in manifest["records"]}),
            "scene_camera_count": len({(record["scene"], record["camera"]) for record in manifest["records"]}),
            "raw_subset_dir": "/mnt/klum",
            "full_manifest": str(manifest_path),
            "full_pairs_dir": str(samples_dir),
            "full_pairs_manifest": str(samples_manifest),
            "tasks_file": str(args.tasks.resolve()),
            "subset_roots": [str(root) for root in roots],
        }
        summary_path = args.summary_dir / f"hypersim_expanded_{name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary_paths.append(summary_path)
        print(
            json.dumps(
                {
                    "summary": str(summary_path),
                    "records": summary["record_count"],
                    "scenes": summary["scene_count"],
                    "scene_cameras": summary["scene_camera_count"],
                    "pairs": summary["pair_count"],
                },
                indent=2,
            )
        )

    manifest_path = args.out_root / "summary_paths.txt"
    manifest_path.write_text(",".join(str(path.resolve()) for path in summary_paths) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
