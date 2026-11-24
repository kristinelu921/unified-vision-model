#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from omnidata_tools.dataset.starter_dataset import STARTER_DATASET_REMOTE_SERVER_METADATAS


@dataclass(frozen=True)
class MaskJob:
    component: str
    scene: str
    url: str


def existing_rgb_scenes(root: Path) -> set[tuple[str, str]]:
    scenes: set[tuple[str, str]] = set()
    for component_dir in root.iterdir():
        if not component_dir.is_dir() or component_dir.name.startswith("_"):
            continue
        for scene_dir in component_dir.iterdir():
            if scene_dir.is_dir() and (scene_dir / "rgb").is_dir():
                scenes.add((component_dir.name, scene_dir.name))
    return scenes


def remote_mask_jobs(root: Path) -> list[MaskJob]:
    wanted = existing_rgb_scenes(root)
    jobs: list[MaskJob] = []
    for metadata in STARTER_DATASET_REMOTE_SERVER_METADATAS:
        for model in metadata.models:
            key = (model.component_name, model.model_name)
            if model.domain == "mask_valid" and key in wanted:
                jobs.append(MaskJob(model.component_name, model.model_name, model.url))
    jobs.sort(key=lambda job: (job.component, job.scene))
    return jobs


def count_pngs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob("*_domain_mask_valid.png"))


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def download_and_extract(job: MaskJob, root: Path, compressed_root: Path, keep_tar: bool) -> tuple[str, int, str]:
    dst = root / job.component / job.scene / "mask_valid"
    if count_pngs(dst) > 0:
        return "skipped", count_pngs(dst), f"{job.component}/{job.scene}"

    compressed_root.mkdir(parents=True, exist_ok=True)
    tar_path = compressed_root / f"mask_valid__{job.component}__{job.scene}.tar"
    if not tar_path.exists() or tar_path.stat().st_size == 0:
        tmp_tar = tar_path.with_suffix(".tar.part")
        if tmp_tar.exists():
            tmp_tar.unlink()
        run(["curl", "-L", "--fail", "--silent", "--show-error", "-o", str(tmp_tar), job.url])
        os.replace(tmp_tar, tar_path)

    with tempfile.TemporaryDirectory(dir=str(root)) as tmp:
        tmp_root = Path(tmp)
        run(["tar", "-xf", str(tar_path), "-C", str(tmp_root)])
        src = tmp_root / "mask_valid" / job.component / job.scene
        if not src.is_dir():
            src = tmp_root / "mask_valid" / job.scene
        if not src.is_dir():
            return "missing", 0, f"{job.component}/{job.scene}"
        dst.mkdir(parents=True, exist_ok=True)
        for file_path in src.glob("*.png"):
            out_name = file_path.name.replace("_domain_depth_zbuffer.png", "_domain_mask_valid.png")
            shutil.move(str(file_path), str(dst / out_name))

    if not keep_tar:
        tar_path.unlink(missing_ok=True)
    return "downloaded", count_pngs(dst), f"{job.component}/{job.scene}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/klum/omindata")
    parser.add_argument("--compressed-root", default="/mnt/klum/omindata/_compressed/mask_valid")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-tar", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    jobs = remote_mask_jobs(root)
    print(f"matched mask_valid jobs: {len(jobs)}")

    done = 0
    downloaded = 0
    skipped = 0
    missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_and_extract, job, root, Path(args.compressed_root), args.keep_tar)
            for job in jobs
        ]
        for future in as_completed(futures):
            status, n_files, label = future.result()
            done += 1
            downloaded += int(status == "downloaded")
            skipped += int(status == "skipped")
            missing += int(status == "missing")
            if done == 1 or done % 25 == 0 or done == len(jobs):
                print(
                    f"[{done}/{len(jobs)}] downloaded={downloaded} skipped={skipped} missing={missing} "
                    f"last={status}:{label}:{n_files}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
