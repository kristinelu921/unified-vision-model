#!/usr/bin/env python3
"""Extract Omnidata target maps that match already-downloaded RGB samples.

The random downloader samples each domain independently. This script fixes that
by treating existing RGB files as the source of truth and extracting only target
members whose ``point_*_view_*`` stem matches those RGB files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.combo.download_omnidata_random_scene_samples import (  # noqa: E402
    TarEntry,
    load_tar_index,
)


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
TARGET_DOMAINS = ("depth_euclidean", "normal")
SAMPLE_MARKER = "_matching_rgb_manifest.json"


@dataclass(frozen=True)
class TargetJob:
    entry: TarEntry
    missing_stems: frozenset[str]
    rgb_count: int
    existing_count: int


def core_stem(path_or_name: str | Path, domain: str) -> str:
    stem = Path(path_or_name).stem
    suffix = f"_domain_{domain}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file()
        and p.name not in {SAMPLE_MARKER, "_sample_manifest.json"}
        and p.name.lower().endswith(IMAGE_SUFFIXES)
    )


def complete_tar(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with tarfile.open(path) as tar:
            tar.getmembers()
    except (EOFError, OSError, tarfile.TarError):
        return False
    return True


def download(url: str, out_path: Path, connections: int, attempts: int = 4) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if complete_tar(out_path):
        return
    for attempt in range(1, attempts + 1):
        if shutil.which("aria2c"):
            cmd = [
                "aria2c",
                "-c",
                "--auto-file-renaming=false",
                "-s",
                str(connections),
                "-x",
                str(connections),
                "-d",
                str(out_path.parent),
                "-o",
                out_path.name,
                url,
            ]
        elif shutil.which("curl"):
            cmd = ["curl", "-L", "-C", "-", "--fail", "--silent", "--show-error", "-o", str(out_path), url]
        else:
            cmd = ["wget", "--no-netrc", "-c", "-O", str(out_path), url]
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def build_jobs(
    out: Path,
    index: dict[tuple[str, str, str], TarEntry],
    components: set[str] | None,
    domains: tuple[str, ...],
) -> list[TargetJob]:
    jobs: list[TargetJob] = []
    for component_dir in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        component = component_dir.name
        if components is not None and component not in components:
            continue
        for scene_dir in sorted(p for p in component_dir.iterdir() if p.is_dir()):
            rgb_dir = scene_dir / "rgb"
            rgb_stems = {core_stem(p, "rgb") for p in image_files(rgb_dir)}
            if not rgb_stems:
                continue
            for domain in domains:
                target_dir = scene_dir / domain
                existing = {core_stem(p, domain) for p in image_files(target_dir)}
                missing = rgb_stems - existing
                if not missing:
                    continue
                entry = index.get((component, scene_dir.name, domain))
                if entry is None:
                    continue
                jobs.append(
                    TargetJob(
                        entry=entry,
                        missing_stems=frozenset(missing),
                        rgb_count=len(rgb_stems),
                        existing_count=len(existing & rgb_stems),
                    )
                )
    return jobs


def extract_matching(tar_path: Path, out_dir: Path, domain: str, missing_stems: frozenset[str]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    remaining = set(missing_stems)
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not remaining:
                break
            if not member.isfile() or not member.name.lower().endswith(IMAGE_SUFFIXES):
                continue
            stem = core_stem(member.name, domain)
            if stem not in remaining:
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            dest = out_dir / Path(member.name).name
            with source, dest.open("wb") as f:
                shutil.copyfileobj(source, f)
            extracted.append(str(dest))
            remaining.remove(stem)

    with (out_dir / SAMPLE_MARKER).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "matched_to_rgb": True,
                "requested_missing": len(missing_stems),
                "num_extracted_this_run": len(extracted),
                "still_missing": sorted(remaining),
            },
            f,
            indent=2,
        )
    return extracted


def process_job(job: TargetJob, out: Path, compressed: Path, connections: int, keep_compressed: bool) -> dict:
    entry = job.entry
    tar_name = Path(urlparse(entry.url).path).name
    tar_path = compressed / entry.component / entry.domain / tar_name
    target_dir = out / entry.component / entry.scene / entry.domain

    download(entry.url, tar_path, connections)
    extracted = extract_matching(tar_path, target_dir, entry.domain, job.missing_stems)
    if not keep_compressed:
        tar_path.unlink(missing_ok=True)
    return {
        "component": entry.component,
        "scene": entry.scene,
        "domain": entry.domain,
        "requested_missing": len(job.missing_stems),
        "extracted": len(extracted),
        "rgb_count": job.rgb_count,
        "existing_count": job.existing_count,
        "url": entry.url,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/omindata"))
    parser.add_argument("--compressed", type=Path, default=None)
    parser.add_argument("--link-cache-dir", type=Path, default=None)
    parser.add_argument("--components", nargs="*", default=None)
    parser.add_argument("--domains", nargs="+", default=list(TARGET_DOMAINS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--keep-compressed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compressed = args.compressed or args.out / "_compressed"
    link_cache_dir = args.link_cache_dir or args.out / "_link_cache"
    index = load_tar_index(link_cache_dir)
    components = set(args.components) if args.components else None
    domains = tuple(args.domains)
    jobs = build_jobs(args.out, index, components, domains)
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]

    total_missing = sum(len(job.missing_stems) for job in jobs)
    print(f"jobs={len(jobs)} missing_targets={total_missing}", flush=True)
    if args.dry_run:
        by_domain: dict[str, int] = {}
        for job in jobs:
            by_domain[job.entry.domain] = by_domain.get(job.entry.domain, 0) + len(job.missing_stems)
        print(json.dumps(by_domain, indent=2, sort_keys=True), flush=True)
        for job in jobs[:20]:
            print(
                f"{job.entry.component}/{job.entry.scene}/{job.entry.domain} "
                f"missing={len(job.missing_stems)} existing={job.existing_count}/{job.rgb_count}",
                flush=True,
            )
        return

    manifest = {"jobs": []}
    manifest_path = args.out / "matching_targets_manifest.json"

    def write_manifest() -> None:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    if args.workers <= 1:
        for idx, job in enumerate(jobs, 1):
            print(
                f"[{idx}/{len(jobs)}] {job.entry.component}/{job.entry.scene}/{job.entry.domain} "
                f"missing={len(job.missing_stems)}",
                flush=True,
            )
            result = process_job(job, args.out, compressed, args.connections, args.keep_compressed)
            manifest["jobs"].append(result)
            print(f"[{idx}/{len(jobs)}] done extracted={result['extracted']}", flush=True)
            write_manifest()
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(process_job, job, args.out, compressed, args.connections, args.keep_compressed): job
            for job in jobs
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[{idx}/{len(jobs)}] FAILED {job.entry.component}/{job.entry.scene}/{job.entry.domain}: {exc}",
                    flush=True,
                )
                raise
            manifest["jobs"].append(result)
            print(
                f"[{idx}/{len(jobs)}] done {job.entry.component}/{job.entry.scene}/{job.entry.domain} "
                f"extracted={result['extracted']}",
                flush=True,
            )
            write_manifest()


if __name__ == "__main__":
    main()
