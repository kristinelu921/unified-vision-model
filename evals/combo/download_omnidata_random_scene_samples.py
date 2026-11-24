#!/usr/bin/env python3
"""Download random per-scene samples from Omnidata starter tarballs.

The Omnidata server exposes scene/domain tar files, not per-image URLs. This
script downloads a tar, extracts a random subset of image-like members, then
optionally deletes the tar.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


BASE_URL = "https://datasets.epfl.ch"
OMNIDATA_LINKS_URL = f"{BASE_URL}/omnidata/links.txt"
TASKONOMY_LINKS_URL = f"{BASE_URL}/taskonomy/links.txt"
SPLITS_ROOT = Path(
    "/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310/lib/python3.10/"
    "site-packages/omnidata_tools/dataset/starter_dataset/component_datasets"
)

DEFAULT_COMPONENTS = ("hypersim", "replica", "replica_gso", "blended_mvg", "taskonomy")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
SAMPLE_MARKER = "_sample_manifest.json"
TASKONOMY_COMPONENT = "taskonomy"


@dataclass(frozen=True)
class TarEntry:
    component: str
    scene: str
    domain: str
    url: str


def _read_split_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    out = []
    for row in rows:
        if not row:
            continue
        scene = row[0].strip()
        if scene.lower() in {"id", "name", "building"}:
            continue
        if scene:
            out.append(scene)
    return sorted(set(out))


def _component_scenes(component: str) -> list[str]:
    if component == "taskonomy":
        path = SPLITS_ROOT / "taskonomy" / "train_val_test_fullplus.csv"
    elif component == "blended_mvg":
        path = SPLITS_ROOT / "blendedmvg" / "train_val_test_blendedMVG.csv"
    elif component == "replica_gso":
        path = SPLITS_ROOT / "replica_gso" / "train_val_test_gso.csv"
    elif component == "hypersim":
        path = SPLITS_ROOT / "hypersim" / "train_val_test_hypersim.csv"
    elif component == "replica":
        path = SPLITS_ROOT / "replica" / "train_val_test_replica.csv"
    else:
        raise ValueError(f"Unsupported component {component!r}.")
    return _read_split_csv(path)


def _parse_omnidata_url(url: str) -> TarEntry | None:
    # .../omnidata_tars/{domain}/{component}/{domain}-{component}-{scene}.tar
    path = urlparse(url).path.lstrip("/")
    parts = path.split("/")
    if len(parts) < 5 or parts[0] != "omnidata" or parts[1] != "omnidata_tars":
        return None
    domain, component, filename = parts[2], parts[3], parts[4]
    suffix = ".tar.gz" if filename.endswith(".tar.gz") else ".tar"
    stem = filename[: -len(suffix)]
    prefix = f"{domain}-{component}-"
    if not stem.startswith(prefix):
        return None
    scene = stem[len(prefix) :]
    return TarEntry(component=component, scene=scene, domain=domain, url=url)


def _parse_taskonomy_url(url: str) -> TarEntry | None:
    # .../taskonomy/{scene}_{domain}.tar
    filename = Path(urlparse(url).path).name
    if not filename.endswith(".tar"):
        return None
    stem = filename[:-4]
    scene, sep, domain = stem.partition("_")
    if not sep:
        return None
    return TarEntry(component="taskonomy", scene=scene, domain=domain, url=url)


def load_tar_index(
    cache_dir: Path | None = None,
    *,
    skip_omnidata_index: bool = False,
) -> dict[tuple[str, str, str], TarEntry]:
    index: dict[tuple[str, str, str], TarEntry] = {}
    if not skip_omnidata_index:
        try:
            omnidata_links = get_text_with_retries(OMNIDATA_LINKS_URL, cache_dir, "omnidata_links.txt")
        except RuntimeError as exc:
            print(f"warning: {exc}; deriving non-Taskonomy URLs from split files", flush=True)
        else:
            for url in omnidata_links.splitlines():
                entry = _parse_omnidata_url(url.strip())
                if entry is not None:
                    index[(entry.component, entry.scene, entry.domain)] = entry
    for url in get_text_with_retries(TASKONOMY_LINKS_URL, cache_dir, "taskonomy_links.txt").splitlines():
        entry = _parse_taskonomy_url(url.strip())
        if entry is not None:
            index[(entry.component, entry.scene, entry.domain)] = entry
    return index


def derive_omnidata_entry(component: str, scene: str, domain: str) -> TarEntry:
    url = (
        f"{BASE_URL}/omnidata/omnidata_tars/{domain}/{component}/"
        f"{domain}-{component}-{scene}.tar"
    )
    return TarEntry(component=component, scene=scene, domain=domain, url=url)


def get_text_with_retries(
    url: str,
    cache_dir: Path | None,
    cache_name: str,
    attempts: int = 4,
    timeout: int = 180,
) -> str:
    cache_path = cache_dir / cache_name if cache_dir is not None else None
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8")
    last_error: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def remote_size_bytes(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None
    content_length = response.headers.get("content-length")
    if content_length is None:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def is_complete_tar(path: Path) -> bool:
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
    if is_complete_tar(out_path):
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
            cmd = ["curl", "-L", "-C", "-", "--fail", "-o", str(out_path), url]
        else:
            cmd = ["wget", "--no-netrc", "-c", "-O", str(out_path), url]
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def extract_random_members(
    tar_path: Path,
    out_dir: Path,
    *,
    samples_per_scene: int,
    seed: int | str,
    rgb_jpeg_quality: int = 0,
) -> list[dict[str, str]]:
    with tarfile.open(tar_path) as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile() and member.name.lower().endswith(IMAGE_SUFFIXES)
        ]
        rng = random.Random(seed)
        extracted = []
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_paths = [
            path
            for path in sorted(out_dir.iterdir())
            if path.is_file() and path.name.lower().endswith(IMAGE_SUFFIXES)
        ]
        existing_keys = {path.stem for path in existing_paths}
        extracted.extend({"member": path.name, "path": str(path)} for path in existing_paths)
        needed = max(0, samples_per_scene - len(existing_paths))
        candidates = [member for member in members if Path(member.name).stem not in existing_keys]
        selected = rng.sample(candidates, min(needed, len(candidates)))
        for member in selected:
            member_path = Path(member.name)
            dest = out_dir / member_path.name
            if rgb_jpeg_quality > 0:
                dest = dest.with_suffix(".jpg")
            source = tar.extractfile(member)
            if source is None:
                continue
            if rgb_jpeg_quality > 0:
                with source, Image.open(source) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dest, "JPEG", quality=rgb_jpeg_quality, optimize=False, progressive=False)
            else:
                with source, dest.open("wb") as f:
                    shutil.copyfileobj(source, f)
            extracted.append({"member": member.name, "path": str(dest)})
            existing_keys.add(dest.stem)
        with (out_dir / SAMPLE_MARKER).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "requested_samples": samples_per_scene,
                    "available_members": len(members),
                    "num_extracted": len(extracted),
                    "seed": str(seed),
                },
                f,
                indent=2,
            )
    return extracted


def existing_samples(out_dir: Path, samples_per_scene: int) -> list[dict[str, str]]:
    if not out_dir.exists():
        return []
    paths = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(IMAGE_SUFFIXES)
    )
    if len(paths) < samples_per_scene:
        marker_path = out_dir / SAMPLE_MARKER
        if marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                marker = {}
            requested = int(marker.get("requested_samples", 0))
            extracted = int(marker.get("num_extracted", -1))
            available = int(marker.get("available_members", samples_per_scene))
            if requested >= samples_per_scene and available <= samples_per_scene and extracted == len(paths):
                return [{"member": path.name, "path": str(path)} for path in paths]
        return []
    return [{"member": path.name, "path": str(path)} for path in paths[:samples_per_scene]]


def has_any_samples(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any(
        path.is_file() and path.name.lower().endswith(IMAGE_SUFFIXES)
        for path in out_dir.iterdir()
    )


def process_job(
    entry: TarEntry,
    *,
    out: Path,
    compressed: Path,
    samples_per_scene: int,
    seed: int,
    connections: int,
    keep_compressed: bool,
    skip_taskonomy_larger_than_bytes: int,
    rgb_jpeg_quality: int,
) -> dict:
    tar_path = compressed / entry.component / entry.domain / Path(urlparse(entry.url).path).name
    sample_dir = out / entry.component / entry.scene / entry.domain
    extracted = existing_samples(sample_dir, samples_per_scene)
    if extracted:
        return {
            "component": entry.component,
            "scene": entry.scene,
            "domain": entry.domain,
            "url": entry.url,
            "tar_path": str(tar_path),
            "num_extracted": len(extracted),
            "samples": extracted,
            "skipped_existing": True,
        }
    remote_size = None
    if entry.component == TASKONOMY_COMPONENT and skip_taskonomy_larger_than_bytes > 0:
        remote_size = remote_size_bytes(entry.url)
        if remote_size is not None and remote_size > skip_taskonomy_larger_than_bytes:
            return {
                "component": entry.component,
                "scene": entry.scene,
                "domain": entry.domain,
                "url": entry.url,
                "tar_path": str(tar_path),
                "num_extracted": 0,
                "samples": [],
                "skipped_existing": False,
                "skipped_large_taskonomy": True,
                "remote_size_bytes": remote_size,
            }
    download(entry.url, tar_path, connections)
    extracted = extract_random_members(
        tar_path,
        sample_dir,
        samples_per_scene=samples_per_scene,
        seed=f"{seed}:{entry.component}:{entry.scene}:{entry.domain}",
        rgb_jpeg_quality=rgb_jpeg_quality if entry.domain == "rgb" else 0,
    )
    if not keep_compressed:
        tar_path.unlink(missing_ok=True)
    return {
        "component": entry.component,
        "scene": entry.scene,
        "domain": entry.domain,
        "url": entry.url,
        "tar_path": str(tar_path),
        "num_extracted": len(extracted),
        "samples": extracted,
        "skipped_existing": False,
        "skipped_large_taskonomy": False,
        "remote_size_bytes": remote_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/mnt/klum/omindata"))
    parser.add_argument("--compressed", type=Path, default=None)
    parser.add_argument("--link-cache-dir", type=Path, default=None)
    parser.add_argument("--skip-omnidata-index", action="store_true")
    parser.add_argument("--components", nargs="+", default=list(DEFAULT_COMPONENTS))
    parser.add_argument("--domains", nargs="+", default=["rgb"])
    parser.add_argument(
        "--scene-list-file",
        type=Path,
        default=None,
        help="Optional newline-delimited scene ids to use instead of component split files.",
    )
    parser.add_argument("--samples-per-scene", type=int, default=200)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Cap total jobs after component/domain filtering and skip-existing checks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-compressed", action="store_true")
    parser.add_argument(
        "--rgb-jpeg-quality",
        type=int,
        default=0,
        help="If >0, save extracted RGB samples as JPEG at this quality instead of PNG.",
    )
    parser.add_argument(
        "--skip-existing-any",
        action="store_true",
        help="Skip scene/domain jobs that already have any extracted samples.",
    )
    parser.add_argument(
        "--skip-taskonomy-larger-than-gb",
        type=float,
        default=0.0,
        help="Skip Taskonomy tarballs with Content-Length larger than this many GB.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    compressed = args.compressed or args.out / "_compressed"
    link_cache_dir = args.link_cache_dir or args.out / "_link_cache"
    index = load_tar_index(link_cache_dir, skip_omnidata_index=args.skip_omnidata_index)

    rng = random.Random(args.seed)
    jobs: list[TarEntry] = []
    explicit_scenes = None
    if args.scene_list_file is not None:
        explicit_scenes = [
            line.strip()
            for line in args.scene_list_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    for component in args.components:
        scenes = list(explicit_scenes) if explicit_scenes is not None else _component_scenes(component)
        if explicit_scenes is None:
            rng.shuffle(scenes)
        if args.max_scenes:
            scenes = scenes[: args.max_scenes]
        for scene in scenes:
            for domain in args.domains:
                lookup_domain = "segment_semantic" if component == "taskonomy" and domain == "semantic" else domain
                entry = index.get((component, scene, lookup_domain))
                if entry is None and component != TASKONOMY_COMPONENT:
                    entry = derive_omnidata_entry(component, scene, lookup_domain)
                if entry is not None:
                    if args.skip_existing_any and has_any_samples(args.out / entry.component / entry.scene / entry.domain):
                        continue
                    jobs.append(entry)
                    if args.max_jobs and len(jobs) >= args.max_jobs:
                        break
            if args.max_jobs and len(jobs) >= args.max_jobs:
                break
        if args.max_jobs and len(jobs) >= args.max_jobs:
            break

    manifest = {
        "out": str(args.out),
        "components": args.components,
        "domains": args.domains,
        "samples_per_scene": args.samples_per_scene,
        "seed": args.seed,
        "skip_taskonomy_larger_than_gb": args.skip_taskonomy_larger_than_gb,
        "jobs": [],
    }
    print(f"jobs={len(jobs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for entry in jobs[:20]:
            print(entry)
        print(f"dry_run_manifest={args.out / 'manifest.json'}")
        return

    def write_manifest() -> None:
        with (args.out / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    workers = max(1, args.workers)
    skip_taskonomy_larger_than_bytes = int(args.skip_taskonomy_larger_than_gb * 1024**3)
    if workers == 1:
        for idx, entry in enumerate(jobs, 1):
            print(f"[{idx}/{len(jobs)}] {entry.component}/{entry.scene}/{entry.domain}", flush=True)
            result = process_job(
                entry,
                out=args.out,
                compressed=compressed,
                samples_per_scene=args.samples_per_scene,
                seed=args.seed,
                connections=args.connections,
                keep_compressed=args.keep_compressed,
                skip_taskonomy_larger_than_bytes=skip_taskonomy_larger_than_bytes,
                rgb_jpeg_quality=args.rgb_jpeg_quality,
            )
            manifest["jobs"].append(result)
            write_manifest()
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_entry = {
            executor.submit(
                process_job,
                entry,
                out=args.out,
                compressed=compressed,
                samples_per_scene=args.samples_per_scene,
                seed=args.seed,
                connections=args.connections,
                keep_compressed=args.keep_compressed,
                skip_taskonomy_larger_than_bytes=skip_taskonomy_larger_than_bytes,
                rgb_jpeg_quality=args.rgb_jpeg_quality,
            ): entry
            for entry in jobs
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_entry):
            entry = future_to_entry[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                print(
                    f"[{completed}/{len(jobs)}] FAILED "
                    f"{entry.component}/{entry.scene}/{entry.domain}: {exc}",
                    flush=True,
                )
                raise
            manifest["jobs"].append(result)
            if result.get("skipped_large_taskonomy"):
                size_gb = result.get("remote_size_bytes", 0) / 1024**3
                print(
                    f"[{completed}/{len(jobs)}] skipped_large "
                    f"{entry.component}/{entry.scene}/{entry.domain} "
                    f"size_gb={size_gb:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"[{completed}/{len(jobs)}] done "
                    f"{entry.component}/{entry.scene}/{entry.domain} "
                    f"extracted={result['num_extracted']}",
                    flush=True,
                )
            write_manifest()


if __name__ == "__main__":
    main()
