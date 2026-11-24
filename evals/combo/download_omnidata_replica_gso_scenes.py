#!/usr/bin/env python3
"""Download selected Omnidata Replica-GSO scenes and domains.

The official Omnidata downloader exposes debug/fullplus subsets, but this helper
keeps the download scoped to named Replica-GSO scenes so category-consistent
multi-scene experiments do not require pulling all 54 scenes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path


BASE_URL = "https://datasets.epfl.ch/omnidata/omnidata_tars"
DEFAULT_DOMAINS = ["rgb", "depth_euclidean", "normal", "semantic"]
DEFAULT_SCENES = [
    "frl_apartment_0-3",
    "frl_apartment_1-3",
    "apartment_0-3",
    "apartment_1-3",
    "office_0-3",
    "office_1-3",
    "room_0-3",
    "hotel_0-3",
    "frl_apartment_2-3",
]


def tar_url(domain: str, scene: str) -> str:
    name = f"{domain}-replica_gso-{scene}.tar"
    return f"{BASE_URL}/{domain}/replica_gso/{name}"


def download(url: str, out_path: Path, connections: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    elif shutil.which("wget"):
        cmd = ["wget", "--no-netrc", "-q", "-c", "-O", str(out_path), url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "--fail", "-o", str(out_path), url]
    else:
        raise RuntimeError("Need one of aria2c, curl, or wget to download Omnidata tars.")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def extract(tar_path: Path, dest: Path) -> None:
    domain, component, scene = tar_path.stem.split("-", 2)
    target = dest / domain / component / scene
    if target.is_dir() and any(target.iterdir()):
        print(f"extracted {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"extract {tar_path} -> {dest}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path("/mnt/klum/omnidata_replica_gso_multiscene"))
    parser.add_argument(
        "--compressed",
        type=Path,
        default=Path("/mnt/klum/omnidata_replica_gso_multiscene_compressed"),
    )
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--keep-compressed", action="store_true")
    parser.add_argument("--only-print", action="store_true")
    args = parser.parse_args()

    for scene in args.scenes:
        for domain in args.domains:
            url = tar_url(domain, scene)
            tar_path = args.compressed / f"{domain}-replica_gso-{scene}.tar"
            if args.only_print:
                print(url)
                continue
            download(url, tar_path, args.connections)
            extract(tar_path, args.dest)
            if not args.keep_compressed:
                tar_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
