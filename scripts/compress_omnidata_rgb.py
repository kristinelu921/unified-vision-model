#!/usr/bin/env python3
"""Compress Omnidata RGB PNG samples to JPEG in place.

Each PNG is first encoded to a staging directory outside /mnt/klum. The source
PNG is deleted only after the staged JPEG has been written and sanity-checked.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import tempfile
import time
from pathlib import Path

from PIL import Image


IMAGE_SUFFIX = ".png"


def iter_rgb_pngs(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob(f"*{IMAGE_SUFFIX}"):
        parts = set(path.parts)
        if "_compressed" in parts or "_link_cache" in parts:
            continue
        if path.parent.name == "rgb":
            out.append(path)
    return sorted(out)


def compress_one(path: Path, *, root: Path, stage_root: Path, quality: int) -> tuple[str, int, int]:
    rel = path.relative_to(root)
    final = path.with_suffix(".jpg")
    stage = stage_root / rel.with_suffix(".jpg")
    stage_tmp = stage.with_suffix(".jpg.tmp")
    final_tmp = final.with_suffix(".jpg.tmp")

    if final.exists() and final.stat().st_size > 0:
        path.unlink(missing_ok=True)
        return ("already_done", 0, final.stat().st_size)

    stage.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")
        img.save(stage_tmp, "JPEG", quality=quality, optimize=False, progressive=False)
    stage_tmp.replace(stage)

    if stage.stat().st_size <= 0:
        raise RuntimeError(f"empty staged JPEG: {stage}")

    original_size = path.stat().st_size
    path.unlink()
    shutil.copy2(stage, final_tmp)
    final_tmp.replace(final)
    stage.unlink(missing_ok=True)
    return ("compressed", original_size, final.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/klum/omindata"))
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/kmh-nfs-ssd-us-mount/code/kristine/lvm/.omnidata_rgb_jpeg_stage"),
    )
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.stage_root.mkdir(parents=True, exist_ok=True)
    paths = iter_rgb_pngs(args.root)
    if args.limit:
        paths = paths[: args.limit]
    total = len(paths)
    print(f"rgb_pngs={total} quality={args.quality} workers={args.workers}", flush=True)

    started = time.time()
    counts: dict[str, int] = {}
    original_total = 0
    jpeg_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(compress_one, path, root=args.root, stage_root=args.stage_root, quality=args.quality)
            for path in paths
        ]
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status, original_size, jpeg_size = future.result()
            counts[status] = counts.get(status, 0) + 1
            original_total += original_size
            jpeg_total += jpeg_size
            if idx == 1 or idx % 1000 == 0 or idx == total:
                elapsed = max(time.time() - started, 1e-6)
                rate = idx / elapsed
                saved = original_total - jpeg_total
                print(
                    f"progress={idx}/{total} rate={rate:.2f}/s "
                    f"compressed={counts.get('compressed', 0)} "
                    f"already_done={counts.get('already_done', 0)} "
                    f"saved_gib={saved / 1024**3:.2f}",
                    flush=True,
                )

    try:
        next(args.stage_root.rglob("*"))
    except StopIteration:
        args.stage_root.rmdir()


if __name__ == "__main__":
    main()
