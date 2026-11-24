#!/usr/bin/env python3
"""Download COCONut-B RGB images referenced by a TSV URL list."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
import sys
import time
from pathlib import Path

import requests


def _read_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                filename, url = parts
            elif len(parts) >= 3:
                _idx, filename, url = parts[:3]
            else:
                continue
            if filename and url:
                rows.append((filename, url))
    return rows


def _candidate_urls(url: str) -> list[str]:
    urls = [url]
    if "/train2017/" in url:
        urls.append(url.replace("/train2017/", "/unlabeled2017/"))
    return urls


def _download_one(dest_dir: Path, row: tuple[str, str], timeout: float, retries: int) -> tuple[str, str]:
    filename, url = row
    out = dest_dir / filename
    if out.exists() and out.stat().st_size > 0:
        return "skipped", filename

    tmp = out.with_suffix(out.suffix + ".tmp")
    last_error = "unknown"
    for candidate_url in _candidate_urls(url):
        for attempt in range(retries + 1):
            try:
                with requests.get(candidate_url, stream=True, timeout=timeout) as resp:
                    if resp.status_code == 404:
                        last_error = f"404 Client Error: Not Found for url: {candidate_url}"
                        break
                    resp.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            if chunk:
                                f.write(chunk)
                if tmp.stat().st_size == 0:
                    raise RuntimeError("downloaded empty file")
                os.replace(tmp, out)
                return "downloaded", filename
            except Exception as exc:
                last_error = str(exc)
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
                if attempt >= retries:
                    break
                time.sleep(min(30.0, 2.0 * (attempt + 1)))
    return "failed", f"{filename}\t{url}\t{last_error}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--fail-log", type=Path, required=True)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    args.fail_log.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.tsv)
    total = len(rows)
    print(f"rows={total} dest={args.dest} workers={args.workers}", flush=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    start = time.time()
    with args.fail_log.open("a", encoding="utf-8") as fail_f:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(_download_one, args.dest, row, args.timeout, args.retries)
                for row in rows
            ]
            for i, fut in enumerate(futures.as_completed(futs), start=1):
                status, detail = fut.result()
                counts[status] += 1
                if status == "failed":
                    fail_f.write(detail + "\n")
                    fail_f.flush()
                if i == 1 or i % 500 == 0 or i == total:
                    elapsed = max(time.time() - start, 1e-6)
                    rate = i / elapsed
                    print(
                        "done={}/{} downloaded={} skipped={} failed={} rate={:.2f}/s".format(
                            i,
                            total,
                            counts["downloaded"],
                            counts["skipped"],
                            counts["failed"],
                            rate,
                        ),
                        flush=True,
                    )

    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
