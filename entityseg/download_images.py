"""
Download EntitySeg RGB images listed in ``url.json`` into a layout suitable for TFDS (e.g. ``entity_seg_semantic`` / ``entity_seg_instance``).

Each row has ``img_name`` (relative path like ``entity_01_11580/coco_….jpg``). Rows must include
a non-empty ``url_image``; some entries only have ``url_dataset`` (no direct URL) and are **skipped**
— obtain those images from the listed datasets separately.

Usage::

  python entityseg/download_images.py --out /mnt/klum/entityseg/images
  python entityseg/download_images.py --out /path/to/out --max-files 1000   # cap for testing
  python entityseg/download_images.py --out /path/to/out --start 9700  # index 9700 → end of list
  python entityseg/download_images.py --workers 16

Then::

  export ENTITYSEG_IMAGES_ROOT=/mnt/klum/entityseg/images
  tfds build entityseg/entity_seg_semantic.py --data_dir=/path/to/tfds_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_DATA_ROOT = "/mnt/klum/entityseg/data"


def _download_one(
    url: str,
    dst: str,
    *,
    timeout_s: float = 20.0,
    retries: int = 4,
) -> tuple[str, str | None]:
    """Returns (dst, error_message_or_none)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return dst, None

    tmp = dst + ".part"
    last_err: str | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "entityseg-download/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return dst, None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_err = repr(e)
            if os.path.isfile(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            time.sleep(min(2**attempt, 30))
    return dst, last_err


def main() -> int:
    p = argparse.ArgumentParser(description="Download EntitySeg images from url.json")
    p.add_argument(
        "--data-root",
        default=os.environ.get("ENTITYSEG_DATA_ROOT", DEFAULT_DATA_ROOT),
        help=f"Directory containing url.json (default: {DEFAULT_DATA_ROOT})",
    )
    p.add_argument(
        "--out",
        "-o",
        default=os.environ.get("ENTITYSEG_IMAGES_ROOT", ""),
        help="Output root (ENTITYSEG_IMAGES_ROOT). Required unless set in env.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Begin at task index N (0-based): use tasks[N:], i.e. skip the first N URLs, "
            "then run through the rest of the list. Does NOT stop at N. "
            "Use --max-files to cap how many are run after that."
        ),
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If >0, only this many tasks after --start (slice [:M] on the tail). 0 = no cap.",
    )
    p.add_argument("--workers", type=int, default=8, help="Parallel download threads.")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds).")
    args = p.parse_args()

    if not args.out:
        print("error: pass --out /path or set ENTITYSEG_IMAGES_ROOT", file=sys.stderr)
        return 2

    url_path = os.path.join(args.data_root, "url.json")
    if not os.path.isfile(url_path):
        print(f"error: missing {url_path}", file=sys.stderr)
        return 2

    with open(url_path, encoding="utf-8") as f:
        rows = json.load(f)

    tasks: list[tuple[str, str]] = []
    no_url = 0
    for row in rows:
        rel = row.get("img_name")
        if not rel:
            continue
        url = row.get("url_image") or ""
        if not isinstance(url, str) or not url.strip():
            no_url += 1
            continue
        url = url.strip()
        dst = os.path.join(args.out, rel)
        tasks.append((url, dst))

    n_queue = len(tasks)
    tasks = tasks[args.start :]
    if args.max_files > 0:
        tasks = tasks[: args.max_files]

    out_abs = os.path.abspath(args.out)
    already = sum(1 for _u, d in tasks if os.path.isfile(d) and os.path.getsize(d) > 0)
    print(
        f"out={out_abs}\n"
        f"url.json: {len(rows)} rows, {no_url} rows have no url_image (not queued), "
        f"{n_queue} downloadable URLs in full list; this run: {len(tasks)} tasks "
        f"(indices [{args.start}:{args.start + len(tasks)}) into that list)"
        + (f", --max-files={args.max_files}" if args.max_files else "")
        + f"\n  already on disk (non-empty file): {already}/{len(tasks)} → progress skip≈{already}\n"
        f"  need download attempt: {len(tasks) - already}"
        + (
            f"\n  (--start: first {args.start} URLs in the full list are not run at all — that is not counted as skip)"
            if args.start
            else ""
        ),
        flush=True,
    )

    ok = 0
    skip = 0
    fail = 0
    errors: list[str] = []
    prev_time = time.time()
    def job(url: str, dst: str) -> tuple[str, str | None, bool]:
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return dst, None, True  # skipped
        _, err = _download_one(url, dst, timeout_s=args.timeout)
        return dst, err, False

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(job, u, d): (u, d) for u, d in tasks}
        done = 0
        total = len(futs)
        for fut in as_completed(futs):
            url, dst = futs[fut]
            res_dst, err, skipped = fut.result()
            done += 1
            if skipped:
                skip += 1
            elif err is None:
                ok += 1
            else:
                fail += 1
                if len(errors) < 50:
                    errors.append(f"{res_dst}\n  url={url}\n  {err}")
            if done % 100 == 0 or done == total:
                print("time:", time.time() - prev_time)
                prev_time = time.time()
                print(f"progress {done}/{total}  ok={ok} skip={skip} fail={fail}", flush=True)

    print(f"done. ok={ok} skip={skip} fail={fail} total={len(tasks)}")
    if errors:
        print("sample errors:", file=sys.stderr)
        for e in errors[:20]:
            print(e, file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
