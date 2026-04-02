"""
Stream .tar.gz members to Google Cloud Storage.

Tar is read sequentially (required by the format). Uploads run in parallel in a
thread pool so network can overlap the next read.

For multiple large archives, fastest is often one process per .tar.gz (each with
parallel uploads), e.g. three terminals or GNU parallel — not one process opening
all tars at once (unless you run them sequentially in a loop).
"""

from __future__ import annotations

import argparse
import glob
import os
import tarfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from google.cloud import storage

_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def _upload_blob(
    client: storage.Client,
    bucket_name: str,
    key: str,
    data: bytes,
    content_type: str = _DEFAULT_CONTENT_TYPE,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(key)
    # Larger chunks help resumable throughput on big objects.
    if blob.chunk_size is None or blob.chunk_size < 8 * 1024 * 1024:
        blob.chunk_size = 8 * 1024 * 1024
    blob.upload_from_string(data, content_type=content_type)


def upload_archive_to_bucket(
    archive_path: str,
    bucket_name: str,
    object_prefix: str,
    *,
    max_workers: int = 16,
    client: storage.Client | None = None,
) -> int:
    """
    Stream one .tar.gz to GCS. Returns number of files uploaded.

    Reads each member fully into memory before upload (needed for parallel
    uploads). Lower ``max_workers`` if members are very large (e.g. huge HDF5).
    """
    own_client = client is None
    if client is None:
        client = storage.Client()

    prefix = object_prefix.strip().strip("/")
    uploaded = 0

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                pending: set = set()
                for member in tar:
                    if not member.isfile():
                        continue
                    reader = tar.extractfile(member)
                    if reader is None:
                        continue
                    name = member.name.lstrip("./")
                    key = f"{prefix}/{name}" if prefix else name
                    data = reader.read()
                    fut = executor.submit(
                        _upload_blob, client, bucket_name, key, data, _DEFAULT_CONTENT_TYPE
                    )
                    pending.add(fut)
                    uploaded += 1
                    if len(pending) >= max_workers:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for f in done:
                            f.result()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
    finally:
        if own_client and hasattr(client, "close"):
            client.close()

    return uploaded


def main() -> None:
    p = argparse.ArgumentParser(description="Stream .tar.gz contents to GCS (parallel uploads).")
    p.add_argument(
        "archives",
        nargs="*",
        help="Paths to .tar.gz files (default: all *.tar.gz under D3_NORMAL_DIR)",
    )
    p.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "kmh-gcp-us-central2"))
    p.add_argument(
        "--prefix",
        default=os.environ.get("GCS_PREFIX", "kristine/lvm/d3_depth_2"),
        help="Object key prefix (no gs://bucket).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("D3_UPLOAD_WORKERS", "16")),
        help="Concurrent GCS uploads per archive (default 16). Lower if OOM on large members.",
    )
    p.add_argument(
        "--dir",
        default=os.environ.get("D3_NORMAL_DIR", "/mnt/zhhm/kristine/lvm/d3_normal"),
        help="If no archives listed, glob *.tar.gz here.",
    )
    p.add_argument(
        "--flat-prefix",
        action="store_true",
        help="When uploading multiple archives, do not add each archive name as a subfolder "
        "(default: use subfolder per .tar.gz to avoid key collisions).",
    )
    args = p.parse_args()

    archives = args.archives
    if not archives:
        archives = sorted(glob.glob(os.path.join(args.dir, "*.tar.gz")))
        if not archives:
            raise SystemExit(f"No .tar.gz found under {args.dir!r}; pass archive paths explicitly.")

    # One client shared across sequential archives in this process.
    client = storage.Client()
    try:
        for path in archives:
            stem = os.path.splitext(os.path.basename(path))[0]
            if len(archives) > 1 and not args.flat_prefix:
                prefix = f"{args.prefix.rstrip('/')}/{stem}"
            else:
                prefix = args.prefix
            print(f"Uploading {path!r} → gs://{args.bucket}/{prefix}/ …", flush=True)
            n = upload_archive_to_bucket(
                path,
                args.bucket,
                prefix,
                max_workers=args.workers,
                client=client,
            )
            print(f"Done: {n} objects from {path}", flush=True)
    finally:
        if hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
