#!/usr/bin/env python3
"""Strict identity checks for the locally staged InstaManip asset stack."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MANIFEST_SHA256 = "32ce1fb1f79086bb0cccc5eb6da50a0e843197ce836690db961ffe13ca487ea5"
EXPECTED_FILES = 20
EXPECTED_BYTES = 74_454_658_841


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("* ")
        rel = Path(name)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or rel.is_absolute() or ".." in rel.parts or name in rows):
            raise ValueError(f"invalid asset manifest row: {line!r}")
        rows[name] = digest
    if len(rows) != EXPECTED_FILES:
        raise ValueError(f"expected {EXPECTED_FILES} asset files, found {len(rows)}")
    return rows


def verify_assets(root: Path) -> dict[str, object]:
    root = Path(root)
    manifest = root / "RUNTIME_SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file() or sha256_file(manifest) != MANIFEST_SHA256:
        raise ValueError("InstaManip runtime manifest is absent, a symlink, or differs")
    expected = parse_manifest(manifest)
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != manifest.name}
    if actual != set(expected):
        raise ValueError(f"asset file set differs: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    total = 0
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"asset hash differs: {name}")
        total += path.stat().st_size
    if total != EXPECTED_BYTES:
        raise ValueError(f"expected {EXPECTED_BYTES} asset bytes, found {total}")
    return {"manifest_sha256": MANIFEST_SHA256, "files": len(expected), "bytes": total}


def _memory_snapshot(root: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "Shmem"}:
            values[key.lower() + "_bytes"] = int(raw.split()[0]) * 1024
    fs = os.statvfs(root)
    values["filesystem_available_bytes"] = fs.f_bavail * fs.f_frsize
    values["filesystem_total_bytes"] = fs.f_blocks * fs.f_frsize
    return values


def _release_asset_files(
    root: Path,
    expected: dict[str, str],
    expected_bytes: int,
    receipt_path: Path,
    *,
    require_dev_shm: bool,
) -> dict[str, object]:
    """Release only a previously authenticated, exact asset file set.

    The caller has already cryptographically verified the manifest and every
    file.  This second boundary rechecks the exact set, regular-file type,
    digest-keyed sizes, and total bytes before unlinking only those listed
    files.  The manifest itself is retained as durable provenance.
    """
    root = Path(root).resolve(strict=True)
    receipt_path = Path(receipt_path)
    if require_dev_shm and not str(root).startswith("/dev/shm/"):
        raise ValueError(f"refusing to release assets outside /dev/shm: {root}")
    manifest = root / "RUNTIME_SHA256SUMS"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("asset manifest changed before release")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest.name
    }
    if actual != set(expected):
        raise ValueError("asset file set changed before release")
    sizes: dict[str, int] = {}
    for name in expected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"asset became unsafe before release: {name}")
        sizes[name] = path.stat().st_size
    if sum(sizes.values()) != expected_bytes:
        raise ValueError("asset byte count changed before release")

    before = _memory_snapshot(root)
    for name in sorted(expected):
        (root / name).unlink()
    remaining = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest.name
    }
    if remaining:
        raise RuntimeError(f"authenticated asset release incomplete: {sorted(remaining)}")
    after = _memory_snapshot(root)
    receipt: dict[str, object] = {
        "schema": "instamanip_q50_authenticated_assets_released_v1",
        "status": "PASS",
        "stage": "after_tpu_runtime_mark_step_and_wait_device_ops",
        "manifest_sha256": MANIFEST_SHA256,
        "released_files": len(expected),
        "released_bytes": expected_bytes,
        "before": before,
        "after": after,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_name(receipt_path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, receipt_path)
    return receipt


def release_verified_assets(
    root: Path, verification: dict[str, object], receipt_path: Path
) -> dict[str, object]:
    """Free run-local tmpfs model files after all weights are on TPU."""
    root = Path(root)
    manifest = root / "RUNTIME_SHA256SUMS"
    if (
        verification != {
            "manifest_sha256": MANIFEST_SHA256,
            "files": EXPECTED_FILES,
            "bytes": EXPECTED_BYTES,
        }
        or sha256_file(manifest) != MANIFEST_SHA256
    ):
        raise ValueError("asset verification token differs before release")
    return _release_asset_files(
        root,
        parse_manifest(manifest),
        EXPECTED_BYTES,
        receipt_path,
        require_dev_shm=True,
    )
