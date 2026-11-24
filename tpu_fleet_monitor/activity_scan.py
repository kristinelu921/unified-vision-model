#!/usr/bin/env python3
"""Run one explicitly authorized, read-only TPU guest occupancy snapshot.

This module is deliberately separate from the fleet web service.  It is never
called by an HTTP endpoint, timer, or background poller.  Every invocation
requires an explicit confirmation flag.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import getpass
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from tpu_fleet_monitor.server import (
    DEFAULT_PROJECT_IDENTIFIER,
    MonitorConfiguration,
    command_error,
    collect_tpu_inventory,
    parse_resource_path,
    run_read_only_command,
    utc_now,
    validate_safe_identifier,
)


DEFAULT_ACTIVITY_PATH = Path(
    "/kmh-nfs-ssd-us-mount/staging/kristine/tpu-fleet-monitor/activity.json"
)
EXPLICIT_PERMISSION_PHRASE = "I_HAVE_EXPLICIT_READ_ONLY_PERMISSION"

# This command only reads process and device-holder metadata.  It contains no
# signal, file-write, cleanup, package-install, or lifecycle operation.
REMOTE_INSPECTION_COMMAND = r"""set -u
worker=$(hostname)
if ! command -v fuser >/dev/null 2>&1 || ! command -v base64 >/dev/null 2>&1; then
    printf 'TPU_ACTIVITY_UNKNOWN worker=%s reason=inspection_tools_unavailable\n' "$worker"
    exit 0
fi
if ! sudo -n true 2>/dev/null; then
    printf 'TPU_ACTIVITY_UNKNOWN worker=%s reason=read_only_sudo_unavailable\n' "$worker"
    exit 0
fi
declare -A seen=()
busy=0
holder_pids=''
for resource in /dev/vfio/*; do
    [[ -e "$resource" ]] || continue
    holders=$(sudo -n fuser "$resource" 2>/dev/null || true)
    holder_pids="$holder_pids $holders"
done
for pid in $(printf '%s\n' $holder_pids | sort -nu); do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    owner=$(ps -o user= -p "$pid" 2>/dev/null | xargs)
    args=$(ps -o args= -p "$pid" 2>/dev/null || true)
    [[ -n "$args" ]] || continue
    encoded=$(printf '%s' "$args" | base64 | tr -d '\n')
    printf 'TPU_ACTIVITY_RUN worker=%s pid=%s user=%s source=vfio command_b64=%s\n' \
        "$worker" "$pid" "${owner:-unknown}" "$encoded"
    seen[$pid]=1
    busy=1
done
while read -r pid owner command_name args; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ "${command_name,,}" == python* ]] || continue
    padded=" $args "
    if [[ "$padded" != *' train.py '* && "$padded" != *'/train.py '* ]]; then
        continue
    fi
    [[ -z "${seen[$pid]+present}" ]] || continue
    encoded=$(printf '%s' "$args" | base64 | tr -d '\n')
    printf 'TPU_ACTIVITY_RUN worker=%s pid=%s user=%s source=train_process command_b64=%s\n' \
        "$worker" "$pid" "${owner:-unknown}" "$encoded"
    seen[$pid]=1
    busy=1
done < <(ps -eo pid=,user=,comm=,args=)
if (( busy )); then
    printf 'TPU_ACTIVITY_BUSY worker=%s\n' "$worker"
elif [[ -e /tmp/libtpu_lockfile ]]; then
    printf 'TPU_ACTIVITY_UNKNOWN worker=%s reason=lock_without_observed_holder\n' "$worker"
else
    printf 'TPU_ACTIVITY_IDLE worker=%s\n' "$worker"
fi
exit 0"""

TAGGED_LINE_PATTERN = re.compile(r"^TPU_ACTIVITY_(RUN|BUSY|IDLE|UNKNOWN)\s+(.*)$")
FIELD_PATTERN = re.compile(r"([a-z0-9_]+)=([^\s]+)")


def validate_inspection_command(command: Sequence[str]) -> None:
    """Fail closed unless this is the exact read-only SSH inspection shape."""

    if len(command) < 12 or Path(command[0]).name != "ssh":
        raise ValueError("activity scanner requires the system SSH reader")
    required = {
        "-T",
        "-F",
        "/dev/null",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=no",
        "-oUserKnownHostsFile=/dev/null",
        "-oConnectionAttempts=1",
        "-oIdentitiesOnly=yes",
        "-i",
    }
    if not required.issubset(set(command)):
        raise ValueError("activity scanner is missing mandatory non-mutating SSH flags")
    if command[-2:] != ["--", REMOTE_INSPECTION_COMMAND]:
        raise ValueError("activity scanner remote command differs from the audited reader")
    if command.count("-i") != 1:
        raise ValueError("activity scanner must use exactly one existing identity file")
    identity_path = Path(command[command.index("-i") + 1])
    if not identity_path.is_absolute():
        raise ValueError("activity scanner identity path must be absolute")
    destination = command[-3]
    if "@" not in destination:
        raise ValueError("activity scanner SSH destination is malformed")
    username, address = destination.rsplit("@", 1)
    validate_safe_identifier(username, "SSH username")
    ipaddress.ip_address(address)


def build_inspection_command(
    ssh_executable: str,
    username: str,
    address: str,
    identity_file: Path,
) -> list[str]:
    """Build one exact direct-SSH reader with no key-upload mechanism."""

    validate_safe_identifier(username, "SSH username")
    ipaddress.ip_address(address)
    if not identity_file.is_absolute():
        raise ValueError("SSH identity path must be absolute")
    command = [
        ssh_executable,
        "-T",
        "-F",
        "/dev/null",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=no",
        "-oUserKnownHostsFile=/dev/null",
        "-oConnectTimeout=10",
        "-oConnectionAttempts=1",
        "-oIdentitiesOnly=yes",
        "-oServerAliveInterval=5",
        "-oServerAliveCountMax=2",
        "-i",
        str(identity_file),
        f"{username}@{address}",
        "--",
        REMOTE_INSPECTION_COMMAND,
    ]
    validate_inspection_command(command)
    return command


def attach_external_endpoints(
    configuration: MonitorConfiguration,
    inventory: list[dict[str, Any]],
) -> None:
    """Attach ephemeral external endpoints without publishing or saving them."""

    result = run_read_only_command(
        [
            configuration.gcloud_executable,
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={configuration.project_identifier}",
            "--zone=-",
            "--format=json(name,networkEndpoints)",
        ],
        90,
    )
    if result.return_code != 0:
        raise command_error(result, "read-only TPU endpoint inventory")
    payload = json.loads(result.standard_output or "[]")
    endpoint_map: dict[str, list[str]] = {}
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        _, name = parse_resource_path(str(item.get("name", "")), "nodes")
        addresses: list[str] = []
        endpoints = item.get("networkEndpoints")
        for endpoint in endpoints if isinstance(endpoints, list) else []:
            access = endpoint.get("accessConfig") if isinstance(endpoint, dict) else None
            address = access.get("externalIp") if isinstance(access, dict) else None
            if not isinstance(address, str):
                continue
            ipaddress.ip_address(address)
            addresses.append(address)
        endpoint_map[name] = addresses
    for resource in inventory:
        resource["scan_endpoints"] = endpoint_map.get(resource["name"], [])


def summarize_command(command: str) -> str:
    """Return a compact entrypoint description without interpreting the run."""

    module_match = re.search(r"(?:^|\s)-m\s+([A-Za-z0-9_.-]+)", command)
    if module_match:
        return module_match.group(1)
    script_match = re.search(r"(?:^|\s)([^\s]+\.py)(?:\s|$)", command)
    if script_match:
        return Path(script_match.group(1)).name
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    return Path(words[0]).name if words else "process"


def parse_inspection_output(output: str) -> dict[str, Any]:
    """Parse only scanner-tagged output and ignore SSH diagnostics."""

    workers: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        match = TAGGED_LINE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        kind, remainder = match.groups()
        fields = dict(FIELD_PATTERN.findall(remainder))
        worker = fields.get("worker", "unknown")
        worker_entry = workers.setdefault(
            worker, {"status": "unknown", "reason": None, "runs": 0}
        )
        if kind == "RUN":
            encoded_command = fields.get("command_b64", "")
            try:
                full_command = base64.b64decode(encoded_command, validate=True).decode(
                    "utf-8", errors="replace"
                )
            except (ValueError, base64.binascii.Error):
                full_command = ""
            full_command = full_command[:1200]
            runs.append(
                {
                    "worker": worker,
                    "pid": int(fields["pid"]) if fields.get("pid", "").isdigit() else None,
                    "user": fields.get("user", "unknown")[:80],
                    "source": fields.get("source", "unknown")[:40],
                    "summary": summarize_command(full_command),
                    "command": full_command,
                }
            )
            worker_entry["runs"] += 1
        elif kind == "BUSY":
            worker_entry["status"] = "busy"
        elif kind == "IDLE":
            worker_entry["status"] = "idle"
        elif kind == "UNKNOWN":
            worker_entry["status"] = "unknown"
            worker_entry["reason"] = fields.get("reason", "inspection_unknown")[:120]

    statuses = {entry["status"] for entry in workers.values()}
    if "busy" in statuses or runs:
        status = "busy"
    elif statuses and statuses == {"idle"}:
        status = "idle"
    else:
        status = "unknown"
    return {"status": status, "workers": workers, "runs": runs}


def inspect_tpu(
    resource: Mapping[str, Any],
    *,
    ssh_executable: str,
    username: str,
    identity_file: Path,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any]]:
    """Inspect one exact READY TPU without a shell or a mutation code path."""

    name = str(resource["name"])
    zone = str(resource["zone"])
    endpoints = resource.get("scan_endpoints")
    endpoints = endpoints if isinstance(endpoints, list) else []
    combined_workers: dict[str, dict[str, Any]] = {}
    combined_runs: list[dict[str, Any]] = []
    errors: list[str] = []
    endpoint_statuses: list[str] = []
    for worker_index, address in enumerate(endpoints):
        command = build_inspection_command(
            ssh_executable, username, address, identity_file
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            endpoint_result = parse_inspection_output(completed.stdout)
            if completed.returncode != 0 and not endpoint_result["workers"]:
                diagnostic = (completed.stderr or completed.stdout).strip().splitlines()
                errors.append(
                    f"worker-{worker_index}: "
                    f"{(diagnostic[-1] if diagnostic else f'SSH exited {completed.returncode}')[:240]}"
                )
                endpoint_result["status"] = "unknown"
        except subprocess.TimeoutExpired:
            endpoint_result = {"status": "unknown", "workers": {}, "runs": []}
            errors.append(
                f"worker-{worker_index}: inspection timed out after {timeout_seconds:.0f}s"
            )
        endpoint_statuses.append(endpoint_result["status"])
        for worker, worker_result in endpoint_result["workers"].items():
            combined_workers[worker] = worker_result
        combined_runs.extend(endpoint_result["runs"])

    if not endpoints:
        status = "unknown"
        errors.append("no external endpoint was available")
    elif "busy" in endpoint_statuses:
        status = "busy"
    elif endpoint_statuses and set(endpoint_statuses) == {"idle"}:
        status = "idle"
    else:
        status = "unknown"
    parsed = {
        "status": status,
        "workers": combined_workers,
        "runs": combined_runs,
    }
    if errors:
        parsed["error"] = "; ".join(errors[:4])
    parsed.update(
        {
            "zone": zone,
            "accelerator_type": resource.get("accelerator_type"),
            "create_time": resource.get("create_time"),
            "inspected_at": utc_now(),
        }
    )
    return name, parsed


def write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a local snapshot for the metadata-only web service."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    previous_umask = os.umask(0o077)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        os.umask(previous_umask)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT_IDENTIFIER)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTIVITY_PATH)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=75)
    parser.add_argument(
        "--explicit-permission",
        required=True,
        help=f"must equal {EXPLICIT_PERMISSION_PHRASE}",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.explicit_permission != EXPLICIT_PERMISSION_PHRASE:
        raise SystemExit("explicit per-scan permission confirmation is required")
    if not 1 <= arguments.max_workers <= 12:
        raise SystemExit("max-workers must be between 1 and 12")
    if not 10 <= arguments.timeout_seconds <= 180:
        raise SystemExit("timeout-seconds must be between 10 and 180")

    gcloud_executable = shutil.which("gcloud") or "/snap/bin/gcloud"
    ssh_executable = shutil.which("ssh") or "/usr/bin/ssh"
    ssh_username = getpass.getuser()
    identity_file = Path.home() / ".ssh/google_compute_engine"
    if not identity_file.is_file():
        raise SystemExit(f"existing SSH identity is unavailable: {identity_file}")
    configuration = MonitorConfiguration(
        project_identifier=arguments.project,
        gcloud_executable=gcloud_executable,
        fleet_poll_seconds=30,
        queue_poll_seconds=30,
    )
    inventory = collect_tpu_inventory(configuration)
    attach_external_endpoints(configuration, inventory)
    ready_resources = [resource for resource in inventory if resource["state"] == "READY"]
    print(
        f"One-time read-only activity scan: {len(ready_resources)} READY TPUs; "
        "no signals or mutations are available",
        flush=True,
    )

    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.max_workers
    ) as executor:
        futures = {
            executor.submit(
                inspect_tpu,
                resource,
                ssh_executable=ssh_executable,
                username=ssh_username,
                identity_file=identity_file,
                timeout_seconds=arguments.timeout_seconds,
            ): resource["name"]
            for resource in ready_resources
        }
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            results[name] = result
            completed_count += 1
            print(
                f"[{completed_count}/{len(ready_resources)}] {name}: {result['status']}",
                flush=True,
            )

    counts = {
        status: sum(result["status"] == status for result in results.values())
        for status in ("busy", "idle", "unknown")
    }
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "project_identifier": arguments.project,
        "automatic": False,
        "permission": "explicit one-time read-only guest inspection",
        "counts": counts,
        "tpus": results,
    }
    write_snapshot(arguments.output, payload)
    print(
        "Snapshot written: "
        f"busy={counts['busy']} idle={counts['idle']} unknown={counts['unknown']} "
        f"path={arguments.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
