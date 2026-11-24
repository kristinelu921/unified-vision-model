#!/usr/bin/env python3
"""Serve a project-wide, metadata-only Cloud TPU fleet and queue dashboard."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as datetime_module
import json
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_PROJECT_IDENTIFIER = "he-vision-group"
DEFAULT_PORT = 8783
DEFAULT_ACTIVITY_SNAPSHOT_PATH = Path(
    "/kmh-nfs-ssd-us-mount/staging/kristine/tpu-fleet-monitor/activity.json"
)
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
READ_ONLY_GCLOUD_PREFIXES = (
    ("compute", "tpus", "tpu-vm", "list"),
    ("compute", "tpus", "queued-resources", "list"),
)
FORBIDDEN_GCLOUD_ACTIONS = frozenset(
    {
        "create",
        "delete",
        "reset",
        "reimage",
        "restart",
        "scp",
        "ssh",
        "start",
        "stop",
        "update",
    }
)
PENDING_QUEUE_STATES = frozenset(
    {"ACCEPTED", "CREATING", "PROVISIONING", "QUEUED", "WAITING_FOR_RESOURCES"}
)
PAUSED_QUEUE_STATES = frozenset(
    {"CANCELLING", "DELETING", "SUSPENDED", "SUSPENDING"}
)


@dataclass(frozen=True)
class CommandResult:
    """Capture one bounded command result."""

    return_code: int
    standard_output: str
    standard_error: str


CommandRunner = Callable[[Sequence[str], float], CommandResult]


@dataclass(frozen=True)
class MonitorConfiguration:
    """Runtime settings for the standalone fleet monitor."""

    project_identifier: str
    gcloud_executable: str
    fleet_poll_seconds: float
    queue_poll_seconds: float
    activity_snapshot_path: Path | None = None


@dataclass
class SourceHeartbeat:
    """Track the last result from an independently polled source."""

    interval_seconds: float
    last_attempt: str | None = None
    last_success: str | None = None
    error: str | None = None


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def validate_safe_identifier(value: str, label: str) -> None:
    """Reject unsafe project and resource identifiers."""

    if not SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")


def validate_read_only_command(command: Sequence[str]) -> None:
    """Fail closed unless a command is one of the two metadata-only list calls."""

    if len(command) < 5 or Path(command[0]).name not in {"gcloud", "gcloud.cmd"}:
        raise ValueError("fleet monitor only permits gcloud list commands")
    command_words = tuple(str(part).lower() for part in command[1:])
    if not any(command_words[: len(prefix)] == prefix for prefix in READ_ONLY_GCLOUD_PREFIXES):
        raise ValueError("gcloud command is outside the read-only fleet allowlist")
    if FORBIDDEN_GCLOUD_ACTIONS.intersection(command_words):
        raise ValueError("mutating or guest-access gcloud action is forbidden")


def run_read_only_command(
    command: Sequence[str], timeout_seconds: float
) -> CommandResult:
    """Run one allowlisted metadata query without a shell."""

    validate_read_only_command(command)
    completed_process = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandResult(
        return_code=completed_process.returncode,
        standard_output=completed_process.stdout,
        standard_error=completed_process.stderr,
    )


def command_error(result: CommandResult, operation: str) -> RuntimeError:
    """Return a bounded diagnostic suitable for the local dashboard."""

    detail = (result.standard_error or result.standard_output).strip()
    if len(detail) > 800:
        detail = f"{detail[:797]}..."
    return RuntimeError(f"{operation} failed: {detail or 'no diagnostic output'}")


def parse_resource_path(full_name: str, collection: str) -> tuple[str, str]:
    """Extract a zone and terminal name from a canonical TPU API resource path."""

    parts = full_name.strip("/").split("/")
    if (
        len(parts) != 6
        or parts[0] != "projects"
        or parts[2] != "locations"
        or parts[4] != collection
    ):
        raise ValueError(f"unexpected TPU resource path: {full_name!r}")
    zone, name = parts[3], parts[5]
    validate_safe_identifier(zone, "TPU zone")
    validate_safe_identifier(name, "TPU resource name")
    return zone, name


def chip_count(accelerator_type: str) -> int:
    """Read the final chip-count component of a TPU accelerator type."""

    try:
        value = int(accelerator_type.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0
    return max(0, value)


def scheduling_mode(serialized_resource: Mapping[str, Any]) -> str:
    """Normalize Spot, preemptible, and on-demand scheduling metadata."""

    scheduling = serialized_resource.get("schedulingConfig") or {}
    if scheduling.get("spot"):
        return "Spot"
    if scheduling.get("preemptible"):
        return "Preemptible"
    return "On-demand"


def owner_label(name: str, labels: Mapping[str, Any] | None = None) -> str:
    """Expose an explicit owner label without guessing identities from arbitrary names."""

    labels = labels or {}
    explicit_owner = labels.get("owner")
    if isinstance(explicit_owner, str) and explicit_owner:
        return explicit_owner
    if "kristine" in name.lower():
        return "kristine"
    return "project"


def collect_tpu_inventory(
    configuration: MonitorConfiguration,
    command_runner: CommandRunner = run_read_only_command,
) -> list[dict[str, Any]]:
    """List every TPU VM using metadata fields only; never SSH or inspect a guest."""

    validate_safe_identifier(configuration.project_identifier, "project identifier")
    result = command_runner(
        [
            configuration.gcloud_executable,
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            f"--project={configuration.project_identifier}",
            "--zone=-",
            "--format=json(name,acceleratorType,state,health,schedulingConfig,labels,createTime)",
        ],
        90,
    )
    if result.return_code != 0:
        raise command_error(result, "project-wide TPU metadata inventory")
    payload = json.loads(result.standard_output or "[]")
    if not isinstance(payload, list):
        raise ValueError("TPU VM list did not return a JSON list")

    resources: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        zone, name = parse_resource_path(str(item.get("name", "")), "nodes")
        accelerator_type = str(item.get("acceleratorType") or "unknown")
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        resources.append(
            {
                "name": name,
                "zone": zone,
                "accelerator_type": accelerator_type,
                "chips": chip_count(accelerator_type),
                "state": str(item.get("state") or "UNKNOWN"),
                "health": str(item.get("health") or "UNKNOWN"),
                "scheduling": scheduling_mode(item),
                "owner": owner_label(name, labels),
                "is_kristine": "kristine" in name.lower(),
                "create_time": item.get("createTime"),
            }
        )
    return sorted(resources, key=lambda resource: (resource["zone"], resource["name"]))


def queue_category(state: str) -> str:
    """Group verbose Queued Resource states for summary and filtering."""

    normalized = state.upper()
    if normalized in PENDING_QUEUE_STATES:
        return "pending"
    if normalized == "ACTIVE":
        return "active"
    if normalized == "FAILED":
        return "failed"
    if normalized in PAUSED_QUEUE_STATES:
        return "paused"
    return "other"


def queue_capacity_mode(item: Mapping[str, Any]) -> str:
    """Identify the scheduling request without exposing node metadata."""

    if "spot" in item:
        return "Spot"
    if "guaranteed" in item:
        return "Guaranteed"
    if "bestEffort" in item:
        return "Best effort"
    return "Unspecified"


def collect_queued_resources(
    configuration: MonitorConfiguration,
    command_runner: CommandRunner = run_read_only_command,
) -> list[dict[str, Any]]:
    """List Queued Resources across all zones with a strict safe-field projection."""

    validate_safe_identifier(configuration.project_identifier, "project identifier")
    result = command_runner(
        [
            configuration.gcloud_executable,
            "compute",
            "tpus",
            "queued-resources",
            "list",
            f"--project={configuration.project_identifier}",
            "--zone=-",
            "--flatten=tpu.nodeSpec[]",
            "--format=json(name,createTime,state,spot,guaranteed,bestEffort,tpu.nodeSpec.nodeId,tpu.nodeSpec.node.acceleratorType)",
        ],
        120,
    )
    if result.return_code != 0:
        raise command_error(result, "project-wide TPU queued-resource inventory")
    payload = json.loads(result.standard_output or "[]")
    if not isinstance(payload, list):
        raise ValueError("Queued Resource list did not return a JSON list")

    grouped: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        full_name = str(item.get("name", ""))
        zone, name = parse_resource_path(full_name, "queuedResources")
        state_detail = item.get("state") if isinstance(item.get("state"), dict) else {}
        state = str(state_detail.get("state") or "UNKNOWN")
        failed_data = (
            state_detail.get("failedData")
            if isinstance(state_detail.get("failedData"), dict)
            else {}
        )
        error_detail = (
            failed_data.get("error")
            if isinstance(failed_data.get("error"), dict)
            else {}
        )
        spec = item.get("tpu", {}).get("nodeSpec", {}) if isinstance(item.get("tpu"), dict) else {}
        if not isinstance(spec, dict):
            spec = {}
        node = spec.get("node") if isinstance(spec.get("node"), dict) else {}
        node_id = str(spec.get("nodeId") or "")
        accelerator_type = str(node.get("acceleratorType") or "unknown")

        entry = grouped.setdefault(
            full_name,
            {
                "name": name,
                "zone": zone,
                "state": state,
                "category": queue_category(state),
                "state_initiator": state_detail.get("stateInitiator"),
                "capacity_mode": queue_capacity_mode(item),
                "create_time": item.get("createTime"),
                "nodes": [],
                "accelerator_types": [],
                "chips": 0,
                "owner": owner_label(f"{name}-{node_id}"),
                "is_kristine": "kristine" in f"{name}-{node_id}".lower(),
                "error_code": error_detail.get("code"),
                "error_message": str(error_detail.get("message") or "")[:800] or None,
            },
        )
        if node_id and node_id not in entry["nodes"]:
            entry["nodes"].append(node_id)
            entry["chips"] += chip_count(accelerator_type)
        if accelerator_type not in entry["accelerator_types"]:
            entry["accelerator_types"].append(accelerator_type)
        if "kristine" in node_id.lower():
            entry["is_kristine"] = True
            entry["owner"] = "kristine"

    resources = list(grouped.values())
    for resource in resources:
        resource["nodes"].sort()
        resource["accelerator_types"].sort()
    return sorted(
        resources,
        key=lambda resource: (resource.get("create_time") or "", resource["name"]),
        reverse=True,
    )


def load_activity_snapshot(path: Path | None) -> dict[str, Any]:
    """Read a locally saved, explicitly authorized one-time activity snapshot."""

    empty = {
        "generated_at": None,
        "automatic": False,
        "permission": None,
        "tpus": {},
        "error": None,
    }
    if path is None or not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported activity snapshot schema")
        if payload.get("automatic") is not False:
            raise ValueError("activity snapshot must declare automatic=false")
        resources = payload.get("tpus")
        if not isinstance(resources, dict):
            raise ValueError("activity snapshot TPU map is invalid")
        return {
            "generated_at": payload.get("generated_at"),
            "automatic": False,
            "permission": payload.get("permission"),
            "tpus": resources,
            "error": None,
        }
    except Exception as error:
        empty["error"] = f"activity snapshot unavailable: {str(error)[:600]}"
        return empty


def merge_activity_snapshot(
    tpus: list[dict[str, Any]], snapshot: Mapping[str, Any]
) -> None:
    """Attach only identity-matched activity records to current TPU resources."""

    activity_resources = snapshot.get("tpus")
    if not isinstance(activity_resources, dict):
        activity_resources = {}
    for resource in tpus:
        activity = activity_resources.get(resource["name"])
        identity_matches = (
            isinstance(activity, dict)
            and activity.get("zone") == resource["zone"]
            and activity.get("create_time") == resource.get("create_time")
        )
        if identity_matches:
            resource["activity"] = copy.deepcopy(activity)
        else:
            resource["activity"] = {
                "status": "not_scanned",
                "inspected_at": None,
                "workers": {},
                "runs": [],
            }


class FleetMonitorService:
    """Poll fleet metadata and queue metadata on independent read-only loops."""

    def __init__(
        self,
        configuration: MonitorConfiguration,
        command_runner: CommandRunner = run_read_only_command,
    ) -> None:
        self.configuration = configuration
        self.command_runner = command_runner
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.refresh_events = {"fleet": threading.Event(), "queue": threading.Event()}
        self.heartbeats = {
            "fleet": SourceHeartbeat(configuration.fleet_poll_seconds),
            "queue": SourceHeartbeat(configuration.queue_poll_seconds),
        }
        self.tpus: list[dict[str, Any]] = []
        self.queued_resources: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        """Start the two metadata pollers."""

        for source, interval, refresh in (
            ("fleet", self.configuration.fleet_poll_seconds, self.refresh_fleet),
            ("queue", self.configuration.queue_poll_seconds, self.refresh_queue),
        ):
            thread = threading.Thread(
                target=self._poll_loop,
                args=(source, interval, refresh),
                daemon=True,
                name=f"tpu-fleet-monitor-{source}",
            )
            thread.start()
            self.threads.append(thread)

    def stop(self) -> None:
        """Stop pollers without touching any cloud resource."""

        self.stop_event.set()
        for event in self.refresh_events.values():
            event.set()
        for thread in self.threads:
            thread.join(timeout=5)

    def _poll_loop(
        self,
        source: str,
        interval_seconds: float,
        refresh: Callable[[], None],
    ) -> None:
        event = self.refresh_events[source]
        while not self.stop_event.is_set():
            refresh()
            event.wait(interval_seconds)
            event.clear()

    def _record_attempt(self, source: str) -> None:
        with self.lock:
            self.heartbeats[source].last_attempt = utc_now()

    def _record_success(self, source: str) -> None:
        with self.lock:
            heartbeat = self.heartbeats[source]
            heartbeat.last_success = utc_now()
            heartbeat.error = None

    def _record_error(self, source: str, error: Exception) -> None:
        with self.lock:
            self.heartbeats[source].error = str(error)[:800]

    def refresh_fleet(self) -> None:
        self._record_attempt("fleet")
        try:
            resources = collect_tpu_inventory(self.configuration, self.command_runner)
            with self.lock:
                self.tpus = resources
            self._record_success("fleet")
        except Exception as error:
            self._record_error("fleet", error)

    def refresh_queue(self) -> None:
        self._record_attempt("queue")
        try:
            resources = collect_queued_resources(self.configuration, self.command_runner)
            with self.lock:
                self.queued_resources = resources
            self._record_success("queue")
        except Exception as error:
            self._record_error("queue", error)

    def request_refresh(self) -> None:
        """Wake both metadata pollers."""

        for event in self.refresh_events.values():
            event.set()

    def snapshot(self) -> dict[str, Any]:
        """Return a coherent copy of the latest project metadata."""

        with self.lock:
            tpus = copy.deepcopy(self.tpus)
            queued_resources = copy.deepcopy(self.queued_resources)
            sources = {
                name: dataclasses.asdict(heartbeat)
                for name, heartbeat in self.heartbeats.items()
            }
        activity_snapshot = load_activity_snapshot(
            self.configuration.activity_snapshot_path
        )
        merge_activity_snapshot(tpus, activity_snapshot)
        sources["activity"] = {
            "interval_seconds": None,
            "last_attempt": activity_snapshot["generated_at"],
            "last_success": (
                activity_snapshot["generated_at"]
                if not activity_snapshot["error"]
                else None
            ),
            "error": activity_snapshot["error"],
            "manual": True,
        }
        ready_tpus = [resource for resource in tpus if resource["state"] == "READY"]
        activity_counts = {
            status: sum(
                resource["activity"]["status"] == status for resource in tpus
            )
            for status in ("busy", "idle", "unknown", "not_scanned")
        }
        queue_counts = {
            category: sum(resource["category"] == category for resource in queued_resources)
            for category in ("pending", "active", "failed", "paused", "other")
        }
        return {
            "generated_at": utc_now(),
            "project_identifier": self.configuration.project_identifier,
            "read_only": True,
            "safety": {
                "guest_access": False,
                "automatic_guest_access": False,
                "manual_activity_snapshot": True,
                "mutation_controls": False,
                "data_scope": (
                    "Cloud TPU control-plane metadata plus a saved, explicitly "
                    "authorized one-time read-only occupancy snapshot"
                ),
            },
            "sources": sources,
            "summary": {
                "total_tpus": len(tpus),
                "ready_tpus": len(ready_tpus),
                "ready_chips": sum(resource["chips"] for resource in ready_tpus),
                "zones": len({resource["zone"] for resource in tpus}),
                "unhealthy": sum(
                    resource["health"] not in {"HEALTHY", "UNKNOWN"} for resource in tpus
                ),
                "kristine_tpus": sum(resource["is_kristine"] for resource in tpus),
                **{f"activity_{key}": value for key, value in activity_counts.items()},
                **{f"queue_{key}": value for key, value in queue_counts.items()},
            },
            "tpus": tpus,
            "queued_resources": queued_resources,
        }


def build_request_handler(
    service: FleetMonitorService, index_path: Path
) -> type[BaseHTTPRequestHandler]:
    """Bind the cached service to a minimal local HTTP API."""

    index_source = index_path.read_bytes()

    class FleetRequestHandler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: HTTPStatus) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(index_source, "text/html; charset=utf-8", HTTPStatus.OK)
                return
            if path == "/api/status":
                payload = json.dumps(service.snapshot(), sort_keys=True).encode("utf-8")
                self._send(payload, "application/json; charset=utf-8", HTTPStatus.OK)
                return
            self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/refresh":
                self._send(b'{"error":"not found"}', "application/json", HTTPStatus.NOT_FOUND)
                return
            service.request_refresh()
            self._send(
                b'{"status":"read-only refresh requested"}',
                "application/json",
                HTTPStatus.ACCEPTED,
            )

        def log_message(self, format_string: str, *arguments: Any) -> None:
            """Keep the local log focused on poll failures."""

    return FleetRequestHandler


def parse_arguments() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--project", default=DEFAULT_PROJECT_IDENTIFIER)
    parser.add_argument("--fleet-poll-seconds", type=float, default=30)
    parser.add_argument("--queue-poll-seconds", type=float, default=30)
    parser.add_argument(
        "--activity-snapshot",
        type=Path,
        default=DEFAULT_ACTIVITY_SNAPSHOT_PATH,
    )
    parser.set_defaults(index_path=repository_root / "tpu_fleet_monitor/index.html")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    validate_safe_identifier(arguments.project, "project identifier")
    if arguments.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("fleet monitor must bind to a loopback address")
    gcloud_executable = shutil.which("gcloud") or "/snap/bin/gcloud"
    configuration = MonitorConfiguration(
        project_identifier=arguments.project,
        gcloud_executable=gcloud_executable,
        fleet_poll_seconds=arguments.fleet_poll_seconds,
        queue_poll_seconds=arguments.queue_poll_seconds,
        activity_snapshot_path=arguments.activity_snapshot,
    )
    service = FleetMonitorService(configuration)
    handler = build_request_handler(service, arguments.index_path)
    server = ThreadingHTTPServer((arguments.host, arguments.port), handler)
    service.start()
    print(f"TPU fleet monitor: http://{arguments.host}:{arguments.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        service.stop()
        server.server_close()


if __name__ == "__main__":
    main()
