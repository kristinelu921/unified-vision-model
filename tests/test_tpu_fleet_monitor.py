from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tpu_fleet_monitor.server import (
    CommandResult,
    FleetMonitorService,
    MonitorConfiguration,
    collect_queued_resources,
    collect_tpu_inventory,
    load_activity_snapshot,
    merge_activity_snapshot,
    validate_read_only_command,
)
from tpu_fleet_monitor.activity_scan import (
    REMOTE_INSPECTION_COMMAND,
    build_inspection_command,
    parse_inspection_output,
    validate_inspection_command,
)


def configuration(tmp_path: Path) -> MonitorConfiguration:
    return MonitorConfiguration(
        project_identifier="project-one",
        gcloud_executable="gcloud",
        fleet_poll_seconds=30,
        queue_poll_seconds=30,
    )


def test_command_allowlist_accepts_only_metadata_lists() -> None:
    validate_read_only_command(
        ["gcloud", "compute", "tpus", "tpu-vm", "list", "--zone=-"]
    )
    validate_read_only_command(
        ["gcloud", "compute", "tpus", "queued-resources", "list", "--zone=-"]
    )

    for forbidden in (
        ["gcloud", "compute", "tpus", "tpu-vm", "ssh", "someone-else"],
        ["gcloud", "compute", "tpus", "tpu-vm", "delete", "someone-else"],
        ["gcloud", "compute", "tpus", "queued-resources", "reset", "queue-one"],
    ):
        with pytest.raises(ValueError):
            validate_read_only_command(forbidden)


def test_project_inventory_accepts_non_kristine_metadata_without_guest_access(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []

    def runner(command: list[str], timeout: float) -> CommandResult:
        commands.append(command)
        return CommandResult(
            0,
            json.dumps(
                [
                    {
                        "name": "projects/project-one/locations/us-east5-b/nodes/team-member-tpu",
                        "acceleratorType": "v6e-16",
                        "state": "READY",
                        "health": "HEALTHY",
                        "schedulingConfig": {"spot": True},
                        "labels": {"owner": "teammate"},
                    }
                ]
            ),
            "",
        )

    resources = collect_tpu_inventory(configuration(tmp_path), runner)

    assert resources[0] == {
        "name": "team-member-tpu",
        "zone": "us-east5-b",
        "accelerator_type": "v6e-16",
        "chips": 16,
        "state": "READY",
        "health": "HEALTHY",
        "scheduling": "Spot",
        "owner": "teammate",
        "is_kristine": False,
        "create_time": None,
    }
    assert commands[0][1:5] == ["compute", "tpus", "tpu-vm", "list"]
    assert "--zone=-" in commands[0]
    assert not {"ssh", "create", "delete", "reset"}.intersection(commands[0])


def test_queue_parser_groups_flattened_nodes_and_keeps_safe_fields_only(
    tmp_path: Path,
) -> None:
    def runner(command: list[str], timeout: float) -> CommandResult:
        payload = []
        for node_id in ("node-a", "node-b"):
            payload.append(
                {
                    "name": "projects/project-one/locations/us-central1-b/queuedResources/queue-one",
                    "createTime": "2026-09-27T12:00:00Z",
                    "spot": {},
                    "state": {"state": "PROVISIONING"},
                    "tpu": {
                        "nodeSpec": {
                            "nodeId": node_id,
                            "node": {"acceleratorType": "v6e-8"},
                        }
                    },
                }
            )
        return CommandResult(0, json.dumps(payload), "")

    resources = collect_queued_resources(configuration(tmp_path), runner)

    assert len(resources) == 1
    assert resources[0]["state"] == "PROVISIONING"
    assert resources[0]["category"] == "pending"
    assert resources[0]["capacity_mode"] == "Spot"
    assert resources[0]["nodes"] == ["node-a", "node-b"]
    assert resources[0]["chips"] == 16
    serialized = json.dumps(resources[0])
    assert "metadata" not in serialized
    assert "serviceAccount" not in serialized
    assert "networkConfig" not in serialized


def test_snapshot_is_explicitly_read_only(tmp_path: Path) -> None:
    service = FleetMonitorService(configuration(tmp_path))
    snapshot = service.snapshot()

    assert snapshot["read_only"] is True
    assert snapshot["safety"] == {
        "guest_access": False,
        "automatic_guest_access": False,
        "manual_activity_snapshot": True,
        "mutation_controls": False,
        "data_scope": (
            "Cloud TPU control-plane metadata plus a saved, explicitly "
            "authorized one-time read-only occupancy snapshot"
        ),
    }


def test_manual_activity_command_has_no_mutation_path() -> None:
    command = build_inspection_command(
        "/usr/bin/ssh",
        "kristinelu",
        "203.0.113.10",
        Path("/home/kristinelu/.ssh/google_compute_engine"),
    )
    validate_inspection_command(command)

    assert "-oBatchMode=yes" in command
    assert "-oConnectionAttempts=1" in command
    assert "-oIdentitiesOnly=yes" in command
    assert "-oUserKnownHostsFile=/dev/null" in command
    lowered_remote = REMOTE_INSPECTION_COMMAND.lower()
    for forbidden in (
        "kill ",
        "rm ",
        "chmod",
        "chown",
        "systemctl",
        "reboot",
        "shutdown",
        "--claim",
    ):
        assert forbidden not in lowered_remote

    altered = command.copy()
    altered[-1] = "kill 123"
    with pytest.raises(ValueError):
        validate_inspection_command(altered)


def test_activity_output_parsing_and_identity_matched_merge(tmp_path: Path) -> None:
    encoded = base64.b64encode(
        b"python -m jflux2.analysis.capacity_ladder_v5 --out /tmp/result"
    ).decode()
    parsed = parse_inspection_output(
        "\n".join(
            (
                f"TPU_ACTIVITY_RUN worker=worker-0 pid=123 user=researcher source=vfio command_b64={encoded}",
                "TPU_ACTIVITY_BUSY worker=worker-0",
            )
        )
    )
    assert parsed["status"] == "busy"
    assert parsed["runs"][0]["summary"] == "jflux2.analysis.capacity_ladder_v5"

    path = tmp_path / "activity.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-09-27T20:00:00+00:00",
                "automatic": False,
                "permission": "explicit one-time read-only guest inspection",
                "tpus": {
                    "team-tpu": {
                        **parsed,
                        "zone": "us-east5-b",
                        "create_time": "2026-09-27T19:00:00Z",
                        "inspected_at": "2026-09-27T20:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_activity_snapshot(path)
    tpus = [
        {
            "name": "team-tpu",
            "zone": "us-east5-b",
            "create_time": "2026-09-27T19:00:00Z",
        },
        {"name": "new-tpu", "zone": "us-east5-b", "create_time": None},
    ]
    merge_activity_snapshot(tpus, snapshot)
    assert tpus[0]["activity"]["status"] == "busy"
    assert tpus[1]["activity"]["status"] == "not_scanned"


def test_standalone_page_and_launcher_preserve_the_safety_boundary() -> None:
    index = (REPOSITORY_ROOT / "tpu_fleet_monitor/index.html").read_text(
        encoding="utf-8"
    )
    launcher = (REPOSITORY_ROOT / "tpu_fleet_monitor/start.sh").read_text(
        encoding="utf-8"
    )

    assert "Strictly read-only" in index
    assert "No automatic SSH" in index
    assert "Observed run" in index
    assert "activity_scan" not in (
        REPOSITORY_ROOT / "tpu_fleet_monitor/server.py"
    ).read_text(encoding="utf-8")
    assert 'href="http://127.0.0.1:8782"' in index
    assert "TPU_FLEET_MONITOR_PORT:-8783" in launcher
    assert 'nohup setsid "$python_executable"' in launcher
    assert "127.0.0.1" in launcher
