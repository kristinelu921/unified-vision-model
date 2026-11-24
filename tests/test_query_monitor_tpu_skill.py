import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIRECTORY = REPOSITORY_ROOT / ".agents" / "skills" / "query-monitor-gcp-tpus"
SWEEP_SCRIPT = SKILL_DIRECTORY / "scripts" / "sweep_tpus.sh"
OCCUPANCY_SCRIPT = SKILL_DIRECTORY / "scripts" / "check_tpu_occupancy.sh"
QUERY_SCRIPT = REPOSITORY_ROOT / "tpu_scripts" / "query_tpus.sh"
DENYLISTED_TPU_IDENTIFIERS = (
    "kmh-tpuvm-v4-8-2",
    "kmh-tpuvm-v4-8-3",
    "kmh-tpuvm-v4-8-4",
    "kmh-tpuvm-v4-8-6",
    "kmh-tpuvm-v4-8-8",
)


class QueryMonitorTpuScriptsTest(unittest.TestCase):
    def test_foreign_holder_kill_guard_is_removed(self) -> None:
        """Keep no tool whose purpose is killing another person's TPU job."""
        self.assertFalse(
            (REPOSITORY_ROOT / "tpu_scripts" / "guard_relaunching_foreign_vfio_holder.sh").exists()
        )

    def test_occupancy_check_supports_internal_tpu_ips(self) -> None:
        """Allow exact-target occupancy checks over the proven internal route."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("ELF_TPU_USE_INTERNAL_IP", occupancy_source)
        self.assertIn("ELF_TPU_USE_INTERNAL_IP:-1", occupancy_source)
        self.assertIn("gcloud_ssh_flags+=(--internal-ip)", occupancy_source)

    def run_worker_probe_locally(
        self, temporary_directory: Path, remove_stale_lock: str, lock_exists: bool, lock_held: bool
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        """Run the helper's worker probe against fake devices, sudo, fuser, and ps."""
        match = re.search(
            r"read -r -d '' worker_probe <<'REMOTE_COMMAND' \|\| true\n(.*?)\nREMOTE_COMMAND",
            OCCUPANCY_SCRIPT.read_text(encoding="utf-8"),
            re.DOTALL,
        )
        self.assertIsNotNone(match)
        lock_path = temporary_directory / "libtpu_lockfile"
        if lock_exists:
            lock_path.touch()
        device_directory = temporary_directory / "vfio"
        device_directory.mkdir()
        (device_directory / "0").touch()
        probe = match.group(1).replace("__REMOVE_STALE_LOCK__", remove_stale_lock)
        probe = probe.replace("/dev/vfio", str(device_directory))
        probe = probe.replace("/dev/accel", str(temporary_directory / "absent-accel"))
        probe = probe.replace("/tmp/libtpu_lockfile", str(lock_path))
        binary_directory = temporary_directory / "bin"
        binary_directory.mkdir()
        fake_sudo = binary_directory / "sudo"
        fake_sudo.write_text(
            """#!/bin/bash
set -eu
[[ "$1" != -n ]] || shift
case "$1" in
 true) exit 0 ;;
 fuser)
   if [[ "$FAKE_LOCK_HELD" == 1 && "$2" == */libtpu_lockfile ]]; then
     printf '777'
     exit 0
   fi
   exit 1 ;;
 stat) printf 'owner=someone uid=2019 mode=600 size=0\\n' ;;
 rm) shift; exec rm "$@" ;;
 *) printf 'UNEXPECTED_MUTATION %s\\n' "$*" >&2; exit 99 ;;
esac
""",
            encoding="utf-8",
        )
        fake_process_listing = binary_directory / "ps"
        fake_process_listing.write_text(
            """#!/bin/bash
case "$*" in
 *-eo*) : ;;
 *user=*) printf 'someone\\n' ;;
 *args=*) printf 'python worker.py\\n' ;;
esac
""",
            encoding="utf-8",
        )
        fake_sudo.chmod(0o755)
        fake_process_listing.chmod(0o755)
        environment = dict(
            os.environ,
            PATH=f"{binary_directory}:{os.environ['PATH']}",
            FAKE_LOCK_HELD="1" if lock_held else "0",
        )
        result = subprocess.run(
            ["bash", "-c", probe],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertNotIn("UNEXPECTED_MUTATION", result.stderr)
        return result, lock_path

    def test_worker_probe_removes_only_a_holderless_lock(self) -> None:
        """Report, then remove, a holderless lock; never remove a held one."""
        cases = (
            ("0", False, 3, "STALE_LOCK worker=", True),
            ("1", False, 0, "STALE_LOCK_REMOVED worker=", False),
            ("0", True, 2, "FOREIGN worker=", True),
            ("1", True, 2, "OCCUPIED worker=", True),
        )
        for remove_stale_lock, lock_held, expected_status, expected_marker, lock_remains in cases:
            with self.subTest(remove_stale_lock=remove_stale_lock, lock_held=lock_held):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    result, lock_path = self.run_worker_probe_locally(
                        Path(temporary_directory_name), remove_stale_lock, True, lock_held
                    )
                    self.assertEqual(result.returncode, expected_status, result.stderr)
                    self.assertIn(expected_marker, result.stdout)
                    self.assertEqual(lock_path.exists(), lock_remains)

    def test_worker_probe_reports_available_without_a_lock(self) -> None:
        """Report an idle worker with no lockfile as available in both phases."""
        for remove_stale_lock in ("0", "1"):
            with self.subTest(remove_stale_lock=remove_stale_lock):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    result, _ = self.run_worker_probe_locally(
                        Path(temporary_directory_name), remove_stale_lock, False, False
                    )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("AVAILABLE worker=", result.stdout)

    def make_fake_environment(
        self, temporary_directory: Path, mode: str
    ) -> tuple[dict[str, str], Path]:
        """Create a fake gcloud executable and return its environment and log."""
        binary_directory = temporary_directory / "bin"
        binary_directory.mkdir()
        command_log = temporary_directory / "gcloud.log"
        fake_gcloud = binary_directory / "gcloud"
        fake_gcloud.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_GCLOUD_LOG}"
arguments=$*
if [[ "${arguments}" == *"tpu-vm create"* && "${FAKE_GCLOUD_MODE}" == 'reservation' ]]; then
    printf 'Reservation not found\n' >&2
    exit 1
elif [[ "${arguments}" == *"tpu-vm create"* && "${FAKE_GCLOUD_MODE}" == 'tenant' ]]; then
    printf 'Failed to perform tenant project creation\n' >&2
    exit 1
elif [[ "${arguments}" == *"asset search-all-resources"* ]]; then
    case "${FAKE_GCLOUD_MODE}" in
        sweep) printf '%s\\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/ready-kristine' \
            'us-central1-b,//tpu.googleapis.com/projects/p/locations/us-central1-b/nodes/pending-kristine' ;;
        unauthorized) printf '%s\\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/alice-tpu' ;;
    esac
elif [[ "${arguments}" == *"tpu-vm describe"* ]]; then
    if [[ "${arguments}" == *"pending-kristine"* ]]; then
        printf 'CREATING\\n'
    else
        printf 'READY\\n'
    fi
elif [[ "${arguments}" == *"tpu-vm ssh"* ]]; then
    lock_removed_marker="${FAKE_GCLOUD_LOG}.lock-removed"
    if [[ "${FAKE_GCLOUD_MODE}" == 'occupied' ]]; then
        printf 'OWN worker=worker-0 pid=42 user=kristinelu\\n'
        printf 'OCCUPIED worker=worker-0\\n'
        exit 2
    elif [[ "${FAKE_GCLOUD_MODE}" == 'foreign' ]]; then
        printf 'FOREIGN worker=worker-0 pid=42 user=other\\n'
        printf 'OCCUPIED worker=worker-0\\n'
        exit 2
    elif [[ "${FAKE_GCLOUD_MODE}" == 'stale_lock' && ! -e "${lock_removed_marker}" ]]; then
        if [[ "${arguments}" == *"remove_stale_lock=1"* ]]; then
            touch "${lock_removed_marker}"
            printf 'STALE_LOCK_REMOVED worker=worker-0\\n'
            printf 'AVAILABLE worker=worker-0\\n'
        else
            printf 'STALE_LOCK worker=worker-0 reason=stale_libtpu_lock\\n'
            exit 3
        fi
    else
        printf 'AVAILABLE worker=worker-0\\n'
    fi
fi
""",
            encoding="utf-8",
        )
        fake_gcloud.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{binary_directory}:{environment['PATH']}",
                "FAKE_GCLOUD_LOG": str(command_log),
                "FAKE_GCLOUD_MODE": mode,
            }
        )
        return environment, command_log

    def run_script(
        self, script_path: Path, arguments: list[str], environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        """Run a skill script against the fake gcloud executable."""
        return subprocess.run(
            ["bash", str(script_path), *arguments],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_sweep_returns_only_current_ready_kristine_identifiers(self) -> None:
        """Return only exact Kristine IDs whose current API state is READY."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "sweep"
            )
            result = self.run_script(SWEEP_SCRIPT, ["project"], environment)
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "ready-kristine\tus-central1-a\n")
        self.assertIn("--query=name:kristine", commands)
        self.assertIn("describe ready-kristine", commands)
        self.assertIn("describe pending-kristine", commands)

    def test_sweep_rejects_an_unexpected_non_kristine_result(self) -> None:
        """Stop before describing any non-Kristine result from asset inventory."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "unauthorized"
            )
            result = self.run_script(SWEEP_SCRIPT, ["project"], environment)
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertIn("unauthorized TPU ID", result.stderr)
        self.assertNotIn("tpu-vm describe", commands)

    def test_occupancy_rejects_denylisted_ids_before_gcloud(self) -> None:
        """Refuse every denylisted TPU before making any cloud call."""
        for tpu_identifier in DENYLISTED_TPU_IDENTIFIERS:
            with self.subTest(tpu_identifier=tpu_identifier):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    environment, command_log = self.make_fake_environment(
                        Path(temporary_directory_name), "available"
                    )
                    result = self.run_script(
                        OCCUPANCY_SCRIPT,
                        ["project", tpu_identifier, "us-central2-b", "--claim"],
                        environment,
                    )

                self.assertEqual(result.returncode, 1)
                self.assertIn("refusing denylisted TPU ID", result.stderr)
                self.assertFalse(command_log.exists())

    def test_occupancy_accepts_non_kristine_and_similar_names(self) -> None:
        """Inspect idle non-Kristine TPUs, including names that only resemble the denylist."""
        for tpu_identifier in ("alice-tpu", "kmh-tpuvm-v4-8-20", "kmh-tpuvm-v4-8-5"):
            with self.subTest(tpu_identifier=tpu_identifier):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    environment, _ = self.make_fake_environment(
                        Path(temporary_directory_name), "available"
                    )
                    result = self.run_script(
                        OCCUPANCY_SCRIPT,
                        ["project", tpu_identifier, "us-central1-a"],
                        environment,
                    )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("AVAILABLE worker=worker-0", result.stdout)

    def test_occupancy_never_signals_processes(self) -> None:
        """Keep the helper free of any process termination path."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("kill ", occupancy_source)
        self.assertNotIn("pkill", occupancy_source)
        self.assertEqual(occupancy_source.count("rm -f -- /tmp/libtpu_lockfile"), 1)
        self.assertIn("/dev/accel[0-9]*", occupancy_source)

    def test_occupancy_ignores_the_host_ssh_configuration(self) -> None:
        """Keep occupancy inspection independent of a malformed host SSH config."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "available"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "ready-kristine", "us-central1-a"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ssh-flag=-F --ssh-flag=/dev/null", commands)

    def test_occupancy_protects_active_trainers_without_device_holders(self) -> None:
        """Treat a live trainer invisible to device inspection as an occupant."""
        occupancy_source = OCCUPANCY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("training_entrypoint=train.py", occupancy_source)
        self.assertIn("reason=active_train_process", occupancy_source)

    def test_claim_never_touches_any_holder(self) -> None:
        """Stop after one inspection when the current user or anyone else holds the TPU."""
        for mode in ("occupied", "foreign"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    environment, command_log = self.make_fake_environment(
                        Path(temporary_directory_name), mode
                    )
                    result = self.run_script(
                        OCCUPANCY_SCRIPT,
                        ["project", "alice-tpu", "us-central1-a", "--claim"],
                        environment,
                    )
                    commands = command_log.read_text(encoding="utf-8")

                self.assertEqual(result.returncode, 2)
                self.assertIn("OCCUPIED", result.stdout)
                self.assertEqual(commands.count("tpu-vm ssh"), 1)
                self.assertNotIn("remove_stale_lock=1", commands)

    def test_inspection_reports_stale_lock_without_removing_it(self) -> None:
        """Report a holderless lockfile but leave it in place without --claim."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "stale_lock"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "alice-tpu", "us-central1-a"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 3)
        self.assertIn("STALE_LOCK worker=worker-0", result.stdout)
        self.assertEqual(commands.count("tpu-vm ssh"), 1)

    def test_claim_removes_only_a_stale_lock_then_reinspects(self) -> None:
        """Clear a holderless lockfile and require a clean follow-up inspection."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name), "stale_lock"
            )
            result = self.run_script(
                OCCUPANCY_SCRIPT,
                ["project", "alice-tpu", "us-central1-a", "--claim"],
                environment,
            )
            commands = command_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("STALE_LOCK_REMOVED worker=worker-0", result.stdout)
        self.assertTrue(result.stdout.rstrip().endswith("AVAILABLE worker=worker-0"))
        self.assertEqual(commands.count("tpu-vm ssh"), 3)
        self.assertEqual(commands.count("remove_stale_lock=1"), 1)

    def test_query_maps_v6e_topologies_to_supported_runtime(self) -> None:
        """Create the smallest and large v6e topologies with the v6e runtime."""
        for chip_count in ("4", "64", "128", "256"):
            with self.subTest(chip_count=chip_count):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    temporary_directory = Path(temporary_directory_name)
                    environment, command_log = self.make_fake_environment(
                        temporary_directory,
                        "available",
                    )
                    initializer_path = temporary_directory / "initialize.sh"
                    initializer_path.write_text(
                        "#!/bin/sh\nexit 0\n",
                        encoding="utf-8",
                    )
                    initializer_path.chmod(0o755)
                    for executable_name in ("curl", "sleep"):
                        fake_executable_path = (
                            temporary_directory / "bin" / executable_name
                        )
                        fake_executable_path.write_text(
                            "#!/bin/sh\nexit 0\n",
                            encoding="utf-8",
                        )
                        fake_executable_path.chmod(0o755)
                    environment.update(
                        {
                            "ELF_TPU_INITIALIZER": str(initializer_path),
                            "ZONES_OVERRIDE": "us-east5-b",
                        }
                    )

                    result = self.run_script(
                        QUERY_SCRIPT,
                        ["v6e", chip_count, "1"],
                        environment,
                    )
                    create_command = next(
                        command
                        for command in command_log.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if "tpu-vm create" in command
                    )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"--accelerator-type=v6e-{chip_count}",
                    create_command,
                )
                self.assertIn("--version=v2-alpha-tpuv6e", create_command)
                self.assertIn(
                    f"machine_type=v6e-{chip_count} "
                    f"machines_per_tpu={int(chip_count) // 4} "
                    f"total_machines={int(chip_count) // 4}",
                    result.stdout,
                )

    def test_query_rejects_family_zone_mismatches_before_cloud_access(self) -> None:
        """Fail before creation when an override is invalid for the TPU family."""
        invalid_combinations = (
            ("v5e", "64", "us-central2-b"),
            ("v5p", "32", "asia-northeast1-b"),
            ("v6e", "64", "us-west4-a"),
            ("v4", "32", "us-east5-b"),
        )
        for accelerator_type, chip_count, zone in invalid_combinations:
            with self.subTest(accelerator_type=accelerator_type, zone=zone):
                with tempfile.TemporaryDirectory() as temporary_directory_name:
                    environment, command_log = self.make_fake_environment(
                        Path(temporary_directory_name),
                        "available",
                    )
                    environment["ZONES_OVERRIDE"] = zone
                    result = self.run_script(
                        QUERY_SCRIPT,
                        [accelerator_type, chip_count, "1"],
                        environment,
                    )

                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    f"zone {zone} does not support {accelerator_type}",
                    result.stderr,
                )
                self.assertFalse(command_log.exists())

    def test_query_rejects_an_empty_zone_override_before_cloud_access(self) -> None:
        """Reject a whitespace-only zone override instead of entering an empty loop."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            environment, command_log = self.make_fake_environment(
                Path(temporary_directory_name),
                "available",
            )
            environment["ZONES_OVERRIDE"] = "   "
            result = self.run_script(
                QUERY_SCRIPT,
                ["v6e", "64", "1"],
                environment,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "ZONES_OVERRIDE must contain at least one zone",
            result.stderr,
        )
        self.assertFalse(command_log.exists())

    def test_query_maps_v5e_64_to_v5litepod_runtime(self) -> None:
        """Create v5e resources with Cloud TPU's v5litepod type and runtime names."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "64", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-64", create_command)
        self.assertIn("--version=v2-alpha-tpuv5-lite", create_command)

    def test_query_maps_v5e_4_to_one_v5litepod_vm(self) -> None:
        """Represent the four-chip v5e slice as one TPU VM."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "4", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-4", create_command)
        self.assertIn("machine_type=v5litepod-4 machines_per_tpu=1 total_machines=1", result.stdout)

    def test_query_uses_us_east5_bucket_and_elf_initializer_contract(self) -> None:
        """Create only an owned us-east5 TPU with the region-bound ELF identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_log = temporary_directory / "initializer.log"
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > \"$INITIALIZER_LOG\"\n",
                encoding="utf-8",
            )
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "INITIALIZER_LOG": str(initializer_log),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(
                QUERY_SCRIPT,
                ["v6e", "8", "1"],
                environment,
            )
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )
            initializer_arguments = initializer_log.read_text(
                encoding="utf-8"
            ).strip().split()
            query_source = QUERY_SCRIPT.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--zone=us-east5-b", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )
        self.assertRegex(
            create_command,
            r"tpu-vm create kmh-tpuvm-v6e-8-spot-kristine-.*-us-east5-b ",
        )
        self.assertEqual(
            initializer_arguments[0], "he-vision-group"
        )
        self.assertRegex(
            initializer_arguments[1],
            r"^kmh-tpuvm-v6e-8-spot-kristine-.*-us-east5-b$",
        )
        self.assertEqual(initializer_arguments[2], "us-east5-b")
        self.assertIn(
            "v6e) printf '%s\\n'",
            query_source,
        )
        self.assertIn(
            "'us-central1-b us-central1-a us-east5-a asia-northeast1-b "
            "us-east5-b us-central1-c'",
            query_source,
        )
        self.assertIn(
            "initializer=${ELF_TPU_INITIALIZER:-"
            "${script_directory}/initialize_elf_runtime.sh}",
            query_source,
        )
        self.assertNotIn("run_init_remote.sh", query_source)

    def test_query_allows_v5e_8_in_pack_backed_us_east5(self) -> None:
        """Use the surveyed v5litepod type with the matching east5 identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--accelerator-type=v5litepod-8", create_command)
        self.assertIn("--zone=us-east5-b", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )

    def test_query_stops_retrying_a_non_retryable_reservation_blocker(self) -> None:
        """Fail cleanly instead of looping forever on a reservation-only offering."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "reservation",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-east5-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_commands = [
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            ]

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(create_commands), 1)
        self.assertIn("Non-retryable create blocker", result.stderr)
        self.assertIn("none:none:BLOCKED", result.stdout)

    def test_query_stops_retrying_a_tenant_project_blocker(self) -> None:
        """Treat a repeated TPU tenant-project failure as deterministic."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "tenant",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-b",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_commands = [
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            ]

        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(create_commands), 1)
        self.assertIn("Non-retryable create blocker", result.stderr)
        self.assertIn("none:none:BLOCKED", result.stdout)

    def test_query_uses_us_west4_bucket_service_account(self) -> None:
        """Create us-west4 TPUs with the matching regional bucket identity."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, command_log = self.make_fake_environment(
                temporary_directory,
                "available",
            )
            initializer_path = temporary_directory / "initialize.sh"
            initializer_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            initializer_path.chmod(0o755)
            for executable_name in ("curl", "sleep"):
                fake_executable_path = temporary_directory / "bin" / executable_name
                fake_executable_path.write_text(
                    "#!/bin/sh\nexit 0\n",
                    encoding="utf-8",
                )
                fake_executable_path.chmod(0o755)
            environment.update(
                {
                    "ELF_TPU_INITIALIZER": str(initializer_path),
                    "ZONES_OVERRIDE": "us-west4-a",
                }
            )

            result = self.run_script(QUERY_SCRIPT, ["v5e", "8", "1"], environment)
            create_command = next(
                command
                for command in command_log.read_text(encoding="utf-8").splitlines()
                if "tpu-vm create" in command
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--zone=us-west4-a", create_command)
        self.assertIn(
            "--service-account="
            "bucket-us-west4@he-vision-group.iam.gserviceaccount.com",
            create_command,
        )


if __name__ == "__main__":
    unittest.main()
