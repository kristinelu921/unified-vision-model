import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECONCILE_SCRIPT = REPOSITORY_ROOT / "tpu_scripts" / "check_tpus.sh"
INITIALIZER_SCRIPT = REPOSITORY_ROOT / "tpu_scripts" / "initialize_elf_runtime.sh"


class ReconcileReadyElfTpusTest(unittest.TestCase):
    def test_supports_internal_ip_for_every_remote_probe(self) -> None:
        """Route reconciliation SSH probes through internal TPU addresses."""
        reconcile_source = RECONCILE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("ELF_TPU_USE_INTERNAL_IP", reconcile_source)
        self.assertIn("ELF_TPU_USE_INTERNAL_IP:-1", reconcile_source)
        self.assertIn("gcloud_ssh_flags+=(--internal-ip)", reconcile_source)

    def make_fake_environment(
        self,
        temporary_directory: Path,
        inventory_mode: str,
    ) -> tuple[dict[str, str], Path, Path]:
        """Create fake cloud and initializer commands for reconciliation tests."""
        binary_directory = temporary_directory / "bin"
        binary_directory.mkdir()
        gcloud_log_path = temporary_directory / "gcloud.log"
        initializer_log_path = temporary_directory / "initializer.log"
        fake_gcloud_path = binary_directory / "gcloud"
        fake_gcloud_path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_GCLOUD_LOG}"
for argument in "$@"; do
    if [[ "$argument" == --command=* ]]; then
        bash -n -c "${argument#--command=}"
    fi
done
arguments=$*
if [[ "$arguments" == *"asset search-all-resources"* ]]; then
    if [[ "${FAKE_INVENTORY_MODE}" == unauthorized ]]; then
        printf '%s\n' 'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/alice-tpu'
    elif [[ "${FAKE_INVENTORY_MODE}" == stale ]]; then
        :
    elif [[ "${FAKE_INVENTORY_MODE}" == parallel ]]; then
        printf '%s\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/first-kristine' \
            'us-central1-b,//tpu.googleapis.com/projects/p/locations/us-central1-b/nodes/second-kristine'
    elif [[ "${FAKE_INVENTORY_MODE}" == readiness_unknown ]]; then
        printf '%s\n' 'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/unknown-kristine'
    else
        printf '%s\n' \
            'us-central1-a,//tpu.googleapis.com/projects/p/locations/us-central1-a/nodes/initialized-kristine' \
            'us-central1-b,//tpu.googleapis.com/projects/p/locations/us-central1-b/nodes/new-kristine' \
            'us-central1-d,//tpu.googleapis.com/projects/p/locations/us-central1-d/nodes/occupied-kristine' \
            'us-central1-c,//tpu.googleapis.com/projects/p/locations/us-central1-c/nodes/pending-kristine'
    fi
elif [[ "$arguments" == *"tpu-vm describe"* ]]; then
    if [[ "$arguments" == *"value(state)"* ]]; then
        if [[ "${FAKE_INVENTORY_MODE}" == parallel ]]; then
            if [[ "$arguments" == *"first-kristine"* ]]; then
                touch "${FAKE_FIRST_PROBE_STARTED}"
                for _attempt in {1..100}; do
                    [[ -f "${FAKE_SECOND_PROBE_STARTED}" ]] && break
                    sleep 0.01
                done
                [[ -f "${FAKE_SECOND_PROBE_STARTED}" ]] || exit 91
            else
                touch "${FAKE_SECOND_PROBE_STARTED}"
            fi
        fi
        [[ "$arguments" == *"pending-kristine"* ]] && printf 'CREATING\n' || printf 'READY\n'
    else
        printf 'HEALTHY\n'
    fi
elif [[ "$arguments" == *"ELF_RUNTIME_READINESS_PROBE"* ]]; then
    if [[ "${FAKE_INVENTORY_MODE}" == readiness_unknown ]]; then
        exit 255
    elif [[ "$arguments" == *"initialized-kristine"* ]]; then
        printf 'ELF_RUNTIME_READY worker=worker-0\n'
    else
        printf 'ELF_RUNTIME_MISSING worker=worker-0\n'
    fi
elif [[ "$arguments" == *"ELF_RUNTIME_OCCUPANCY_PROBE"* ]]; then
    if [[ "$arguments" == *"occupied-kristine"* ]]; then
        printf 'HOLDER worker=worker-0 device=/dev/vfio/0 pid=42 owner=alice process=python3\n'
        exit 20
    fi
    printf 'IDLE\n'
elif [[ "$arguments" == *"ELF_RUNTIME_NOT_INITIALIZING worker="* ]]; then
    printf 'ELF_RUNTIME_NOT_INITIALIZING worker=worker-0\n'
fi
""",
            encoding="utf-8",
        )
        fake_gcloud_path.chmod(0o755)
        fake_initializer_path = temporary_directory / "fake_initializer.sh"
        fake_initializer_path.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_INITIALIZER_LOG}"
""",
            encoding="utf-8",
        )
        fake_initializer_path.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "ELF_TPU_INITIALIZER": str(fake_initializer_path),
                "ELF_TPU_INITIALIZER_FOREGROUND": "1",
                "ELF_TPU_INIT_LOG_DIR": str(temporary_directory / "logs"),
                "FAKE_GCLOUD_LOG": str(gcloud_log_path),
                "FAKE_INITIALIZER_LOG": str(initializer_log_path),
                "FAKE_INVENTORY_MODE": inventory_mode,
                "PATH": f"{binary_directory}:{environment['PATH']}",
            }
        )
        return environment, gcloud_log_path, initializer_log_path

    def test_initializes_only_ready_uninitialized_tpus(self) -> None:
        """Report idle TPUs as available independently of ELF runtime readiness."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, gcloud_log_path, initializer_log_path = self.make_fake_environment(
                temporary_directory,
                inventory_mode="normal",
            )
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            initializer_log = initializer_log_path.read_text(encoding="utf-8")
            gcloud_log = gcloud_log_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "TPU_NAME\tZONE\tTPU_STATUS\tELF_RUNTIME_STATUS\tDETAIL",
            result.stdout,
        )
        self.assertIn(
            "initialized-kristine\tus-central1-a\tAVAILABLE\tREADY",
            result.stdout,
        )
        self.assertIn(
            "new-kristine\tus-central1-b\tAVAILABLE\tINITIALIZED_NOW",
            result.stdout,
        )
        self.assertIn(
            "occupied-kristine\tus-central1-d\tOCCUPIED\tNOT_CHECKED",
            result.stdout,
        )
        self.assertIn(
            "HOLDER worker=worker-0 device=/dev/vfio/0 pid=42 owner=alice process=python3",
            result.stdout,
        )
        self.assertIn(
            "pending-kristine\tus-central1-c\tUNAVAILABLE\tNOT_CHECKED\tstate=CREATING",
            result.stdout,
        )
        self.assertIn("project new-kristine us-central1-b", initializer_log)
        self.assertNotIn("initialized-kristine", initializer_log)
        self.assertNotIn("occupied-kristine", initializer_log)
        self.assertNotIn("pending-kristine", initializer_log)
        self.assertIn("--ssh-flag=-F --ssh-flag=/dev/null", gcloud_log)
        self.assertIn("wandb.api.api_key", gcloud_log)
        self.assertNotIn("dataset_info.json", gcloud_log)
        self.assertNotIn("/state.json", gcloud_log)
        self.assertIn(
            "AVAILABLE\nTPU_NAME\tZONE\n"
            "initialized-kristine\tus-central1-a\n"
            "new-kristine\tus-central1-b\n",
            result.stdout,
        )

    def test_rejects_unexpected_non_kristine_inventory_before_describe(self) -> None:
        """Reject a bad server-filter result before inspecting any returned TPU."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, gcloud_log_path, _ = self.make_fake_environment(
                temporary_directory,
                inventory_mode="unauthorized",
            )
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            gcloud_log = gcloud_log_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 1)
        self.assertIn("unauthorized TPU ID", result.stderr)
        self.assertNotIn("tpu-vm describe", gcloud_log)

    def test_reconciles_exact_candidate_missing_from_asset_inventory(self) -> None:
        """Reconcile an authorized exact TPU when Cloud Asset inventory is stale."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, gcloud_log_path, _ = self.make_fake_environment(
                temporary_directory,
                inventory_mode="stale",
            )
            environment["ELF_TPU_EXACT_CANDIDATES"] = (
                "existing-kristine,us-west4-a"
            )
            environment["ELF_TPU_INITIALIZE_MISSING"] = "0"
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            gcloud_log = gcloud_log_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "existing-kristine\tus-west4-a\tAVAILABLE\tMISSING",
            result.stdout,
        )
        self.assertIn(
            "AVAILABLE\nTPU_NAME\tZONE\nexisting-kristine\tus-west4-a",
            result.stdout,
        )
        self.assertIn("tpu-vm describe existing-kristine", gcloud_log)

    def test_reconciles_tpu_candidates_in_parallel(self) -> None:
        """Overlap independent TPU probes while preserving inventory output order."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, _, _ = self.make_fake_environment(
                temporary_directory,
                inventory_mode="parallel",
            )
            environment.update(
                {
                    "ELF_TPU_CHECK_PARALLELISM": "2",
                    "ELF_TPU_INITIALIZE_MISSING": "0",
                    "FAKE_FIRST_PROBE_STARTED": str(
                        temporary_directory / "first-probe-started"
                    ),
                    "FAKE_SECOND_PROBE_STARTED": str(
                        temporary_directory / "second-probe-started"
                    ),
                }
            )
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("first-kristine\tus-central1-a\tAVAILABLE", result.stdout)
        self.assertIn("second-kristine\tus-central1-b\tAVAILABLE", result.stdout)
        self.assertIn(
            "AVAILABLE\nTPU_NAME\tZONE\n"
            "first-kristine\tus-central1-a\n"
            "second-kristine\tus-central1-b",
            result.stdout,
        )

    def test_rejects_unauthorized_exact_candidate_before_cloud_access(self) -> None:
        """Reject an exact non-kristine TPU before making any cloud request."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, gcloud_log_path, _ = self.make_fake_environment(
                temporary_directory,
                inventory_mode="stale",
            )
            environment["ELF_TPU_EXACT_CANDIDATES"] = "alice-tpu,us-west4-a"
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing non-kristine exact TPU candidate", result.stderr)
        self.assertFalse(gcloud_log_path.exists())

    def test_runtime_probe_failure_does_not_start_initialization(self) -> None:
        """Keep idle hardware available without initializing after an unknown probe."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, _, initializer_log_path = self.make_fake_environment(
                temporary_directory,
                inventory_mode="readiness_unknown",
            )
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "unknown-kristine\tus-central1-a\tAVAILABLE\tUNKNOWN\t"
            "ELF runtime probe failed",
            result.stdout,
        )
        self.assertIn(
            "AVAILABLE\nTPU_NAME\tZONE\nunknown-kristine\tus-central1-a",
            result.stdout,
        )
        self.assertFalse(initializer_log_path.exists())

    def test_read_only_discovery_does_not_initialize_missing_runtimes(self) -> None:
        """List available TPUs without mutating runtimes during candidate discovery."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, _, initializer_log_path = self.make_fake_environment(
                temporary_directory,
                inventory_mode="normal",
            )
            environment["ELF_TPU_INITIALIZE_MISSING"] = "0"
            result = subprocess.run(
                ["bash", str(RECONCILE_SCRIPT), "project"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "new-kristine\tus-central1-b\tAVAILABLE\tMISSING\t"
            "read-only discovery; initialization disabled",
            result.stdout,
        )
        self.assertFalse(initializer_log_path.exists())

    def test_discovery_probe_rejects_active_trainers_without_device_holders(self) -> None:
        """Fail closed on v4 trainers that appear in neither VFIO nor lockfile state."""
        reconcile_source = RECONCILE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("resource=active_train_process", reconcile_source)
        self.assertIn("training_entrypoint=train.py", reconcile_source)
        self.assertIn("ps -eo pid=,user=,comm=,args=", reconcile_source)

    def test_initializer_rejects_denylisted_tpu_before_gcloud(self) -> None:
        """Reject direct initializer calls on a denylisted TPU before cloud access."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            environment, gcloud_log_path, _ = self.make_fake_environment(
                temporary_directory,
                inventory_mode="normal",
            )
            result = subprocess.run(
                ["bash", str(INITIALIZER_SCRIPT), "project", "kmh-tpuvm-v4-8-8", "us-central2-b"],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("refusing denylisted TPU ID", result.stderr)
        self.assertFalse(gcloud_log_path.exists())

    def test_initializer_provisions_wandb_netrc_on_every_worker(self) -> None:
        """Copy W&B credentials separately and verify authenticated readiness."""
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            binary_directory = temporary_directory / "bin"
            binary_directory.mkdir()
            gcloud_log_path = temporary_directory / "gcloud.log"
            fake_gcloud_path = binary_directory / "gcloud"
            fake_gcloud_path.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_GCLOUD_LOG}"
arguments=$*
if [[ "$arguments" == *"tpu-vm describe"*"value(state)"* ]]; then
    printf 'READY\n'
elif [[ "$arguments" == *"tpu-vm describe"*"value(health)"* ]]; then
    printf 'HEALTHY\n'
elif [[ "$arguments" == *"RUNTIME_INSTALLED worker="* ]]; then
    printf 'RUNTIME_INSTALLED worker=worker-0 runtime=/tmp/runtime freeze_sha256=%064d\n' 0
fi
""",
                encoding="utf-8",
            )
            fake_gcloud_path.chmod(0o755)
            netrc_path = temporary_directory / ".netrc"
            netrc_secret = "test-wandb-secret"
            netrc_path.write_text(
                f"machine api.wandb.ai login user password {netrc_secret}\n",
                encoding="utf-8",
            )
            netrc_path.chmod(0o600)
            environment = os.environ.copy()
            environment.update(
                {
                    "ELF_NETRC_PATH": str(netrc_path),
                    "ELF_REMOTE_USER": "testuser",
                    "FAKE_GCLOUD_LOG": str(gcloud_log_path),
                    "PATH": f"{binary_directory}:{environment['PATH']}",
                }
            )

            result = subprocess.run(
                [
                    "bash",
                    str(INITIALIZER_SCRIPT),
                    "project",
                    "test-kristine-tpu",
                    "us-central1-a",
                ],
                cwd=REPOSITORY_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            gcloud_log = gcloud_log_path.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tpu-vm scp", gcloud_log)
        self.assertIn("--worker=all", gcloud_log)
        self.assertIn("/home/testuser/.netrc", gcloud_log)
        self.assertIn("INITIALIZED_WITH_WANDB", gcloud_log)
        self.assertIn("wandb.api.api_key", gcloud_log)
        self.assertNotIn(netrc_secret, gcloud_log)


if __name__ == "__main__":
    unittest.main()
