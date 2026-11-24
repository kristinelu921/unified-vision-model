#!/usr/bin/env bash
# Do not use `set -e`: one inaccessible TPU should produce an UNKNOWN row while
# the sweep continues to report the other authorized candidates.
set -uo pipefail

# Report general TPU availability separately from optional ELF runtime readiness.
# Missing ELF runtimes are initialized asynchronously by default.
#
# TPU_STATUS describes only the accelerator: READY, HEALTHY, and idle VFIO.
# ELF_RUNTIME_STATUS describes reusable ELF dependencies, Muon, and W&B auth.
# Dataset and run-config readiness intentionally belong to the launch script.

project_identifier=${1:-he-vision-group}
readonly server_side_name_query='name:kristine'
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
initializer=${ELF_TPU_INITIALIZER:-${script_directory}/initialize_elf_runtime.sh}
initialization_log_root=${ELF_TPU_INIT_LOG_DIR:-/tmp/elf-tpu-initialization}
run_initializer_foreground=${ELF_TPU_INITIALIZER_FOREGROUND:-0}
initialize_missing_runtimes=${ELF_TPU_INITIALIZE_MISSING:-1}
gcloud_ssh_flags=(--ssh-flag=-F --ssh-flag=/dev/null)
use_internal_tpu_ip=${ELF_TPU_USE_INTERNAL_IP:-1}
exact_tpu_candidates=${ELF_TPU_EXACT_CANDIDATES:-}
parallel_probe_count=${ELF_TPU_CHECK_PARALLELISM:-8}
probe_scratch_root=${ELF_TPU_CHECK_TMP_ROOT:-/tmp}

if [[ "$initialize_missing_runtimes" != 0 && "$initialize_missing_runtimes" != 1 ]]; then
    printf 'ERROR: ELF_TPU_INITIALIZE_MISSING must be 0 or 1\n' >&2
    exit 2
fi
if [[ "$use_internal_tpu_ip" != 0 && "$use_internal_tpu_ip" != 1 ]]; then
    printf 'ERROR: ELF_TPU_USE_INTERNAL_IP must be 0 or 1\n' >&2
    exit 2
fi
if [[ ! "$parallel_probe_count" =~ ^[1-9][0-9]*$ ]]; then
    printf 'ERROR: ELF_TPU_CHECK_PARALLELISM must be a positive integer\n' >&2
    exit 2
fi
if [[ ! -d "$probe_scratch_root" || ! -w "$probe_scratch_root" ]]; then
    printf 'ERROR: ELF_TPU_CHECK_TMP_ROOT must be an existing writable directory: %s\n' \
        "$probe_scratch_root" >&2
    exit 2
fi
if [[ "$use_internal_tpu_ip" == 1 ]]; then
    gcloud_ssh_flags+=(--internal-ip)
fi

if [[ ! -x "$initializer" ]]; then
    printf 'ERROR: ELF TPU initializer is missing or not executable: %s\n' "$initializer" >&2
    exit 1
fi
mkdir -p "$initialization_log_root"

declare -a tpu_identifiers=()
declare -a tpu_zones=()
declare -A seen_tpu_locations=()

# Cloud Asset Search is eventually consistent. Accept exact, user-confirmed
# NAME,ZONE pairs so newly created or stale-index TPUs still receive the same
# health, occupancy, and runtime checks as discovered candidates.
for exact_tpu_candidate in $exact_tpu_candidates; do
    IFS=, read -r tpu_identifier tpu_zone extra_field <<<"$exact_tpu_candidate"
    if [[ -z "$tpu_identifier" || -z "$tpu_zone" || -n "${extra_field:-}" ]]; then
        printf 'ERROR: ELF_TPU_EXACT_CANDIDATES entries must be NAME,ZONE pairs: %s\n' \
            "$exact_tpu_candidate" >&2
        exit 2
    fi
    if [[ "${tpu_identifier,,}" != *kristine* ]]; then
        printf 'ERROR: refusing non-kristine exact TPU candidate: %s\n' \
            "$tpu_identifier" >&2
        exit 2
    fi
    tpu_location_key="${tpu_identifier},${tpu_zone}"
    seen_tpu_locations["$tpu_location_key"]=1
    tpu_identifiers+=("$tpu_identifier")
    tpu_zones+=("$tpu_zone")
done

# The server-side name filter prevents discovery from returning other owners'
# resources; the client-side check below fails closed if the API violates it.
if ! asset_inventory=$(gcloud asset search-all-resources \
        --scope="projects/${project_identifier}" \
        --asset-types='tpu.googleapis.com/Node' \
        --query="$server_side_name_query" \
        --format='csv[no-heading](location,name)'); then
    printf 'ERROR: filtered TPU discovery failed\n' >&2
    exit 1
fi

while IFS=, read -r tpu_zone full_resource_name; do
    [[ -z "$tpu_zone" || -z "$full_resource_name" ]] && continue
    tpu_identifier=${full_resource_name##*/}
    if [[ "${tpu_identifier,,}" != *kristine* ]]; then
        printf 'ERROR: server-side filter returned unauthorized TPU ID: %s\n' \
            "$tpu_identifier" >&2
        exit 1
    fi
    tpu_location_key="${tpu_identifier},${tpu_zone}"
    if [[ -z "${seen_tpu_locations[$tpu_location_key]:-}" ]]; then
        seen_tpu_locations["$tpu_location_key"]=1
        tpu_identifiers+=("$tpu_identifier")
        tpu_zones+=("$tpu_zone")
    fi
done <<<"$asset_inventory"

reconcile_tpu_candidate() {
    local candidate_index=$1
    local tpu_identifier=$2
    local tpu_zone=$3
    local candidate_available_path="${probe_result_root}/${candidate_index}.available"
    local candidate_initialization_path="${probe_result_root}/${candidate_index}.initialization"
    local current_state current_health occupancy_probe occupancy_output
    local initialization_probe initialization_output readiness_probe readiness_output log_path

    # Re-read current state and health for every server-filtered candidate.
    if ! current_state=$(gcloud compute tpus tpu-vm describe "$tpu_identifier" \
            --project="$project_identifier" --zone="$tpu_zone" --format='value(state)'); then
        printf '%s\t%s\tUNKNOWN\tNOT_CHECKED\tstate lookup failed\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if ! current_health=$(gcloud compute tpus tpu-vm describe "$tpu_identifier" \
            --project="$project_identifier" --zone="$tpu_zone" --format='value(health)'); then
        printf '%s\t%s\tUNKNOWN\tNOT_CHECKED\thealth lookup failed\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if [[ "$current_state" != READY ]]; then
        printf '%s\t%s\tUNAVAILABLE\tNOT_CHECKED\tstate=%s\n' \
            "$tpu_identifier" "$tpu_zone" "$current_state"
        return
    fi
    if [[ "$current_health" != HEALTHY ]]; then
        printf '%s\t%s\tUNAVAILABLE\tNOT_CHECKED\thealth=%s\n' \
            "$tpu_identifier" "$tpu_zone" "$current_health"
        return
    fi

    # Availability is fail-closed across every worker: an occupied device or an
    # inability to inspect VFIO keeps the TPU out of the AVAILABLE summary.
    read -r -d '' occupancy_probe <<'REMOTE_COMMAND' || true
set -euo pipefail
printf 'ELF_RUNTIME_OCCUPANCY_PROBE worker=%s\n' "$(hostname)"
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'UNKNOWN worker=%s reason=vfio_inspection_unavailable\n' "$(hostname)" >&2
    exit 21
fi
if ! compgen -G '/dev/vfio/*' >/dev/null; then
    printf 'UNKNOWN worker=%s reason=no_vfio_devices\n' "$(hostname)" >&2
    exit 21
fi
accelerator_resources=(/dev/vfio/*)
if [[ -e /tmp/libtpu_lockfile ]]; then
    accelerator_resources+=(/tmp/libtpu_lockfile)
fi
for accelerator_resource in "${accelerator_resources[@]}"; do
    [[ -e "$accelerator_resource" ]] || continue
    device_holders=$(sudo -n fuser "$accelerator_resource" 2>/dev/null || true)
    for process_identifier in $device_holders; do
        [[ "$process_identifier" =~ ^[0-9]+$ ]] || continue
        process_owner=$(ps -o user= -p "$process_identifier" | xargs || true)
        process_name=$(ps -o comm= -p "$process_identifier" | xargs || true)
        printf 'HOLDER worker=%s resource=%s pid=%s owner=%s process=%s\n' \
            "$(hostname)" "$accelerator_resource" "$process_identifier" \
            "${process_owner:-unknown}" "${process_name:-unknown}"
        holder_found=1
    done
done
# Some v4 images expose no VFIO or lockfile holder while a live trainer still
# owns the accelerator. Fail closed on any active training entrypoint.
training_entrypoint=train.py
while read -r process_identifier process_owner process_name process_arguments; do
    [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
    padded_process_arguments=" ${process_arguments} "
    if [[ "${padded_process_arguments}" == *" ${training_entrypoint} "* \
            || "${padded_process_arguments}" == *"/${training_entrypoint} "* ]]; then
        printf 'HOLDER worker=%s resource=active_train_process pid=%s owner=%s process=%s\n' \
            "$(hostname)" "${process_identifier}" "${process_owner:-unknown}" \
            "${process_name:-unknown}"
        holder_found=1
    fi
done < <(ps -eo pid=,user=,comm=,args=)
if [[ "${holder_found:-0}" == 1 ]]; then
    exit 20
fi
REMOTE_COMMAND
    if ! occupancy_output=$(gcloud compute tpus tpu-vm ssh "$tpu_identifier" \
            "${gcloud_ssh_flags[@]}" \
            --project="$project_identifier" --zone="$tpu_zone" --worker=all \
            --command="$occupancy_probe" 2>&1); then
        if grep -q 'HOLDER worker=' <<<"$occupancy_output"; then
            printf '%s\t%s\tOCCUPIED\tNOT_CHECKED\tholder details follow\n' \
                "$tpu_identifier" "$tpu_zone"
            grep 'HOLDER worker=' <<<"$occupancy_output" | sed 's/^/  /'
        else
            printf '%s\t%s\tUNKNOWN\tNOT_CHECKED\tvfio inspection failed\n' \
                "$tpu_identifier" "$tpu_zone"
            grep 'UNKNOWN worker=' <<<"$occupancy_output" | sed 's/^/  /' || true
        fi
        return
    fi

    # From this point onward the hardware is generally available. Later ELF
    # failures change only ELF_RUNTIME_STATUS, not TPU_STATUS.
    printf '%s\t%s\n' "$tpu_identifier" "$tpu_zone" >"$candidate_available_path"

    # Avoid duplicate installs when another reconciler has already started one
    # on any worker. A failed or malformed probe is UNKNOWN and never treated as
    # permission to initialize.
    read -r -d '' initialization_probe <<'REMOTE_COMMAND' || true
set -euo pipefail
current_user=$(id -un)
if [[ -d "/home/${current_user}/elf-runs/runtime/INITIALIZING" ]]; then
    printf 'ELF_RUNTIME_INITIALIZING worker=%s\n' "$(hostname)"
else
    printf 'ELF_RUNTIME_NOT_INITIALIZING worker=%s\n' "$(hostname)"
fi
REMOTE_COMMAND
    if ! initialization_output=$(gcloud compute tpus tpu-vm ssh "$tpu_identifier" \
            "${gcloud_ssh_flags[@]}" \
            --project="$project_identifier" --zone="$tpu_zone" --worker=all \
            --command="$initialization_probe" 2>&1); then
        printf '%s\t%s\tAVAILABLE\tUNKNOWN\tinitialization-state probe failed\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if grep -q 'ELF_RUNTIME_INITIALIZING worker=' <<<"$initialization_output"; then
        printf '%s\t%s\tAVAILABLE\tINITIALIZING\tan ELF initializer is already running\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if ! grep -q 'ELF_RUNTIME_NOT_INITIALIZING worker=' <<<"$initialization_output"; then
        printf '%s\t%s\tAVAILABLE\tUNKNOWN\tunexpected initialization-state response\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi

    # Probe on CPU so dependency validation never acquires the TPU. Every worker
    # must have the completion marker, imports, Muon, and W&B authentication.
    read -r -d '' readiness_probe <<'REMOTE_COMMAND' || true
set -uo pipefail
printf 'ELF_RUNTIME_READINESS_PROBE worker=%s\n' "$(hostname)"
current_user=$(id -un)
runtime_root="/home/${current_user}/elf-runs/runtime"
python_executable="${runtime_root}/venv/bin/python"
runtime_ready=1
mountpoint -q / || runtime_ready=0
test -w "/home/${current_user}" || runtime_ready=0
compgen -G '/dev/vfio/*' >/dev/null || runtime_ready=0
test -f "${runtime_root}/READY" || runtime_ready=0
if ! JAX_PLATFORMS=cpu "$python_executable" -c \
        'import datasets, einops, flax, huggingface_hub, jax, mauve, numpy, optax, sacrebleu, torch, tqdm, transformers, wandb, yaml' \
        >/dev/null 2>&1; then
    runtime_ready=0
fi
# W&B authentication and Muon support are part of ELF runtime readiness.
if ! "$python_executable" -c \
        'import optax, wandb; assert hasattr(optax.contrib, "muon"); assert wandb.api.api_key' \
        >/dev/null 2>&1; then
    runtime_ready=0
fi
if (( runtime_ready )); then
    printf 'ELF_RUNTIME_READY worker=%s\n' "$(hostname)"
else
    printf 'ELF_RUNTIME_MISSING worker=%s\n' "$(hostname)"
fi
REMOTE_COMMAND
    if ! readiness_output=$(gcloud compute tpus tpu-vm ssh "$tpu_identifier" \
            "${gcloud_ssh_flags[@]}" \
            --project="$project_identifier" --zone="$tpu_zone" --worker=all \
            --command="$readiness_probe" 2>&1); then
        printf '%s\t%s\tAVAILABLE\tUNKNOWN\tELF runtime probe failed\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if grep -q 'ELF_RUNTIME_READY worker=' <<<"$readiness_output" \
            && ! grep -q 'ELF_RUNTIME_MISSING worker=' <<<"$readiness_output"; then
        printf '%s\t%s\tAVAILABLE\tREADY\tall workers have the ELF runtime\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi
    if ! grep -q 'ELF_RUNTIME_MISSING worker=' <<<"$readiness_output"; then
        printf '%s\t%s\tAVAILABLE\tUNKNOWN\tunexpected ELF runtime response\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi

    if [[ "$initialize_missing_runtimes" == 0 ]]; then
        printf '%s\t%s\tAVAILABLE\tMISSING\tread-only discovery; initialization disabled\n' \
            "$tpu_identifier" "$tpu_zone"
        return
    fi

    # Initialization is permitted only after a worker explicitly reports the
    # runtime missing; transport errors and unexpected output were handled above.
    log_path="${initialization_log_root}/${tpu_identifier}-${tpu_zone}.log"
    printf '%s\t%s\t%s\n' "$tpu_identifier" "$tpu_zone" "$log_path" \
        >"$candidate_initialization_path"
    if [[ "$run_initializer_foreground" == 1 ]]; then
        if bash "$initializer" "$project_identifier" "$tpu_identifier" "$tpu_zone" \
                >"$log_path" 2>&1; then
            printf '%s\t%s\tAVAILABLE\tINITIALIZED_NOW\tlog=%s\n' \
                "$tpu_identifier" "$tpu_zone" "$log_path"
        else
            printf '%s\t%s\tAVAILABLE\tINITIALIZATION_FAILED\tlog=%s\n' \
                "$tpu_identifier" "$tpu_zone" "$log_path"
        fi
    else
        nohup bash "$initializer" "$project_identifier" "$tpu_identifier" "$tpu_zone" \
            >"$log_path" 2>&1 &
        printf '%s\t%s\tAVAILABLE\tINITIALIZATION_STARTED\tpid=%s log=%s\n' \
            "$tpu_identifier" "$tpu_zone" "$!" "$log_path"
    fi
}

if ! probe_result_root=$(mktemp -d "${probe_scratch_root%/}/elf-check-tpus.XXXXXXXXXX"); then
    printf 'ERROR: could not create TPU reconciliation scratch directory under %s\n' \
        "$probe_scratch_root" >&2
    exit 1
fi
trap 'rm -rf -- "$probe_result_root"' EXIT
probe_output_lock_path="${probe_result_root}/output.lock"
printf 'TPU_NAME\tZONE\tTPU_STATUS\tELF_RUNTIME_STATUS\tDETAIL\n'
active_probe_count=0
for index in "${!tpu_identifiers[@]}"; do
    (
        reconcile_tpu_candidate \
            "$index" "${tpu_identifiers[$index]}" "${tpu_zones[$index]}" \
            >"${probe_result_root}/${index}.output"
        (
            flock -x 9
            cat "${probe_result_root}/${index}.output"
        ) 9>"$probe_output_lock_path"
    ) &
    ((active_probe_count += 1))
    if (( active_probe_count >= parallel_probe_count )); then
        wait -n
        ((active_probe_count -= 1))
    fi
done
while (( active_probe_count > 0 )); do
    wait -n
    ((active_probe_count -= 1))
done

declare -a initialization_started=()
declare -a available_tpus=()
for index in "${!tpu_identifiers[@]}"; do
    if [[ -f "${probe_result_root}/${index}.available" ]]; then
        available_tpus+=("$(<"${probe_result_root}/${index}.available")")
    fi
    if [[ -f "${probe_result_root}/${index}.initialization" ]]; then
        initialization_started+=("$(<"${probe_result_root}/${index}.initialization")")
    fi
done

# This section reports initializer attempts from this invocation, including
# foreground attempts that may already have completed or failed.
printf '\nINITIALIZATION_KICKED_OFF\n'
if (( ${#initialization_started[@]} == 0 )); then
    printf 'none\n'
else
    printf 'TPU_NAME\tZONE\tLOG\n'
    printf '%s\n' "${initialization_started[@]}"
fi

# This list is intentionally workload-agnostic: it includes idle hardware even
# while ELF initialization is running or its runtime status is UNKNOWN.
printf '\nAVAILABLE\n'
if (( ${#available_tpus[@]} == 0 )); then
    printf 'none\n'
else
    printf 'TPU_NAME\tZONE\n'
    printf '%s\n' "${available_tpus[@]}"
fi
