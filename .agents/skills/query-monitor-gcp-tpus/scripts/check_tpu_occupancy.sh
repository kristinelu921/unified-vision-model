#!/usr/bin/env bash
set -euo pipefail

# Exit codes: 0 AVAILABLE, 2 OCCUPIED (any holder, including another person's
# job), 3 STALE_LOCK (only a holderless /tmp/libtpu_lockfile blocks the TPU),
# 4 UNKNOWN. `--claim` removes a proven-stale lockfile and nothing else; this
# helper never signals or terminates any process.

usage='usage: check_tpu_occupancy.sh PROJECT TPU_ID ZONE [--claim]'
project_identifier=${1:?$usage}
tpu_identifier=${2:?$usage}
tpu_zone=${3:?$usage}
claim_mode=${4:-}
gcloud_ssh_flags=(--ssh-flag=-F --ssh-flag=/dev/null)
use_internal_tpu_ip=${ELF_TPU_USE_INTERNAL_IP:-1}
readonly denylisted_tpu_identifiers=(
    kmh-tpuvm-v4-8-2
    kmh-tpuvm-v4-8-3
    kmh-tpuvm-v4-8-4
    kmh-tpuvm-v4-8-6
    kmh-tpuvm-v4-8-8
)

for denylisted_tpu_identifier in "${denylisted_tpu_identifiers[@]}"; do
    if [[ "${tpu_identifier,,}" == "${denylisted_tpu_identifier}" ]]; then
        printf 'ERROR: refusing denylisted TPU ID: %s\n' "${tpu_identifier}" >&2
        exit 1
    fi
done
if [[ "$use_internal_tpu_ip" != 0 && "$use_internal_tpu_ip" != 1 ]]; then
    printf 'ERROR: ELF_TPU_USE_INTERNAL_IP must be 0 or 1\n' >&2
    exit 1
fi
if [[ "$use_internal_tpu_ip" == 1 ]]; then
    gcloud_ssh_flags+=(--internal-ip)
fi
if [[ ! "$tpu_identifier" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'ERROR: TPU ID contains unsupported characters: %s\n' "$tpu_identifier" >&2
    exit 1
fi
if [[ -n "${claim_mode}" && "${claim_mode}" != '--claim' ]]; then
    printf 'ERROR: optional fourth argument must be --claim\n' >&2
    exit 1
fi

current_state=$(
    gcloud compute tpus tpu-vm describe "${tpu_identifier}" \
        --project="${project_identifier}" \
        --zone="${tpu_zone}" \
        --format='value(state)'
)
if [[ "${current_state}" != 'READY' ]]; then
    printf 'UNKNOWN TPU_ID=%s ZONE=%s STATE=%s\n' \
        "${tpu_identifier}" "${tpu_zone}" "${current_state}"
    exit 4
fi

# The same worker probe runs in inspection and claim phases. The claim phase
# re-proves every invariant on the worker immediately before removing the lock.
read -r -d '' worker_probe <<'REMOTE_COMMAND' || true
set -euo pipefail
remove_stale_lock=__REMOVE_STALE_LOCK__
worker_hostname=$(hostname)
current_user=$(id -un)
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'UNKNOWN worker=%s reason=vfio_inspection_unavailable\n' "${worker_hostname}"
    exit 4
fi
holder_found=0
for accelerator_resource in /dev/vfio/* /dev/accel[0-9]* /tmp/libtpu_lockfile; do
    [[ -e "${accelerator_resource}" ]] || continue
    device_holders=$(sudo -n fuser "${accelerator_resource}" 2>/dev/null || true)
    for process_identifier in ${device_holders}; do
        [[ "${process_identifier}" =~ ^[0-9]+$ ]] || continue
        holder_found=1
        process_owner=$(ps -o user= -p "${process_identifier}" | xargs || true)
        process_arguments=$(ps -o args= -p "${process_identifier}" || true)
        holder_label=FOREIGN
        if [[ "${process_owner}" == "${current_user}" ]]; then
            holder_label=OWN
        fi
        printf '%s worker=%s resource=%s pid=%s user=%s command=%q\n' \
            "${holder_label}" "${worker_hostname}" "${accelerator_resource}" \
            "${process_identifier}" "${process_owner:-unknown}" "${process_arguments}"
    done
done

# Some v4 images expose no device or lockfile holder while a live trainer
# still owns the accelerator. Inspect NUL-delimited argv entries so the
# helper's own remote-command argument, which contains the text "train.py",
# cannot be mistaken for a trainer entrypoint.
for process_dir in /proc/[0-9]*; do
    process_identifier=${process_dir##*/}
    process_name=$(cat "${process_dir}/comm" 2>/dev/null || true)
    [[ "${process_name,,}" == python* ]] || continue
    [[ -r "${process_dir}/cmdline" ]] || continue
    mapfile -d '' -t process_argv 2>/dev/null < "${process_dir}/cmdline" || continue
    training_found=0
    for process_arg in "${process_argv[@]}"; do
        if [[ "${process_arg}" == train.py || "${process_arg}" == */train.py ]]; then
            training_found=1
            break
        fi
    done
    (( training_found )) || continue
    holder_found=1
    process_owner=$(ps -o user= -p "${process_identifier}" | xargs || true)
    process_arguments=$(ps -o args= -p "${process_identifier}" || true)
    printf 'TRAINING worker=%s pid=%s user=%s reason=active_train_process command=%q\n' \
        "${worker_hostname}" "${process_identifier}" "${process_owner:-unknown}" \
        "${process_arguments}"
done

if (( holder_found )); then
    printf 'OCCUPIED worker=%s\n' "${worker_hostname}"
    exit 2
fi
if [[ -e /tmp/libtpu_lockfile ]]; then
    lockfile_metadata=$(sudo -n stat -c 'owner=%U uid=%u mode=%a size=%s modified=%y' \
        /tmp/libtpu_lockfile 2>/dev/null || true)
    if (( ! remove_stale_lock )); then
        printf 'STALE_LOCK worker=%s reason=stale_libtpu_lock %s\n' \
            "${worker_hostname}" "${lockfile_metadata}"
        exit 3
    fi
    sudo -n rm -f -- /tmp/libtpu_lockfile
    printf 'STALE_LOCK_REMOVED worker=%s %s\n' "${worker_hostname}" "${lockfile_metadata}"
fi
printf 'AVAILABLE worker=%s\n' "${worker_hostname}"
REMOTE_COMMAND

run_worker_probe() {
    local remove_stale_lock=$1
    local probe_output probe_status=0
    probe_output=$(gcloud compute tpus tpu-vm ssh "${tpu_identifier}" \
        "${gcloud_ssh_flags[@]}" \
        --project="${project_identifier}" \
        --zone="${tpu_zone}" \
        --worker=all \
        --command="${worker_probe/__REMOVE_STALE_LOCK__/$remove_stale_lock}") || probe_status=$?
    printf '%s\n' "${probe_output}"
    if grep -q '^OCCUPIED worker=' <<<"${probe_output}"; then
        return 2
    fi
    if grep -q '^UNKNOWN worker=' <<<"${probe_output}"; then
        return 4
    fi
    if (( probe_status != 0 )) && ! grep -q '^STALE_LOCK worker=' <<<"${probe_output}"; then
        printf 'UNKNOWN TPU_ID=%s ZONE=%s reason=ssh_or_inspection_failed\n' \
            "${tpu_identifier}" "${tpu_zone}"
        return 4
    fi
    if grep -q '^STALE_LOCK worker=' <<<"${probe_output}"; then
        return 3
    fi
    return 0
}

inspection_status=0
run_worker_probe 0 || inspection_status=$?
if (( inspection_status != 3 )) || [[ "${claim_mode}" != '--claim' ]]; then
    exit "${inspection_status}"
fi

# Every worker is free apart from holderless lockfiles, so clear them and
# require a clean follow-up inspection.
claim_status=0
run_worker_probe 1 || claim_status=$?
if (( claim_status != 0 )); then
    exit "${claim_status}"
fi
run_worker_probe 0
