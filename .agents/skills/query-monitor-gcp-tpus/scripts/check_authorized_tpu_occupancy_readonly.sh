#!/usr/bin/env bash
set -euo pipefail

project_identifier=${1:?'usage: check_authorized_tpu_occupancy_readonly.sh PROJECT TPU_ID ZONE'}
tpu_identifier=${2:?'usage: check_authorized_tpu_occupancy_readonly.sh PROJECT TPU_ID ZONE'}
tpu_zone=${3:?'usage: check_authorized_tpu_occupancy_readonly.sh PROJECT TPU_ID ZONE'}
authorized_id=${ELF_TPU_AUTHORIZED_READ_ONLY_ID:-}
use_internal_tpu_ip=${ELF_TPU_USE_INTERNAL_IP:-1}
gcloud_ssh_flags=(--ssh-flag=-F --ssh-flag=/dev/null)

if [[ "${authorized_id}" != "${tpu_identifier}" ]]; then
    printf 'ERROR: exact read-only TPU authorization mismatch\n' >&2
    exit 1
fi
if [[ ! "$tpu_identifier" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'ERROR: TPU ID contains unsupported characters: %s\n' "$tpu_identifier" >&2
    exit 1
fi
if [[ "$use_internal_tpu_ip" != 0 && "$use_internal_tpu_ip" != 1 ]]; then
    printf 'ERROR: ELF_TPU_USE_INTERNAL_IP must be 0 or 1\n' >&2
    exit 1
fi
if [[ "$use_internal_tpu_ip" == 1 ]]; then
    gcloud_ssh_flags+=(--internal-ip)
fi

current_state=$(gcloud compute tpus tpu-vm describe "$tpu_identifier" \
    --project="$project_identifier" --zone="$tpu_zone" --format='value(state)')
if [[ "$current_state" != READY ]]; then
    printf 'UNKNOWN TPU_ID=%s ZONE=%s STATE=%s\n' \
        "$tpu_identifier" "$tpu_zone" "$current_state"
    exit 4
fi

read -r -d '' inspect_command <<'REMOTE_COMMAND' || true
set -euo pipefail
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'UNKNOWN worker=%s reason=vfio_inspection_unavailable\n' "$(hostname)"
    exit 4
fi
declare -A holders=()
lockfile_holder_found=0
resources=(/dev/vfio/*)
if [[ -e /tmp/libtpu_lockfile ]]; then
    resources+=(/tmp/libtpu_lockfile)
fi
for resource in "${resources[@]}"; do
    [[ -e "$resource" ]] || continue
    holder_pids=$(sudo -n fuser "$resource" 2>/dev/null || true)
    for pid in $holder_pids; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        holders["$pid"]=1
        [[ "$resource" == /tmp/libtpu_lockfile ]] && lockfile_holder_found=1
    done
done

active_python=0
while read -r pid owner command_name command_arguments; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    padded=" $command_arguments "
    if [[ "${command_name,,}" == python* && \
          ("$padded" == *" train.py "* || "$padded" == *"/train.py "*) ]]; then
        active_python=1
        printf 'ACTIVE worker=%s pid=%s user=%s command=%q\n' \
            "$(hostname)" "$pid" "${owner:-unknown}" "$command_arguments"
    fi
done < <(ps -eo pid=,user=,comm=,args=)

for pid in "${!holders[@]}"; do
    owner=$(ps -o user= -p "$pid" | xargs || true)
    arguments=$(ps -o args= -p "$pid" || true)
    printf 'HOLDER worker=%s pid=%s user=%s command=%q\n' \
        "$(hostname)" "$pid" "${owner:-unknown}" "$arguments"
done
if [[ -e /tmp/libtpu_lockfile && "$lockfile_holder_found" == 0 ]]; then
    metadata=$(sudo -n stat -c 'owner=%U uid=%u mode=%a size=%s modified=%y' \
        /tmp/libtpu_lockfile 2>/dev/null || true)
    printf 'STALE_LOCK worker=%s %s\n' "$(hostname)" "$metadata"
    exit 2
fi
if (( ${#holders[@]} > 0 || active_python > 0 )); then
    printf 'OCCUPIED worker=%s\n' "$(hostname)"
    exit 2
fi
printf 'AVAILABLE worker=%s\n' "$(hostname)"
REMOTE_COMMAND

if ! inspection_output=$(gcloud compute tpus tpu-vm ssh "$tpu_identifier" \
    "${gcloud_ssh_flags[@]}" --project="$project_identifier" --zone="$tpu_zone" \
    --worker=all --command="$inspect_command"); then
    printf '%s\n' "$inspection_output"
    if grep -Eq 'HOLDER worker=|ACTIVE worker=|STALE_LOCK worker=|OCCUPIED worker=' \
        <<<"$inspection_output"; then
        exit 2
    fi
    printf 'UNKNOWN TPU_ID=%s ZONE=%s reason=ssh_or_inspection_failed\n' \
        "$tpu_identifier" "$tpu_zone"
    exit 4
fi
printf '%s\n' "$inspection_output"
if grep -Eq 'HOLDER worker=|ACTIVE worker=|STALE_LOCK worker=|OCCUPIED worker=' \
    <<<"$inspection_output"; then
    exit 2
fi
if grep -q 'UNKNOWN worker=' <<<"$inspection_output"; then
    exit 4
fi
