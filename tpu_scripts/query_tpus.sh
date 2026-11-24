#!/usr/bin/env bash
set -euo pipefail

# Create exactly the requested number of owner-scoped Spot TPU VMs. Each worker
# rotates through approved zones until creation succeeds, then initializes ELF.
if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: %s TYPE CHIPS [COUNT]\n' "$0" >&2
    exit 2
fi

accelerator_type=$1
chip_count=$2
requested_count=${3:-8}
retry_delay_seconds=${RETRY_DELAY_SECONDS:-10}
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_identifier=${ELF_GCP_PROJECT:-he-vision-group}
initializer=${ELF_TPU_INITIALIZER:-${script_directory}/initialize_elf_runtime.sh}

[[ "$accelerator_type" =~ ^(v5e|v5p|v6e|v4)$ ]] \
    || { printf 'Type must be v5e, v5p, v6e, or v4\n'; exit 1; }
if [[ "$accelerator_type" == v5e ]]; then
    [[ "$chip_count" =~ ^(4|8|16|32|64)$ ]] \
        || { printf 'v5e amount must be 4, 8, 16, 32, or 64\n'; exit 1; }
elif [[ "$accelerator_type" == v6e ]]; then
    [[ "$chip_count" =~ ^(4|8|16|32|64|128|256)$ ]] \
        || { printf 'v6e amount must be 4, 8, 16, 32, 64, 128, or 256\n'; exit 1; }
else
    [[ "$chip_count" =~ ^(8|16|32)$ ]] \
        || { printf '%s amount must be 8, 16, or 32\n' "$accelerator_type"; exit 1; }
fi
[[ "$requested_count" =~ ^[1-9][0-9]*$ ]] \
    || { printf 'COUNT must be a positive integer\n' >&2; exit 1; }
[[ -x "$initializer" ]] \
    || { printf 'ELF initializer is missing: %s\n' "$initializer" >&2; exit 1; }

# Keep creation attempts inside zones whose live catalog supports each family.
supported_zones_for_accelerator() {
    case "$1" in
        v5e) printf '%s\n' \
            'us-west4-b us-east5-a us-east5-c us-east5-b us-west4-a us-central1-a' ;;
        v5p) printf '%s\n' 'us-central1-a us-east5-a' ;;
        v6e) printf '%s\n' \
            'us-central1-b us-central1-a us-east5-a asia-northeast1-b us-east5-b us-central1-c' ;;
        v4) printf '%s\n' 'us-central2-b' ;;
    esac
}

read -r -a supported_zones <<<"$(supported_zones_for_accelerator "$accelerator_type")"
if [[ -n ${ZONES_OVERRIDE:-} ]]; then
    read -r -a query_zones <<<"$ZONES_OVERRIDE"
    if (( ${#query_zones[@]} == 0 )); then
        printf 'ERROR: ZONES_OVERRIDE must contain at least one zone\n' >&2
        exit 2
    fi
    for query_zone in "${query_zones[@]}"; do
        zone_is_supported=0
        for supported_zone in "${supported_zones[@]}"; do
            if [[ "$query_zone" == "$supported_zone" ]]; then
                zone_is_supported=1
                break
            fi
        done
        if [[ "$zone_is_supported" == 0 ]]; then
            printf 'ERROR: zone %s does not support %s; allowed zones: %s\n' \
                "$query_zone" "$accelerator_type" "${supported_zones[*]}" >&2
            exit 2
        fi
    done
else
    query_zones=("${supported_zones[@]}")
fi

# Region-local service accounts prevent accidental cross-region dataset access.
bucket_service_account() {
    local zone=$1
    case "${zone%-*}" in
        us-east1) printf 'bucket-ruler@he-vision-group.iam.gserviceaccount.com\n' ;;
        us-east5) printf 'bucket-us-east5@he-vision-group.iam.gserviceaccount.com\n' ;;
        asia-northeast1) printf 'bucket-asia@he-vision-group.iam.gserviceaccount.com\n' ;;
        us-central1) printf 'bucket-us-central1@he-vision-group.iam.gserviceaccount.com\n' ;;
        us-central2) printf 'bucket-us-central2@he-vision-group.iam.gserviceaccount.com\n' ;;
        us-west4) printf 'bucket-us-west4@he-vision-group.iam.gserviceaccount.com\n' ;;
        *) return 1 ;;
    esac
}

runtime_version() {
    case "$1" in
        v5e) printf 'v2-alpha-tpuv5-lite\n' ;;
        v5p) printf 'v2-alpha-tpuv5\n' ;;
        v6e) printf 'v2-alpha-tpuv6e\n' ;;
        v4) printf 'tpu-ubuntu2204-base\n' ;;
    esac
}

accelerator_resource_type() {
    if [[ "$accelerator_type" == v5e ]]; then
        printf 'v5litepod-%s\n' "$chip_count"
    else
        printf '%s-%s\n' "$accelerator_type" "$chip_count"
    fi
}

machine_count_per_tpu() {
    case "$accelerator_type" in
        v6e) printf '%s\n' "$((chip_count / 4))" ;;
        v5e) printf '%s\n' "$(((chip_count + 7) / 8))" ;;
        v5p|v4) printf '%s\n' "$((chip_count / 8))" ;;
    esac
}

resource_type=$(accelerator_resource_type)
machines_per_tpu=$(machine_count_per_tpu)
total_machine_count=$((requested_count * machines_per_tpu))

results_directory=$(mktemp -d)
trap 'rm -rf -- "$results_directory"' EXIT

# One worker can create at most one TPU, so parallel retries never exceed COUNT.
create_one_tpu() {
    local worker_index=$1
    local zone_count=${#query_zones[@]}
    local offset zone service_account tpu_name create_output
    local blocked_zone_count=0
    local -A blocked_zones=()

    while true; do
        for ((offset = 0; offset < zone_count; offset++)); do
            zone=${query_zones[$(((worker_index + offset) % zone_count))]}
            if [[ -n ${blocked_zones[$zone]+blocked} ]]; then
                continue
            fi
            if ! service_account=$(bucket_service_account "$zone"); then
                printf '[%s] No regional bucket service account; skipping\n' "$zone" >&2
                continue
            fi
            tpu_name="kmh-tpuvm-${accelerator_type}-${chip_count}-spot-kristine-$(date -u +'%d-%H-%M-%S-%N')-${worker_index}-${zone}"
            printf '[%s] Creating %s\n' "$zone" "$tpu_name"
            if ! create_output=$(gcloud compute tpus tpu-vm create "$tpu_name" \
                    --zone="$zone" \
                    --accelerator-type="$resource_type" \
                    --version="$(runtime_version "$accelerator_type")" \
                    --spot \
                    --service-account="$service_account" \
                    --project="$project_identifier" \
                    --scopes='https://www.googleapis.com/auth/cloud-platform' 2>&1); then
                printf '%s\n' "$create_output" >&2
                case "$create_output" in
                    *"Reservation not found"*|*"Failed to perform tenant project creation"*|*"PERMISSION_DENIED"*|*"does not have permission to submit requests"*|*"Quota limit"*"has been exceeded"*)
                        blocked_zones["$zone"]=1
                        blocked_zone_count=$((blocked_zone_count + 1))
                        printf '[%s] Non-retryable create blocker for this request; disabling zone\n' \
                            "$zone" >&2
                        ;;
                esac
                continue
            fi
            printf '%s\n' "$create_output"

            # A created TPU counts toward capacity even if initialization fails.
            if ! bash "$initializer" "$project_identifier" "$tpu_name" "$zone"; then
                printf '%s:%s:INITIALIZATION_FAILED\n' "$zone" "$tpu_name" \
                    >"${results_directory}/${worker_index}.result"
                return 1
            fi
            printf '%s:%s:READY\n' "$zone" "$tpu_name" \
                >"${results_directory}/${worker_index}.result"
            curl -fsS -d "TPU ${tpu_name} in ${zone} ready" \
                https://ntfy.sh/KMH-TPU-QUERIES >/dev/null || true
            return 0
        done
        if (( blocked_zone_count == zone_count )); then
            printf 'none:none:BLOCKED\n' >"${results_directory}/${worker_index}.result"
            return 1
        fi
        sleep "$retry_delay_seconds"
    done
}

# Create independent bounded workers and report every resulting resource.
printf 'Creating %s TPU(s): machine_type=%s machines_per_tpu=%s total_machines=%s zones=%s\n' \
    "$requested_count" "$resource_type" "$machines_per_tpu" \
    "$total_machine_count" "${#query_zones[@]}"
worker_pids=()
for ((worker_index = 0; worker_index < requested_count; worker_index++)); do
    create_one_tpu "$worker_index" &
    worker_pids+=("$!")
done

failed_workers=0
for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid" || failed_workers=$((failed_workers + 1))
done

printf '\nTPU_RESULTS\n'
for result_path in "${results_directory}"/*.result; do
    [[ -f "$result_path" ]] && cat "$result_path"
done
(( failed_workers == 0 ))
