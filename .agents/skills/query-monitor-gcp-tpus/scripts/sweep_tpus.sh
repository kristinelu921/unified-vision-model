#!/usr/bin/env bash
set -euo pipefail

project_identifier=${1:-he-vision-group}
readonly server_side_name_query='name:kristine'

asset_inventory=$(
    gcloud asset search-all-resources \
        --scope="projects/${project_identifier}" \
        --asset-types='tpu.googleapis.com/Node' \
        --query="${server_side_name_query}" \
        --format='csv[no-heading](location,name)'
)

declare -a tpu_identifiers=()
declare -a tpu_zones=()
while IFS=, read -r tpu_zone full_resource_name; do
    [[ -z "${tpu_zone}" || -z "${full_resource_name}" ]] && continue
    tpu_identifier=${full_resource_name##*/}
    if [[ "${tpu_identifier,,}" != *kristine* ]]; then
        printf 'ERROR: server-side filter returned unauthorized TPU ID: %s\n' \
            "${tpu_identifier}" >&2
        exit 1
    fi
    tpu_identifiers+=("${tpu_identifier}")
    tpu_zones+=("${tpu_zone}")
done <<<"${asset_inventory}"

for index in "${!tpu_identifiers[@]}"; do
    tpu_identifier=${tpu_identifiers[$index]}
    tpu_zone=${tpu_zones[$index]}
    current_state=$(
        gcloud compute tpus tpu-vm describe "${tpu_identifier}" \
            --project="${project_identifier}" \
            --zone="${tpu_zone}" \
            --format='value(state)'
    )
    if [[ "${current_state}" == 'READY' ]]; then
        printf '%s\t%s\n' "${tpu_identifier}" "${tpu_zone}"
    fi
done
