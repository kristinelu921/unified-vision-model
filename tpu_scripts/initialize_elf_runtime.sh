#!/usr/bin/env bash
set -euo pipefail

# Prepare an idle TPU slice for ELF; this script does not stage data or train.
# It validates the target, reuses a complete runtime, installs pinned packages
# when needed, and copies W&B credentials only after every worker is reachable.

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s PROJECT TPU_NAME ZONE\n' "$0" >&2
    exit 2
fi

project_identifier=$1
tpu_identifier=$2
tpu_zone=$3
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
netrc_path=${ELF_NETRC_PATH:-${repository_root}/.netrc}
remote_user=${ELF_REMOTE_USER:-${USER:-$(id -un)}}
initialize_missing_runtime=${ELF_TPU_INITIALIZE_MISSING:-1}
runtime_profile=${ELF_RUNTIME_PROFILE:-jax-0.6.2}

# Keep the proven legacy stack as the default and install newer JAX separately.
case "$runtime_profile" in
    jax-0.6.2)
        runtime_directory_name=runtime
        python_bootstrap="python3.10"
        managed_python_version=3.10
        exact_python_version=
        force_managed_python=false
        uv_version="0.8.3"
        jax_package_spec='jax[tpu]==0.6.2'
        libtpu_package_spec=
        flax_package_spec='flax==0.10.5'
        numpy_package_spec='numpy==1.26.4'
        torch_package_spec='torch==2.2.2+cpu'
        expected_python_minor=10
        expected_jax_version=0.6.2
        expected_flax_version=0.10.5
        expected_numpy_version=1.26.4
        require_freeze=false
        ;;
    jax-0.11.0)
        runtime_directory_name=runtime-jax-0.11.0
        python_bootstrap="python3"
        managed_python_version=3.12.13
        exact_python_version=3.12.13
        force_managed_python=true
        uv_version="0.12.2"
        jax_package_spec='jax[tpu]==0.11.0'
        libtpu_package_spec='libtpu==0.0.44.1'
        flax_package_spec='flax==0.12.8'
        numpy_package_spec='numpy==2.5.1'
        torch_package_spec='torch==2.13.0+cpu'
        expected_python_minor=12
        expected_jax_version=0.11.0
        expected_flax_version=0.12.8
        expected_numpy_version=2.5.1
        require_freeze=true
        ;;
    *)
        printf 'ERROR: unsupported ELF_RUNTIME_PROFILE: %s\n' "$runtime_profile" >&2
        exit 2
        ;;
esac

# Print one consistent diagnostic and stop with the requested status.
die() {
    local exit_status=$1
    shift
    printf 'ERROR: %s\n' "$*" >&2
    exit "$exit_status"
}

# Use the public route while bypassing the control host's broken SSH config.
run_on_all_workers() {
    local remote_command=$1

    gcloud compute tpus tpu-vm ssh "$tpu_identifier" \
        --project="$project_identifier" \
        --zone="$tpu_zone" \
        --worker=all \
        --ssh-flag=-F \
        --ssh-flag=/dev/null \
        --command="$remote_command"
}

# Phase 1: reject unsafe input locally, before the first cloud call.
case "${tpu_identifier,,}" in
    kmh-tpuvm-v4-8-2|kmh-tpuvm-v4-8-3|kmh-tpuvm-v4-8-4|kmh-tpuvm-v4-8-6|kmh-tpuvm-v4-8-8) die 2 "refusing denylisted TPU ID: $tpu_identifier" ;;
esac
[[ "$remote_user" =~ ^[A-Za-z0-9._-]+$ ]] \
    || die 2 "invalid remote user: $remote_user"
[[ -f "$netrc_path" ]] \
    || die 1 "ELF W&B credential file is missing: $netrc_path"
netrc_permissions=$(stat -c '%a' "$netrc_path")
[[ "$netrc_permissions" == 600 || "$netrc_permissions" == 400 ]] \
    || die 1 "ELF W&B credential file must have mode 0600 or 0400: $netrc_path"
python3 -c \
        'import netrc, sys; raise SystemExit(0 if netrc.netrc(sys.argv[1]).authenticators("api.wandb.ai") else 1)' \
        "$netrc_path" \
    || die 1 "ELF credential file has no api.wandb.ai entry: $netrc_path"

current_state=$(gcloud compute tpus tpu-vm describe "$tpu_identifier" \
    --project="$project_identifier" --zone="$tpu_zone" --format='value(state)')
current_health=$(gcloud compute tpus tpu-vm describe "$tpu_identifier" \
    --project="$project_identifier" --zone="$tpu_zone" --format='value(health)')
[[ "$current_state" == READY && "$current_health" == HEALTHY ]] \
    || die 4 "TPU is not READY and HEALTHY: name=$tpu_identifier state=$current_state health=$current_health"
run_on_all_workers true \
    || die 78 'SSH preflight failed on at least one worker; refusing partial initialization'

# Phase 2: a READY marker is valid only when imports and W&B still work.
read -r -d '' runtime_readiness_command <<'REMOTE_COMMAND' || true
set -euo pipefail
current_user=$(id -un)
runtime_root="/home/${current_user}/elf-runs/${runtime_directory_name}"
python_executable="${runtime_root}/venv/bin/python"
if [[ -d "${runtime_root}/INITIALIZING" ]]; then
    printf 'ELF_RUNTIME_INITIALIZING worker=%s\n' "$(hostname)"
    exit 0
fi
runtime_ready=1
test -f "${runtime_root}/READY" || runtime_ready=0
test -x "$python_executable" || runtime_ready=0
if [[ "$require_freeze" == true ]]; then
    test -f "${runtime_root}/requirements.freeze.txt" || runtime_ready=0
    test -f "${runtime_root}/requirements.freeze.sha256" || runtime_ready=0
fi
if (( runtime_ready )) && ! JAX_PLATFORMS=cpu "$python_executable" -c \
        'import datasets, einops, flax, huggingface_hub, jax, mauve, numpy, optax, sacrebleu, sys, torch, tqdm, transformers, wandb, yaml; assert sys.version_info[:2] == (3, int(sys.argv[1])); assert jax.__version__ == sys.argv[2]; assert flax.__version__ == sys.argv[3]; assert numpy.__version__ == sys.argv[4]; assert hasattr(optax.contrib, "muon"); assert wandb.api.api_key' \
        "$expected_python_minor" "$expected_jax_version" "$expected_flax_version" "$expected_numpy_version" \
        >/dev/null 2>&1; then
    runtime_ready=0
fi
if (( runtime_ready )); then
    printf 'ELF_RUNTIME_READY worker=%s profile=%s\n' "$(hostname)" "$runtime_profile"
else
    printf 'ELF_RUNTIME_MISSING worker=%s profile=%s\n' "$(hostname)" "$runtime_profile"
fi
REMOTE_COMMAND

runtime_readiness_command="runtime_profile='${runtime_profile}'
runtime_directory_name='${runtime_directory_name}'
expected_python_minor='${expected_python_minor}'
expected_jax_version='${expected_jax_version}'
expected_flax_version='${expected_flax_version}'
expected_numpy_version='${expected_numpy_version}'
require_freeze='${require_freeze}'
${runtime_readiness_command}"

if ! runtime_readiness_output=$(run_on_all_workers "$runtime_readiness_command"); then
    printf '%s\n' "$runtime_readiness_output" >&2
    die 78 'unable to verify the existing ELF runtime on every worker'
fi
printf '%s\n' "$runtime_readiness_output"
if grep -q '^ELF_RUNTIME_INITIALIZING worker=' <<<"$runtime_readiness_output"; then
    die 75 'ELF runtime initialization is already active on at least one worker'
fi
if grep -q '^ELF_RUNTIME_READY worker=' <<<"$runtime_readiness_output" \
        && ! grep -q '^ELF_RUNTIME_MISSING worker=' <<<"$runtime_readiness_output"; then
    printf 'RUNTIME_INITIALIZATION=REUSED tpu=%s zone=%s\n' \
        "$tpu_identifier" "$tpu_zone"
    exit 0
fi
if ! grep -q '^ELF_RUNTIME_MISSING worker=' <<<"$runtime_readiness_output"; then
    printf 'RUNTIME_READINESS=UNKNOWN action=initialize\n'
fi
[[ "$initialize_missing_runtime" != 0 ]] \
    || die 1 'ELF runtime is missing and ELF_TPU_INITIALIZE_MISSING=0'

# Phase 3: install the pinned runtime concurrently, but only on idle workers.
read -r -d '' initialization_command <<'REMOTE_COMMAND' || true
set -euo pipefail

# Abort rather than touching a TPU after any process has acquired VFIO.
if ! command -v fuser >/dev/null || ! sudo -n true 2>/dev/null; then
    printf 'ERROR: VFIO inspection unavailable on %s\n' "$(hostname)" >&2
    exit 21
fi
if ! compgen -G '/dev/vfio/*' >/dev/null; then
    printf 'ERROR: no VFIO devices found on %s\n' "$(hostname)" >&2
    exit 21
fi
accelerator_resources=(/dev/vfio/*)
if [[ -e /tmp/libtpu_lockfile ]]; then
    accelerator_resources+=(/tmp/libtpu_lockfile)
fi
for accelerator_resource in "${accelerator_resources[@]}"; do
    [[ -e "$accelerator_resource" ]] || continue
    if sudo -n fuser "$accelerator_resource" >/dev/null 2>&1; then
        printf 'OCCUPIED worker=%s resource=%s\n' "$(hostname)" "$accelerator_resource" >&2
        exit 20
    fi
done

# Fresh images may still be replacing the user package directory. Wait until
# that stock initialization is finished before creating our isolated venv.
for ((attempt = 1; attempt <= 120; attempt++)); do
    if pgrep -f '[p]ip install --no-index --find-links=wheels|/[h]ome/sqa/.disk_mounted|[w]heels.tar.gz' >/dev/null; then
        if [[ "$attempt" -eq 120 ]]; then
            printf 'ERROR: stock TPU initialization timed out on %s\n' "$(hostname)" >&2
            exit 1
        fi
        sleep 15
    else
        break
    fi
done

current_user=$(id -un)
runtime_root="/home/${current_user}/elf-runs/${runtime_directory_name}"
virtual_environment="${runtime_root}/venv"
mkdir -p "$runtime_root"
initialization_lock="${runtime_root}/INITIALIZING"
if ! mkdir "$initialization_lock" 2>/dev/null; then
    printf 'ALREADY_INITIALIZING worker=%s\n' "$(hostname)" >&2
    exit 75
fi
trap 'rmdir "$initialization_lock" 2>/dev/null || true' EXIT
mountpoint -q /
test -w "/home/${current_user}"
# Use the image's Python only for the legacy profile; newer JAX stays isolated.
use_managed_python=$force_managed_python
if [[ "$use_managed_python" != true ]] \
        && { ! "$python_bootstrap" -c 'import sys; assert sys.version_info[:2] == (3, 10)' 2>/dev/null \
            || ! "$python_bootstrap" -m venv "$virtual_environment"; }; then
    use_managed_python=true
fi
if [[ "$use_managed_python" == true ]]; then
    uv_install_directory="${runtime_root}/uv"
    uv_installer="${runtime_root}/uv-installer.sh"
    curl --fail --location --silent --show-error \
        "https://astral.sh/uv/${uv_version}/install.sh" \
        --output "$uv_installer"
    UV_INSTALL_DIR="$uv_install_directory" UV_NO_MODIFY_PATH=1 sh "$uv_installer"
    rm -f "$uv_installer"
    "${uv_install_directory}/uv" venv \
        --allow-existing \
        --seed \
        --python "$managed_python_version" \
        --python-preference only-managed \
        "$virtual_environment"
fi
python_executable="${virtual_environment}/bin/python"
if [[ -n "$exact_python_version" ]]; then
    "$python_executable" -c \
        "import platform; assert platform.python_version() == '${exact_python_version}'"
fi
# TPU boot disks are small; do not retain multi-gigabyte wheel caches after setup.
export PIP_NO_CACHE_DIR=1
"$python_executable" -m pip install --upgrade pip setuptools wheel
if [[ "$runtime_profile" == jax-0.6.2 ]]; then
    "$python_executable" -m pip install \
        "$jax_package_spec" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
    "$python_executable" -m pip install "$jax_package_spec" "$libtpu_package_spec"
fi
"$python_executable" -m pip install \
    "$flax_package_spec" \
    optax==0.2.8 \
    "$numpy_package_spec" \
    datasets==5.0.1 \
    einops==0.8.2 \
    huggingface_hub==0.36.2 \
    pyyaml==6.0.3 \
    rouge-score==0.1.2 \
    sacrebleu==2.6.0 \
    tqdm==4.70.0 \
    transformers==4.57.6 \
    wandb==0.28.1
"$python_executable" -m pip install \
    "$torch_package_spec" \
    --index-url https://download.pytorch.org/whl/cpu
"$python_executable" -m pip install mauve-text==0.4.0

# Validate imports on CPU so initialization does not acquire the TPU device.
JAX_PLATFORMS=cpu "$python_executable" -c \
    'import datasets, einops, flax, huggingface_hub, jax, mauve, numpy, optax, rouge_score, sacrebleu, torch, tqdm, transformers, wandb, yaml'
"$python_executable" -c 'import optax; assert hasattr(optax.contrib, "muon")'
"$python_executable" -m pip check
"$python_executable" -m pip freeze --all > "${runtime_root}/requirements.freeze.txt"
sha256sum "${runtime_root}/requirements.freeze.txt" \
    > "${runtime_root}/requirements.freeze.sha256"
if gcloud storage ls --help >/dev/null 2>&1 \
        && gcloud storage cp --help >/dev/null 2>&1 \
        && gcloud storage rm --help >/dev/null 2>&1; then
    printf 'GCS_CLI=gcloud-storage\n'
elif command -v gsutil >/dev/null 2>&1; then
    printf 'GCS_CLI=gsutil\n'
else
    printf 'ERROR: no supported GCS CLI is installed on %s\n' "$(hostname)" >&2
    exit 1
fi
printf 'RUNTIME_INSTALLED worker=%s runtime=%s freeze_sha256=%s\n' \
    "$(hostname)" "$virtual_environment" \
    "$(awk '{print $1}' "${runtime_root}/requirements.freeze.sha256")"
REMOTE_COMMAND

initialization_command="runtime_profile='${runtime_profile}'
runtime_directory_name='${runtime_directory_name}'
python_bootstrap='${python_bootstrap}'
managed_python_version='${managed_python_version}'
exact_python_version='${exact_python_version}'
force_managed_python='${force_managed_python}'
uv_version='${uv_version}'
jax_package_spec='${jax_package_spec}'
libtpu_package_spec='${libtpu_package_spec}'
flax_package_spec='${flax_package_spec}'
numpy_package_spec='${numpy_package_spec}'
torch_package_spec='${torch_package_spec}'
${initialization_command}"

if initialization_output=$(run_on_all_workers "$initialization_command"); then
    printf '%s\n' "$initialization_output"
else
    initialization_status=$?
    printf '%s\n' "$initialization_output" >&2
    exit "$initialization_status"
fi
runtime_freeze_hashes=$(sed -n \
    's/.*RUNTIME_INSTALLED .* freeze_sha256=\([0-9a-f]\{64\}\).*/\1/p' \
    <<< "$initialization_output")
[[ -n "$runtime_freeze_hashes" ]] \
    || die 1 'no runtime freeze hash was returned by worker initialization'
(( $(sort -u <<< "$runtime_freeze_hashes" | wc -l) == 1 )) \
    || die 1 "installed runtime differs across TPU workers: ${runtime_freeze_hashes}"
printf 'RUNTIME_FREEZE_CONSENSUS=OK sha256=%s\n' \
    "$(head -n 1 <<< "$runtime_freeze_hashes")"

# Phase 4: copy W&B credentials separately, validate them, then write READY.
# The temporary upload is mode 0600 so SCP never exposes the API key.
credential_upload_name=".elf-netrc-upload-$(date -u +'%Y%m%dT%H%M%SZ')-$$"
credential_upload_path="/home/${remote_user}/${credential_upload_name}"
run_on_all_workers \
    "umask 077; : > '${credential_upload_path}'; chmod 600 '${credential_upload_path}'"
gcloud compute tpus tpu-vm scp "$netrc_path" \
    "${tpu_identifier}:${credential_upload_path}" \
    --project="$project_identifier" \
    --zone="$tpu_zone" \
    --scp-flag=-F \
    --scp-flag=/dev/null \
    --scp-flag=-O \
    --worker=all

finalization_command="set -euo pipefail
credential_upload_path='${credential_upload_path}'
credential_path='/home/${remote_user}/.netrc'
runtime_root='/home/${remote_user}/elf-runs/${runtime_directory_name}'
python_executable=\"\${runtime_root}/venv/bin/python\"
if test -f \"\${credential_upload_path}\"; then
    chmod 600 \"\${credential_upload_path}\"
    mv -f \"\${credential_upload_path}\" \"\${credential_path}\"
fi
test -f \"\${credential_path}\"
test \"\$(stat -c '%a' \"\${credential_path}\")\" = 600
\"\${python_executable}\" -c 'import wandb; assert wandb.api.api_key'
date -u +'%Y-%m-%dT%H:%M:%SZ' > \"\${runtime_root}/READY\"
printf 'INITIALIZED_WITH_WANDB worker=%s runtime=%s\\n' \"\$(hostname)\" \"\${runtime_root}/venv\""
run_on_all_workers "$finalization_command"
