#!/usr/bin/env bash
set -euo pipefail

kind="${1:?usage: run_d3_300k_build.sh depth|normal}"
repo="/kmh-nfs-ssd-us-mount/code/kristine/lvm"
python_env="/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310"
scratch="$repo/tmp_tfds_scratch/${kind}"
log_dir="$repo/tmp_tfds_build_logs"
mkdir -p "$scratch" "$log_dir"

export TMPDIR="$scratch"
export D3_OMNIDATA_ROOT="/mnt/klum/omindata"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

cd "$repo"

case "$kind" in
  depth)
    builder="d3/d3_depth_300k.py"
    data_dir="/kmh-nfs-ssd-us-mount/code/kristine/lvm/tfds/d3_depth_300k_tfds"
    ;;
  normal)
    builder="d3/d3_normal_300k.py"
    data_dir="/kmh-nfs-ssd-us-mount/code/kristine/lvm/tfds/d3_normal_300k_tfds"
    ;;
  *)
    echo "unknown kind: $kind" >&2
    exit 2
    ;;
esac

exec "$python_env/bin/tfds" build "$builder" --data_dir="$data_dir"
