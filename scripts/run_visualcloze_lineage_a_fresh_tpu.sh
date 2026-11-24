#!/usr/bin/env bash
set -euo pipefail

ROOT=/dev/shm/visualcloze_eval
REPO=/kmh-nfs-ssd-us-mount/code/kristine/lvm

export VISUALCLOZE_ROOT="$ROOT"
export VISUALCLOZE_SKIP_TPU_CHECK=1

bash "$REPO/scripts/setup_visualcloze_tpu.sh"

export HF_HOME="$ROOT/hf-home"
export HF_XET_HIGH_PERFORMANCE=1

"$ROOT/venv/bin/python" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('VisualCloze/VisualClozePipeline-384', revision='59c469d2772d927ffe55f3543c4d3bd556fd46a4', cache_dir='$ROOT/hf')"

exec bash "$REPO/scripts/run_visualcloze_lineage_a_tpu.sh"
