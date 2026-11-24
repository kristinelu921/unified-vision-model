#!/usr/bin/env bash
set -euo pipefail

export PJRT_DEVICE=TPU
export HF_HOME=/dev/shm/visualcloze_eval/hf-home
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

exec /dev/shm/visualcloze_eval/venv/bin/python \
  /kmh-nfs-ssd-us-mount/code/kristine/lvm/evals/visualcloze_v5_eval.py infer \
  --run-dir /kmh-nfs-ssd-us-mount/code/kristine/lvm/logs/visualcloze_v5_full_20260923 \
  --cache-dir /dev/shm/visualcloze_eval \
  --steps 30 \
  --batch-size 2 \
  --flash-block-size 1024 \
  --local-files-only \
  --resume
