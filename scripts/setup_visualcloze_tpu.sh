#!/usr/bin/env bash
set -euo pipefail

ROOT="${VISUALCLOZE_ROOT:-/dev/shm/visualcloze_eval}"
VENV="$ROOT/venv"
mkdir -p "$ROOT/tmp" "$ROOT/pip-cache" "$ROOT/hf"

python3 -m virtualenv "$VENV"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/pip-cache"

"$VENV/bin/pip" install --no-cache-dir torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install --no-cache-dir 'torch_xla[tpu]==2.7.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
"$VENV/bin/pip" install --no-cache-dir --pre 'torch_xla[pallas]==2.7.0' \
  --index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html
"$VENV/bin/pip" install --no-cache-dir \
  'git+https://github.com/huggingface/diffusers.git@0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec' \
  transformers accelerate sentencepiece structlog pillow scipy

if [[ "${VISUALCLOZE_SKIP_TPU_CHECK:-0}" != "1" ]]; then
  PJRT_DEVICE=TPU "$VENV/bin/python" -c \
    'import torch, torch_xla, torch_xla.runtime as xr; from torch_xla.experimental.custom_kernel import FlashAttention; print(torch.__version__, torch_xla.__version__, xr.global_runtime_device_count(), FlashAttention.__name__)'
fi
