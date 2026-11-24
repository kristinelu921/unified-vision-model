#!/usr/bin/env bash
set -euo pipefail

# Launch the three requested checkpoint validations on named TPU VMs.
# The script deliberately refuses to start an online W&B run until the
# copied netrc contains a current (40+ character) API key.

ROOT=/kmh-nfs-ssd-us-mount/code/kristine/lvm
NETRC=/kmh-nfs-ssd-us-mount/code/kristine/elf/.netrc
CFG=$ROOT/experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_100k_wandb_lr22_resume2000_fullshuffle/optimize_concise_prompt_init_100k_wandb_lr22_resume2000_fullshuffle.yaml
TRAINER=$ROOT/experiments/flux2_cv_prompt_probe/trainer_optimize_text_token_only_prompt_init.py
SNAPDIR=$ROOT/experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_100k_wandb_lr22_resume2000_fullshuffle/embedding_snapshots

if [[ ! -r "$NETRC" ]]; then
  echo "Missing W&B netrc: $NETRC" >&2; exit 2
fi
key_len=$(awk '$1=="password"{print length($2)}' "$NETRC")
if [[ -z "$key_len" || "$key_len" -lt 40 ]]; then
  echo "W&B netrc key is missing or too short (found ${key_len:-0}; need 40+)." >&2
  exit 2
fi

copy_common() {
  local vm=$1 zone=$2
  gcloud compute tpus tpu-vm scp --zone="$zone" "$NETRC" "$vm":~/.netrc --quiet
  gcloud compute tpus tpu-vm ssh "$vm" --zone="$zone" --command \
    "chmod 600 ~/.netrc; python -m pip install -q einops safetensors; mkdir -p /tmp/vgi_validation" --quiet
}

launch_validation() {
  local vm=$1 zone=$2 step=$3
  local snap="$SNAPDIR/task_embeddings_microstep_$(printf '%09d' "$step").npz"
  copy_common "$vm" "$zone"
  gcloud compute tpus tpu-vm scp --zone="$zone" "$CFG" "$vm":/tmp/vgi_validation/config.yaml --quiet
  gcloud compute tpus tpu-vm scp --zone="$zone" "$snap" "$vm":/tmp/vgi_validation/task_tokens.npz --quiet
  gcloud compute tpus tpu-vm ssh "$vm" --zone="$zone" --command \
    "cd $ROOT && sed -i '0,/split: train/s//split: val/' /tmp/vgi_validation/config.yaml && \
     JAX_PLATFORMS=tpu PYTHONPATH=$ROOT:$ROOT/jflux python -u $TRAINER \
     --config /tmp/vgi_validation/config.yaml --task_init_path /tmp/vgi_validation/task_tokens.npz \
     --resume_workdir '' --run_name vgi_validation_${step} --local_output_dir /tmp/vgi_validation/out_${step} --workdir /tmp/vgi_validation/work_${step}/vgi_validation_${step} --max_steps $step --learning_rate 0 --validation_every 1 \
     --wandb --wandb_mode online --wandb_run_name vgi_validation_${step} \
     > /tmp/validation_${step}.log 2>&1 &" --quiet
}

usage() { echo "Usage: $0 3000|6000|9000 VM ZONE"; }
[[ $# -eq 3 ]] || { usage >&2; exit 2; }
case "$1" in 3000|6000|9000) ;; *) usage >&2; exit 2 ;; esac
launch_validation "$2" "$3" "$1"
echo "Launched checkpoint $1 validation on $2 ($3); inspect /tmp/validation_$1.log."
