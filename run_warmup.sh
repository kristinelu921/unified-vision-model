#!/usr/bin/env bash
# warm_ramds_all.sh
# Purpose: On all TPU workers in parallel: install dependencies → fetch GCS parts → stream-decompress → extract into /dev/shm/ds/current
# Requirements: gcloud already authenticated; compute_node.sh provides VM_NAME / ZONE

set -euo pipefail

# Workaround for OpenSSL compatibility issue with gsutil
export CLOUDSDK_PYTHON_SITEPACKAGES=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# source "$SCRIPT_DIR/compute_node.sh"   # Must define: VM_NAME, ZONE

FROM=$1
TO=$2

# Example settings:
# FROM=gs://kmh-gcp-us-east1/hanhong/data/us_vae_cached_muvar_imagenet_zhh/vae_cached_muvar_imagenet_zhh
# TO=/dev/shm/zhh/latents

# FROM=gs://kmh-gcp-us-east1/hanhong/data/imagenet_parts/imagenet
# TO=/dev/shm/zhh/imagenet

# DEST="/dev/shm/zhh"                               # In-memory target directory
# DEST="data"                                        # In-memory target directory
# If your part files are PREFIX.tar.zst.part.aa, ab, ac... keep the following pattern:
PART_GLOB="$FROM"
# To cap /dev/shm size, set a value (e.g., "300G"); leave empty to keep default
REMOUNT_SHM_SIZE="450G"
DEDICATED_MOUNT="/mnt/klum"
LOOP_FILE="/dev/shm/data_450GB.img"

# Clear any existing loop mount (idempotent - safe if nothing exists) (UNCOMMENT IF WANT TO REUSE TPU, MOUNT ALREADY USED)
#sudo umount "$DEDICATED_MOUNT" 2>/dev/null || true
#LOOP_DEV=$(losetup -j "$LOOP_FILE" 2>/dev/null | cut -d: -f1)
#[ -n "$LOOP_DEV" ] && sudo losetup -d "$LOOP_DEV" 2>/dev/null || true
#sudo rm -f "$LOOP_FILE"

echo "[worker] $(hostname): starting prep..."

sudo mount -o remount,size="$REMOUNT_SHM_SIZE" /dev/shm || true

if [ ! -d "$TO" ] || [ -z "$(ls -a "$TO" 2>/dev/null)" ]; then
  sudo mkdir -p "$DEDICATED_MOUNT"

  if ! mountpoint -q "$DEDICATED_MOUNT"; then
    echo "[worker] Creating 450GB space instantly with fallocate..."
    sudo fallocate -l 450G "$LOOP_FILE" || sudo truncate -s 450G "$LOOP_FILE"

    LOOP_DEV=$(sudo losetup -f)
    sudo losetup "$LOOP_DEV" "$LOOP_FILE"
    sudo mkfs.ext4 -F "$LOOP_DEV" >/dev/null
    sudo mount "$LOOP_DEV" "$DEDICATED_MOUNT"
  fi

  TO="$DEDICATED_MOUNT/data"
  sudo mkdir -p "$TO"

fi

mkdir -p /dev/shm/tmp_data
echo "[worker] Listing files in $FROM..."
gcloud storage ls "${FROM}" 2>&1 | head -20 || echo "[worker] Warning: Could not list files"
echo "[worker] Downloading from $FROM..."

gcloud storage cp -r --no-clobber "${FROM}*" /dev/shm/tmp_data/

echo "[worker] Staging files to $TO..."

sudo rsync -a --remove-source-files /dev/shm/tmp_data/ "$TO"/
sudo rm -rf /dev/shm/tmp_data

sudo chmod -R 555 "$TO"
echo "[worker] Setup complete"
