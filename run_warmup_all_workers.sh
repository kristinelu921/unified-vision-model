#!/usr/bin/env bash
# Wrapper script to run run_warmup.sh on all TPU workers in parallel
# Usage: ./run_warmup_all_workers.sh <VM_NAME> <ZONE> <FROM> <TO>
# Example: ./run_warmup_all_workers.sh kmh-tpuvm-v5e-32-spot-kristine1 us-central1-a gs://bucket/data /mnt/klum/data

# Don't exit on error - we want to see what went wrong
set -uo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <VM_NAME> <ZONE> <FROM> <TO>"
    echo "Example: $0 kmh-tpuvm-v5e-32-spot-kristine1 us-central1-a gs://bucket/data /mnt/klum/data"
    exit 1
fi

VM_NAME=$1
ZONE=$2
FROM=$3
TO=$4


# HARD ZONE CHECKS
if [[ "$ZONE" == *"asia"* ]] && [[ "$FROM" != *"asia"* ]]; then
    echo "Error: Zone ($ZONE) contains 'asia' but FROM path ($FROM) does not contain 'asia'."
    exit 1
fi

if [[ "$ZONE" == *"east-5"* ]] && [[ "$FROM" != *"east-5"* ]]; then
    echo "Error: Zone ($ZONE) contains 'east-5' but FROM path ($FROM) does not contain 'east-5'."
    exit 1
fi

if [[ "$ZONE" == *"east-1"* ]] && [[ "$FROM" != *"east-1"* ]]; then
    echo "Error: Zone ($ZONE) contains 'east-1' but FROM path ($FROM) does not contain 'east-1'."
    exit 1
fi

if [[ "$ZONE" == *"central"* ]] && [[ "$FROM" != *"central"* ]]; then
    echo "Error: Zone ($ZONE) contains 'central' but FROM path ($FROM) does not contain 'central'."
    exit 1
fi


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WARMUP_SCRIPT="$SCRIPT_DIR/run_warmup.sh"
LOG_FILE="/tmp/warmup_${VM_NAME}_$(date +%Y%m%d_%H%M%S).log"
REMOTE_SCRIPT="/tmp/run_warmup.sh"

echo "Running warmup on all workers of $VM_NAME in zone $ZONE"
echo "FROM: $FROM"
echo "TO: $TO"
echo "Log file: $LOG_FILE"
echo ""

# Remove 'set -e' so errors don't immediately exit - we want to see the full output
# Use 'set +e' explicitly to allow commands to fail without exiting
# Run on all workers in parallel using --worker=all

gcloud compute tpus tpu-vm scp $WARMUP_SCRIPT "${VM_NAME}:${REMOTE_SCRIPT}" --zone "$ZONE" --worker=all

if [ $? -ne 0 ]; then
    echo "Error: Failed to scp $WARMUP_SCRIPT to $VM_NAME"
    exit 1
fi

echo "Successfully scp'd $WARMUP_SCRIPT to $VM_NAME"

gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --worker=all \
    --ssh-flag="-o ServerAliveInterval=60" \
    --ssh-flag="-o ServerAliveCountMax=10" \
    --ssh-flag="-o ConnectTimeout=30" \
    --ssh-flag="-o TCPKeepAlive=yes" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --command "
set +e
set -u
set -o pipefail
cd /tmp || { echo '[ERROR] Failed to cd to /tmp'; exit 1; }
echo \"[worker] Starting warmup script on \$(hostname)...\"
# Run the script and capture exit code
bash $REMOTE_SCRIPT '$FROM' '$TO'
EXIT_CODE=\$?
# Always print success/error message before exiting
# This helps even if connection drops right after
if [ \$EXIT_CODE -ne 0 ]; then
    echo '[ERROR] Worker \$(hostname) failed with exit code \$EXIT_CODE' >&2
    exit \$EXIT_CODE
else
    echo '[SUCCESS] Worker \$(hostname) completed successfully'
    # Small delay to ensure message is sent before connection closes
    sleep 0.5
    exit 0
fi
" 2>&1 | tee -a "$LOG_FILE"

# Capture exit code from gcloud command (before tee)
GCLOUD_EXIT=${PIPESTATUS[0]}

echo ""
# Check the actual output for success/error messages rather than relying on gcloud exit code
# gcloud --worker=all can sometimes return non-zero even when all workers succeed
NUM_SUCCESS=$(grep -c "\[SUCCESS\]" "$LOG_FILE" 2>/dev/null || echo "0")
NUM_ERRORS=$(grep -c "\[ERROR\]" "$LOG_FILE" 2>/dev/null || echo "0")

if [ $NUM_ERRORS -gt 0 ]; then
    echo "❌ Warmup failed - $NUM_ERRORS worker(s) reported errors"
    echo "Check the log file for details: $LOG_FILE"
    exit 1
elif [ $NUM_SUCCESS -gt 0 ]; then
    echo "✅ Warmup completed successfully on $NUM_SUCCESS worker(s)!"
    exit 0
else
    # Fallback to gcloud exit code if we can't determine from output
    if [ $GCLOUD_EXIT -eq 0 ]; then
        echo "✅ Warmup completed (gcloud reported success)"
        exit 0
    else
        echo "⚠️  Unable to determine success/failure from output"
        echo "gcloud exit code: $GCLOUD_EXIT"
        echo "Check the log file: $LOG_FILE"
        exit $GCLOUD_EXIT
    fi
fi