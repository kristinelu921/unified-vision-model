#!/usr/bin/env bash
# Read-only capacity watcher for the Lavin-DiT recovery.  It never initializes,
# locks, or otherwise modifies a TPU; it only alerts when a v5e-16/32 is free.
set -euo pipefail

readonly PROJECT=he-vision-group
readonly INTERVAL_SECONDS="${LAVIN_TPU_MONITOR_INTERVAL_SECONDS:-300}"
readonly STATE_FILE="${LAVIN_TPU_MONITOR_STATE_FILE:-/tmp/lavin_v5e_capacity_alerted.txt}"
readonly LOG_FILE="${LAVIN_TPU_MONITOR_LOG_FILE:-/tmp/lavin_v5e_capacity_monitor.log}"
readonly WIDE_CHECKER=/kmh-nfs-ssd-us-mount/code/kristine/elf/tpu_scripts/check_tpus.sh

mkdir -p "$(dirname "$STATE_FILE")" "$(dirname "$LOG_FILE")"
touch "$STATE_FILE"

alert_once() {
  local candidate=$1
  if grep -Fxq "$candidate" "$STATE_FILE"; then
    return
  fi
  printf '%s\n' "$candidate" >>"$STATE_FILE"
  local message="Lavin-DiT TPU capacity: eligible free v5e candidate ${candidate}. Verify occupancy immediately before launch."
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$message" >>"$LOG_FILE"
  curl -fsS -d "$message" https://ntfy.sh/KMH-TPU-QUERIES >/dev/null || true
}

while true; do
  local_output=$(mktemp)
  wide_output=$(mktemp)
  trap 'rm -f "$local_output" "$wide_output"' EXIT

  ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh "$PROJECT" \
    >"$local_output" 2>&1 || true
  ELF_TPU_INITIALIZE_MISSING=0 bash "$WIDE_CHECKER" "$PROJECT" --wide-read-only \
    >"$wide_output" 2>&1 || true

  # Local checker candidates are already all-worker checked.  The wide sweep's
  # inactive table is likewise read-only occupancy-verified by its contract.
  while IFS= read -r row; do
    [[ -n "$row" ]] && alert_once "$row"
  done < <(
    {
      awk '$1 ~ /v5e-(16|32)-/ && $3 == "AVAILABLE" {print $1 " " $2}' "$local_output"
      awk '$1 ~ /v5e-(16|32)-/ && $3 == "INACTIVE_READ_ONLY" {print $1 " " $2}' "$wide_output"
    } | sort -u
  )
  cat "$local_output" "$wide_output" >>"$LOG_FILE"
  rm -f "$local_output" "$wide_output"
  trap - EXIT
  sleep "$INTERVAL_SECONDS"
done
