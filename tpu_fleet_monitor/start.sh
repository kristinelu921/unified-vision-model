#!/usr/bin/env bash
set -euo pipefail

# Start the project-wide, metadata-only TPU fleet monitor as a detached process.
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
state_directory=${TPU_FLEET_MONITOR_STATE_DIRECTORY:-/kmh-nfs-ssd-us-mount/staging/kristine/tpu-fleet-monitor}
python_executable=${TPU_FLEET_MONITOR_PYTHON:-$(command -v python3)}
dashboard_host=${TPU_FLEET_MONITOR_HOST:-127.0.0.1}
dashboard_port=${TPU_FLEET_MONITOR_PORT:-8783}
current_hostname=$(hostname)
pid_path="${state_directory}/server.pid"
log_path="${state_directory}/server.log"
dashboard_url="http://${dashboard_host}:${dashboard_port}"

[[ "$dashboard_host" == 127.0.0.1 || "$dashboard_host" == localhost || "$dashboard_host" == ::1 ]] \
    || { printf 'ERROR: the fleet monitor may bind only to loopback\n' >&2; exit 2; }
[[ "$dashboard_port" =~ ^[0-9]+$ ]] \
    || { printf 'ERROR: dashboard port must be numeric\n' >&2; exit 2; }
[[ -x "$python_executable" ]] \
    || { printf 'ERROR: Python executable is unavailable: %s\n' "$python_executable" >&2; exit 1; }

if curl --fail --silent --show-error --max-time 2 \
        "${dashboard_url}/api/status" >/dev/null 2>&1; then
    printf 'TPU_FLEET_MONITOR=ALREADY_RUNNING url=%s\n' "$dashboard_url"
    exit 0
fi

mkdir -p "$state_directory"
if [[ -f "$pid_path" ]]; then
    read -r previous_hostname previous_process_identifier < "$pid_path" || true
    if [[ "$previous_hostname" == "$current_hostname" \
            && "$previous_process_identifier" =~ ^[0-9]+$ ]] \
            && kill -0 "$previous_process_identifier" 2>/dev/null; then
        printf 'ERROR: monitor PID %s is alive but %s did not answer\n' \
            "$previous_process_identifier" "$dashboard_url" >&2
        exit 1
    fi
fi

cd "$repository_root"
nohup setsid "$python_executable" -m tpu_fleet_monitor.server \
    --host "$dashboard_host" --port "$dashboard_port" \
    </dev/null >>"$log_path" 2>&1 &
monitor_process_identifier=$!
printf '%s %s\n' "$current_hostname" "$monitor_process_identifier" > "$pid_path"

for _ in {1..24}; do
    if curl --fail --silent --show-error --max-time 2 \
            "${dashboard_url}/api/status" >/dev/null 2>&1; then
        printf 'TPU_FLEET_MONITOR=STARTED pid=%s url=%s log=%s\n' \
            "$monitor_process_identifier" "$dashboard_url" "$log_path"
        exit 0
    fi
    if ! kill -0 "$monitor_process_identifier" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

printf 'ERROR: TPU fleet monitor did not become ready; inspect %s\n' "$log_path" >&2
exit 1
