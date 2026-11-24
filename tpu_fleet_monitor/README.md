# Project-wide TPU fleet monitor

This is a separate, strictly read-only dashboard for Cloud TPU control-plane
metadata in `he-vision-group`. It does not replace or modify the Kristine ELF
run monitor on port 8782.

The fleet monitor shows:

- every TPU VM returned by the project-wide TPU VM list API;
- every Cloud TPU Queued Resource across all zones;
- state, health, accelerator type, topology chip count, scheduling mode, zone,
  creation time, and safe failure diagnostics;
- a saved one-time `Busy`, `Idle`, or `Unknown` occupancy snapshot when a user
  has explicitly authorized that scan;
- filters for name, owner label, shape, state, and zone.

## Safety contract

The server has an explicit command allowlist containing only:

```text
gcloud compute tpus tpu-vm list
gcloud compute tpus queued-resources list
```

The web service never SSHs, runs guest commands, modifies firewalls, or exposes
create, delete, reset, start, stop, update, or other lifecycle controls. The
queued-resource command projects only safe fields and omits node metadata, SSH
keys, service-account details, IP addresses, and network config. The HTTP
server rejects non-loopback binding.

Guest occupancy is handled by a separate, manually invoked scanner. It is not
reachable from HTTP and is never scheduled. Each invocation requires a literal
explicit-permission argument. Its audited remote command only reads VFIO
holders and active `train.py` processes; it has no signal, cleanup, file-write,
package-install, firewall, or lifecycle code path. The scanner uses direct
system SSH with the existing key, a single connection attempt, and disabled
known-host writes. It has no key-upload mechanism and cannot change project or
instance metadata. A TPU on which the existing key is not already authorized
is reported as `Unknown`. SSH and API access may still appear in normal
connection or audit logs.

## Start

From the repository root:

```bash
bash tpu_fleet_monitor/start.sh
```

The detached dashboard is available at <http://127.0.0.1:8783>. Its PID and log
are stored in `/kmh-nfs-ssd-us-mount/staging/kristine/tpu-fleet-monitor/`.
The launcher is idempotent.

The fleet and queue caches poll independently every 30 seconds. The page reads
the cached API and never initiates a cloud call per browser tab. `Refresh
metadata` only wakes the two control-plane pollers and never performs a guest
scan.

## One-time occupancy snapshot

Run this only after the user has explicitly approved a project-wide guest
inspection for that specific scan:

```bash
python3 -m tpu_fleet_monitor.activity_scan \
  --explicit-permission I_HAVE_EXPLICIT_READ_ONLY_PERMISSION
```

Only TPUs that are `READY` at the beginning of the scan are contacted. The
result is written locally to
`/kmh-nfs-ssd-us-mount/staging/kristine/tpu-fleet-monitor/activity.json` and
the dashboard reads it as a static snapshot. Running the scanner again always
requires new explicit user permission.
