---
name: query-monitor-gcp-tpus
description: Safely find, provision, inspect, and acquire Google Cloud TPU VMs for LVM/ELF-style evaluation. Use for TPU capacity queries, tpu_scripts/query_tpus.sh, READY checks, VFIO occupancy checks, stale-lock cleanup, TPU selection before launch, Spot capacity failures, or replacement capacity. Prefer resources whose names contain kristine, but any idle non-denylisted TPU may be acquired without asking. Never touch another person's job and never use the denylisted v4-8 TPUs.
---

# Query and Claim GCP TPUs

## Keep these barriers

- Prefer a TPU whose exact resource name contains `kristine`, case-insensitively; when none is suitable, acquire any other idle TPU without asking. Acquiring an idle TPU, launching, resuming, relaunching, and removing a stale lock need no authorization phrase or separate approval. Never acquire, claim, initialize, clear a lock on, or launch on these denylisted TPUs, matched by exact resource name: `kmh-tpuvm-v4-8-2`, `kmh-tpuvm-v4-8-3`, `kmh-tpuvm-v4-8-4`, `kmh-tpuvm-v4-8-6`, and `kmh-tpuvm-v4-8-8`. Discovery of non-`kristine` capacity uses only the strictly read-only `check_tpus.sh he-vision-group --wide-read-only` sweep. Always check every worker for other people's jobs with the occupancy helper before acquiring a TPU. Never touch somebody else's job, on any TPU: do not kill, signal, edit, stop, interrupt, or announce yourself to it, and do not touch its processes, logs, checkpoints, configs, or devices; select another TPU instead. Once a TPU is acquired, keep everything in that TPU's zone and region: data, checkpoints, logs, staging artifacts, and outputs stay on node-local storage and the matching regional bucket, under the existing regional `kristine` bucket rules.
- Never reboot, reset, restart, stop-and-start, or reimage a TPU or its guest OS. If the kernel is stuck, leave the TPU untouched and choose another idle TPU.
- Standing user authorization permits ending a current-user ELF process only when it is verified stuck or race-blocking the selected run: prove the exact TPU, owner, PID, full command, and run identity; record lack of progress; send TERM; revalidate before KILL escalation; and confirm all workers' accelerator resources become idle. Never use this exception for foreign, unknown, non-ELF, or healthy processes.
- Standing user authorization permits clearing a holderless `/tmp/libtpu_lockfile` on any non-denylisted TPU without asking. Use only the occupancy helper's `--claim` mode: it removes the lockfile only after every worker shows no holder of any `/dev/vfio/*`, `/dev/accel*`, or lockfile, no running `train.py`, and no `UNKNOWN` worker; it rechecks on each worker before removal and reinspects afterward. It never removes another file or signals a process.

## Find existing capacity; provision only on explicit request

Run the reconciler in read-only discovery mode. It applies the server-side owner filter, checks current state, health, occupancy, and runtime readiness, and prints workload-agnostic candidates in its final `AVAILABLE` table without initializing missing runtimes:

```bash
ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group
```

The checker streams one row after each TPU finishes its probes. Never select
from partial output: wait for the command to exit successfully, then parse only
the final `AVAILABLE` section. If the user supplies an exact
`NAME,ZONE` pair that Cloud Asset Search has not indexed yet, pass it through
`ELF_TPU_EXACT_CANDIDATES`; the checker will apply the same state, health,
occupancy, and runtime probes before including it in the final table.

Select candidates from the final `AVAILABLE` table, or from the wide sweep's `WIDE_READ_ONLY_INACTIVE_NON_KRISTINE` table when no `kristine` TPU fits. Never select a denylisted TPU. Do not treat `ELF_RUNTIME_STATUS=MISSING` or `UNKNOWN` as launch-ready; initialize and revalidate only the one selected idle TPU. Do not use `scripts/sweep_tpus.sh` for new discovery workflows.

If no suitable ID is returned, do not ask the user for a TPU name. Run the wide
sweep below; if it also has nothing suitable, keep rerunning both audits on the
monitoring cadence. For an active v6e run recovering from confirmed TPU loss,
an existing `AVAILABLE` v5p-8, v5p-16, or v5p-32 is an approved fallback when
no matching v6e is available. Keep the same active-run count and Spot policy;
require the checkpoint, dataset, and future outputs to remain in the
replacement's region. Treat it as a new retry and reapply compatibility, HBM,
and throughput gates before launch.

Never invoke `tpu_scripts/query_tpus.sh` unless the user explicitly requests new
TPU provisioning in the current conversation. A generic launch, resume,
relocation, recovery, or persistence request does not authorize provisioning.
Only after that explicit request, route the requested accelerator type and
topology to:

```bash
bash tpu_scripts/query_tpus.sh TYPE SIZE 8
```

Allow these type and size combinations:

- `v5e`: `8`, `16`, `32`, or `64`.
- `v5p` and `v4`: `8`, `16`, or `32`.
- `v6e`: `4`, `8`, `16`, `32`, `64`, `128`, or `256`.

The script restricts creation attempts to GCP-surveyed zones with an advertised
shape, positive effective Spot quota, and a matching regional bucket service
account: v5e uses `asia-northeast1-b`, `us-west4-a`, or `us-central1-a`; v5p
uses `us-central1-a` or `us-east5-a`; v6e uses `us-central1-b`,
`us-central1-a`, `us-east5-a`, `asia-northeast1-b`, `us-east5-b`, or
`us-central1-c`; and v4 uses only `us-central2-b`. An explicit
`ZONES_OVERRIDE` must be a subset of that family allowlist or the script exits
before any cloud request. Use the type and size required by the run, as well as
any above it that will allow the run.

The 2026-09-23 read-only survey established these slice-to-machine counts:

| requested type | TPU VMs per requested TPU |
| --- | ---: |
| v5e-8 / 16 / 32 / 64 | 1 / 2 / 4 / 8 |
| v5p-8 / 16 / 32 | 1 / 2 / 4 |
| v6e-4 / 8 / 16 / 32 / 64 / 128 / 256 | 1 / 2 / 4 / 8 / 16 / 32 / 64 |
| v4-8 / 16 / 32 | 1 / 2 / 4 |

The survey used accelerator-type and topology metadata, effective per-zone
quota limits, aggregate Cloud Monitoring quota usage, bucket locations, and
`testIamPermissions`. It did not create or queue capacity and did not inspect a
non-`kristine` TPU. GCP does not expose live Spot physical capacity through a
read-only API, so a cataloged zone can still return `RESOURCE_EXHAUSTED`.


## Wide read-only sweep of the TPU fleet

Use the filtered `kristine` inventory first. When it has no suitable TPU, run
ELF's shared sweep program for the rest of the fleet (this repository's
`check_tpus.sh` has no wide mode); never assemble ad hoc asset,
describe, SSH, occupancy-helper, or legacy `sweep_tpus.sh` calls for discovery:

```bash
ELF_TPU_INITIALIZE_MISSING=0 \
  bash /kmh-nfs-ssd-us-mount/code/kristine/elf/tpu_scripts/check_tpus.sh he-vision-group --wide-read-only
```

Wait for successful completion and use its complete output. The program probes
all `kristine` candidates first and keeps them in the final `AVAILABLE` table,
then performs one read-only occupancy fan-out for each READY, HEALTHY
non-`kristine` TPU. Denylisted TPUs are reported as `DENYLISTED` and never
probed. Idle foreign TPUs appear under `WIDE_READ_ONLY_INACTIVE_NON_KRISTINE`
and are eligible without asking; TPUs under
`WIDE_READ_ONLY_OCCUPIED_NON_KRISTINE` are running somebody else's job and stay
untouched. The mode disables runtime initialization and deep runtime checks
regardless of inherited environment.

The sweep itself is strictly read-only: do not write, signal, lock, copy,
reboot, reset, reimage, start a workload, or leave a session connected while
sweeping. When a read shows an active holder, do not announce it or touch that
job's processes, logs, checkpoints, or devices. A later sweep may re-check the
same TPU read-only, because its job may have finished.

## Focus on the selected TPU ID

Once an ID is selected, stop broad discovery. Validate that its exact name is not denylisted, then describe only that ID to confirm its current state, health, type, topology, zone, and Spot/preemptible policy.

Before launch, check every worker for other people's jobs with the occupancy helper. It accepts any non-denylisted TPU and never signals a process:

```bash
bash .agents/skills/query-monitor-gcp-tpus/scripts/check_tpu_occupancy.sh \
    he-vision-group TPU_ID ZONE
```

- `AVAILABLE` (exit 0): use the TPU.
- `OCCUPIED` (exit 2): a process holds `/dev/vfio/*`, `/dev/accel*`, or the lockfile, or a `train.py` is running. `FOREIGN` lines are somebody else's job: leave it untouched and select another TPU. `OWN` lines are the current user's processes: leave a healthy run alone, or apply the verified stuck/race exception above.
- `STALE_LOCK` (exit 3): the only blocker is a holderless `/tmp/libtpu_lockfile`. Rerun the same command with `--claim` to remove it, and launch only when the helper finishes with `AVAILABLE`.
- `UNKNOWN` (exit 4): do not touch the TPU.

For any occupancy result, record owner, command, and run identity. If a stale lock is cleared, record the exact TPU, workers, lock metadata, and post-removal occupancy result. If the holder is the active selected ELF run, leave it running and monitor it rather than launching a duplicate.

Once acquired, keep all data, checkpoints, logs, and outputs in that TPU's zone and region, under the existing regional `kristine` bucket rules.

## Report

Report the exact TPU ID, zone, type/topology, state, health, Spot flag, occupancy result, and any capacity failure. Prove readiness from current API state rather than query-script output alone.
