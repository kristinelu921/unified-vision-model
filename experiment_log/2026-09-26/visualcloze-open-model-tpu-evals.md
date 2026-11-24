# VisualCloze open-model TPU evaluations

Date: 2026-09-26 UTC

## Scope

Run the LaVin-DiT, LVM, and Painter open-weight baselines against the existing
frozen VisualCloze episodes. Keep every model's launch, adapter, audit, and
scoring assets inside its existing `eval_runs/2026-09-26/<model>/` directory.

- Test: `test_s64_px256_v5`, pack ID beginning `48e0624`, 960 queries.
- Validation: `val_s32_px256_lineageA`, pack ID beginning `9e77a9`, 128
  queries.
- Conditions: `real`, `scrambled`, `grey`, and `oracle`.
- Demonstrations: `P=2`, same-source draws.
- Episode seed: `[42, task_seed(task), q, P]`.
- Test workload: 3,840 episodes.
- Validation workload: 512 episodes.

## TPU discovery and assignments

Read-only discovery command:

```bash
ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group
```

The command exited successfully. Only the final `AVAILABLE` table was used.
No resources were provisioned.

- LVM: `kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`,
  `us-west4-a`, `v5litepod-8`, `READY`, `HEALTHY`, Spot, occupancy
  `AVAILABLE`.
- Painter: `kmh-tpuvm-v5e-4-spot-kristine-24-05-32-08-461592794-0-us-west4-a`,
  `us-west4-a`, `v5litepod-4`, `READY`, `HEALTHY`, Spot, occupancy
  `AVAILABLE`.
- LaVin-DiT: queued behind Painter on the same exact authorized v5e-4. Painter
  released VFIO and the occupancy helper returned `AVAILABLE` before LaVin-DiT
  began any remote setup or TPU work.

Every workstream must repeat the exact occupancy check immediately before
launch. It must use a real TPU backend, us-west4-local data/assets, and a
faithful open-weight implementation. CUDA-only runners and the stopped L4 VMs
are not permitted fallbacks for this attempt.

## Status

- Painter: terminal and scored. The TPU-backed runner produced 3,840/3,840
  test predictions and 512/512 validation predictions, then released the
  v5e-4 naturally. Exact prediction counts and final manifests were audited.
  Test scoring contains 3,840 unique rows / 60 task-condition cells, SHA-256
  `60a117ae644d3f5b05edeb41758452f41eb4e9e9d2564feeb8432c3c6a1da104`;
  validation scoring contains 512 rows / 16 cells, SHA-256
  `93039314a4fa6dd12abf6b7fa1af33e6b8c454f20cfafc3ae5d8ff7e2e14a26a`.
  Local bundle: `eval_runs/2026-09-26/painter/tpu_v5e4/`. Durable west4 root:
  `gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-26/painter/tpu_v5e4/`.
- LVM: live on the assigned v5e-8. Exact open weights were converted to the
  official EasyLM layout with a byte-exact inverse round-trip for all 291
  tensors / 6,544,199,680 parameters. The exact 2023 JAX 0.4.13 runtime passed
  CPU model/VQ preflight but the host rejected TPU initialization with the IRQ
  `NORESIZE` error before backend/model load; that attempt is preserved. The
  host-supported JAX 0.6.2 compatibility path passed an eight-device TPU
  backend check. A 1,536-token attention chunk was validated against dense
  attention (maximum absolute error `2.384e-7`) before launch. The first
  durable prediction, `relative_depth__q00__P2__real`, committed at 19:03:07Z.
  The first full attempt stopped closed after 337 test episodes when inherited
  text EOS ID 2 (also a valid VQ code) caused the remaining image positions to
  be padded with invalid visual ID 8192. The resumed runner binds suppressed
  special ID 8192 as an unreachable EOS sentinel and always generates exactly
  256 codes in the valid range 0..8191. CPU preflight passed, all eight TPU
  devices were reverified, the failed episode regenerated successfully, and
  the run advanced beyond 337 at about seven seconds per episode. It continues
  through all 3,840 test episodes followed by 512 validation episodes. Local bundle:
  `eval_runs/2026-09-26/lvm/tpu_v5e8/`. Durable west4 root:
  `gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-26/lvm/tpu_v5e8/`.
- LaVin-DiT: full TPU-native evaluation is live on the exact authorized
  Kristine v5e-4. On 2026-09-27 the user explicitly accepted backend-specific
  RNG streams, so this run is approved as a TPU-native baseline and does not
  claim CUDA/TPU numerical equivalence. No L4 or other GPU was used. Both
  filtered inventories, both exact no-claim occupancy gates, and the
  READY/HEALTHY/Spot/type checks passed before launch. Real TPU preflight
  proved four addressable devices and finite Pallas FlashAttention output at
  head dimension 72. The strict STVAE loader found zero missing keys and
  ignored exactly 146 checkpoint tensors, all under the enumerated
  training-only LPIPS/discriminator prefixes. The first full-run validation
  episode (`depth__q00__P2__real`) committed atomically in 1,662.31 seconds at
  2026-09-27 04:27:45Z; both raw512 and pred256 images plus the partial
  manifest are durable in west4 GCS. The sequence is 512 validation episodes
  followed by 3,840 test episodes, preserving all four frozen conditions.
  Local bundle: `eval_runs/2026-09-26/lavin_dit/tpu_queue/`. Durable west4
  root: `gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-26/lavin_dit/tpu_queue/`.

Results, exact commands, output paths, hashes, and terminal status will be
appended without replacing this initial record.
