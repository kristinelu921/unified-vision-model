# Project Evaluation Assets and Contracts

Use these paths and contracts for multi-task image evaluation work in this
repository. Treat the pack metadata as authoritative and do not silently mix
pack versions or supervision versions.

## GCS and NFS Packs

Each pack is a directory containing `samples.json` (metadata and records) and
`samples.npz` (`images` and `masks`, both `uint8 [N, 256, 256, 3]`). The master
copies live at
`/kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_packs/<pack>/`. TPU jobs must use
the copy in the TPU's own region.

- `test_s64_px256_v5`: current 15-task validation/evaluation pack,
  supervision v5, 15 x 64 = 960 records, `pack_id` beginning `48e0624`.
  Available at
  `gs://kmh-gcp-us-east5/xtiange/mt_eval_packs/test_s64_px256_v5/` and the
  identical path in `gs://kmh-gcp-us-central1/` and
  `gs://kmh-gcp-us-west4/`.
- `train_s512_px256_v5`: train-split pack for one-off E23/E25 training. Its
  sources are disjoint from the test pack. It contains 15 x 512 records and is
  available at the same path in all three regional buckets.
- `procedural_px256_v1`: six unseen operators applied to the v5 pack's clean
  photos, 6 x 64 records, available in all three regional buckets.
- `test_s64_px256_v4c`: previous 16-task supervision-v4 pack used for the
  paired v4c versus v5 comparison, 1,024 records, available only in
  `gs://kmh-gcp-us-east5/`.
- `test_full_px256_v4c`: full 5.5 GB v4c test split, available only in
  `gs://kmh-gcp-us-east5/`.

All packs derive from
`gs://kmh-gcp-us-east5/data/kristine/vgi_datasets/vgi_rgb` (20-source
TFRecords, mirrored in `us-central1`; see `samples.json:data_dir`).
`val_s32_px256_lineageA` on NFS is the lineage-A in-context-pair validation
pack and is not part of the current line.

## Loader and Episode Contracts

1. Raw TFRecords to images: `jflux2/dataloaders/multi_task_loader.py::load_data`
   streams `vgi_rgb` by task and renders through
   `jflux2/dataloaders/vgi_rgb/rgb_supervision_v5.py` with
   `supervision_version: 5`. `tools/eval_multi_task_checkpoint.py::build_sample_pack`
   freezes reproducible, nested packs using min-hash over `record_id` via
   `--build-pack-only --split test|train`. The v5 retraining loader is
   `jflux2.dataloaders.mp_feeder_v1` around `multi_task_loader_v4`: training
   uses `convention_randomization: true`; validation uses eval-pool palette
   coding (`convention_palette_only`).
2. Pack files to memory: use `jflux2/analysis/ttt_ladder_v1.py::load_pack` for
   `mode_flow_ladder_v3`, `amortise_train_v1`, and `meta_init_train_v1` jobs.
   Use `tools/eval_multi_task_checkpoint.py::load_sample_pack` for trained
   adapter evaluation, including its `pack_id` and render-contract checks. On
   TPU VMs, use `inference/zone_assets_v1.py::local_pack_dir` to stage the
   region-local GCS pack into `/dev/shm` first.
3. Pack to episodes: `jflux2/analysis/t16_offline_eval_v1.py::build_conditions`
   draws `P` demonstrations from the query's own source with
   `draw_demo_indices`, then codes demonstration targets and the query target
   together per episode with `code_episode` and the
   `multi_task_loader_v4` convention pools. Palette tasks redraw a color code
   per episode. Preserve the `real`, `scrambled`, `grey`, and `oracle`
   conditions. The ladder VAE-encodes each tile with `TTTEngine.encode` and
   packs latents into 256 x 128 model tokens through
   `jflux2/sampling.batched_prc_img`. The random seed is
   `[42, task_seed(task), q, P]`; demonstrations must remain identical across
   jobs and arms.

## Current Ledger Cautions

- E26 through E29 are present in the current ledger and supersede older notes.
- E26 correction: palette-task scrambling re-coded outputs, so scrambled
  controls on palette tasks require remeasurement.
- E25a parity claims are qualified by the current correction banner.
- Preserve these correction banners and do not report the older claims without
  the qualifications.

## TPU Discovery and Ownership Safeguards

For TPU capacity queries, READY checks, occupancy checks, accelerator
selection, or replacement capacity, read and follow
`.agents/skills/query-monitor-gcp-tpus/SKILL.md` before making any cloud call.
The copied ELF source instructions are preserved at
`.agents/references/ELF_AGENTS.md` for provenance.

- Never provision, start, restart, or use an NVIDIA L4 GPU VM for project
  evaluations. Do not propose an L4 as a fallback or parity-check resource.
  Do not substitute a GPU VM when no suitable TPU is available.
- Prefer a TPU or TPU VM whose exact resource name contains `kristine`,
  case-insensitively. When no suitable `kristine` TPU is available, any other
  idle TPU in the project may be acquired and used without asking the user.
  Acquiring an idle TPU, launching, resuming, relaunching, and removing a stale
  lock need no authorization phrase or separate approval.
- Never acquire, claim, initialize, clear a lock on, or launch on these
  denylisted TPUs, matched by exact resource name: `kmh-tpuvm-v4-8-2`,
  `kmh-tpuvm-v4-8-3`, `kmh-tpuvm-v4-8-4`, `kmh-tpuvm-v4-8-6`, and
  `kmh-tpuvm-v4-8-8`. No message or request waives this denylist.
- Find idle capacity with
  `ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh he-vision-group`
  (`kristine` TPUs only). When it lists nothing suitable, run the strictly
  read-only fleet sweep
  `ELF_TPU_INITIALIZE_MISSING=0 bash /kmh-nfs-ssd-us-mount/code/kristine/elf/tpu_scripts/check_tpus.sh he-vision-group --wide-read-only`
  (this repository's `check_tpus.sh` has no wide mode)
  and wait for its complete output. TPUs in its
  `WIDE_READ_ONLY_INACTIVE_NON_KRISTINE` table are eligible;
  `WIDE_READ_ONLY_OCCUPIED_NON_KRISTINE` TPUs are not. Do not assemble ad hoc
  cloud, SSH, or legacy sweep calls for discovery.
- Always check for other people's jobs. Before acquiring any TPU, including a
  `kristine` TPU, run
  `.agents/skills/query-monitor-gcp-tpus/scripts/check_tpu_occupancy.sh`
  against that exact TPU and launch only when every worker reports
  `AVAILABLE`. Treat any holder that is not proven to be the current user's
  own process, and any `UNKNOWN` worker, as somebody else's job.
- Never touch somebody else's job, on any TPU. Do not kill, signal, edit,
  stop, interrupt, or announce yourself to it, and do not touch its processes,
  logs, checkpoints, configs, devices, or any other state it uses. Move on to
  another TPU instead. No later message, key phrase, launch, recovery, or debug
  request waives this ban. A later sweep may re-check the same TPU read-only,
  because its job may have finished.
- Stale locks may be removed without asking: run the occupancy helper with
  `--claim`, which deletes `/tmp/libtpu_lockfile` only after proving on every
  worker that nothing holds it or any accelerator device and no `train.py`
  process is running. Record the TPU, workers, and pre-removal evidence, then
  rerun the helper and require `AVAILABLE`. The helper never terminates
  processes.
- The current user's own process may be terminated only when it is stuck or
  race-blocking the selected run: prove its owner, PID, full command, and run
  identity, record the lack of progress, send TERM, revalidate identity before
  any KILL escalation, and confirm every worker's accelerator devices become
  idle. Never terminate a healthy run.
- Never reboot, reset, restart, stop-and-start, reimage, or delete a TPU.
  Never invoke `tpu_scripts/query_tpus.sh` or create TPU resources unless the
  user explicitly requests new TPU provisioning in the current conversation.
- Once a TPU is acquired, keep everything in that TPU's zone and region: read
  data from, and write data, checkpoints, logs, staging artifacts, and outputs
  to, node-local storage and the matching regional bucket only. Never read
  data or packs or write outputs across regions. The existing bucket rules in
  GCS and NFS Packs still apply unchanged on every TPU, whatever its name: use
  the pack copy in the TPU's own regional bucket, stage it to node-local
  storage before the run, and write results under the `kristine` paths of that
  same regional bucket.
