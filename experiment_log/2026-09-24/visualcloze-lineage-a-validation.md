# VisualCloze open-weight lineage-A validation

Status: completed on
`kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`.

## Frozen inputs and runtime

- Pack: `val_s32_px256_lineageA`, pack ID
  `9e77a92f82d0e08c12f540d5c15c2b15e6c503a2`.
- Coverage: four tasks x 32 queries x four conditions (`real`, `scrambled`,
  `grey`, `oracle`) = 512 episodes.
- Scope: this is the lineage-A in-context-pair validation pack. It is separate
  from the current 15-task supervision-v5 line and from the larger raw test
  split. Results must not be pooled with `test_s64_px256_v5`.
- Episode construction: P=2 same-source demonstrations with seed
  `[42, task_seed(task), q, P]`. The native lineage-A masks were preserved;
  no v5 palette coding or CV4 re-coding was applied.
- Model: `VisualCloze/VisualClozePipeline-384`, revision
  `59c469d2772d927ffe55f3543c4d3bd556fd46a4`.
- Diffusers: revision `0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec`.
- Generation: stage 1, 384 px, 30 steps, guidance 30, model seed 0. The raw
  384 px output was retained and a 256 px Lanczos copy was scored.
- TPU execution: batch size 2 on a 2 x 4 data/model mesh with FlashAttention
  tile 1,024.
- Timing: 2026-09-24 19:22:09 UTC to 20:54:21 UTC. The first two batches
  included XLA warm-up (1,866.8 and 139.5 seconds); the remaining batches had
  a median time of 11.8 seconds.

## Integrity

- 512 manifest episodes and 512 unique inference results.
- 1,024 prediction PNGs: raw 384 px plus scored 256 px for every episode.
- 1,536 SHA-256 checks passed: every episode NPZ and both output PNGs for
  every result.
- Exact counts: 128 episodes per task and 128 episodes per condition.
- The score report contains 512 unique rows and 16 task-condition cells, all
  with 32 finite scores.
- Twelve outputs were uniform all-black generations and remain included in
  the reported means: depth real 1, depth oracle 3; segmentation real 2,
  segmentation scrambled 3, and segmentation oracle 3. Excluding them does
  not change the direction of the headline real-versus-copy or
  real-versus-scrambled counts, but it materially improves the affected
  oracle means.

## Native primary results

Every value is the mean followed by its 95% 20,000-sample bootstrap interval.

| task | primary metric | real | scrambled | grey | oracle | copy input | direction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| depth | `mae` | 0.2407 [0.1946, 0.2892] | 0.2510 [0.1970, 0.3092] | 0.3148 [0.2834, 0.3477] | 0.0914 [0.0576, 0.1353] | 0.3699 | lower |
| normal | `mean_deg` | 50.4659 [45.6698, 55.4976] | 50.8511 [46.4312, 55.5475] | 117.4477 [112.8114, 122.2624] | 17.7527 [15.6235, 19.9282] | 94.3164 | lower |
| segmentation | `f1_best` | 0.0428 [0.0219, 0.0666] | 0.0684 [0.0194, 0.1367] | 0.0578 [0.0309, 0.0897] | 0.4404 [0.2971, 0.5843] | 0.0750 | higher |
| panoptic | `pq` | 0.0736 [0.0472, 0.1036] | 0.0656 [0.0416, 0.0932] | 0.0427 [0.0236, 0.0635] | 0.2824 [0.2127, 0.3590] | 0.0617 | higher |

## Readout cautions

- `real` beats the copy-input baseline on 3 of 4 task means and is
  directionally better than `scrambled` on 3 of 4. Binary segmentation is
  the exception to both comparisons.
- `oracle` is the best arm on all four tasks, while `grey` is generally a
  negative control. The all-black failures disproportionately depress the
  depth and segmentation oracle means.
- The real-versus-scrambled statements are directional comparisons of means,
  not paired-significance claims; the bootstrap intervals overlap strongly
  for all four tasks.
- This lineage-A run does not revise the current E26 correction or the
  qualification on E25a. It contains no palette-task scrambling and should
  not be used to reintroduce the superseded v5 parity framing.

Artifacts are under `logs/visualcloze_lineageA_full_20260924/`:
`manifest.json`, `inference.json`, `scores.json`, `episodes/`, `predictions/`,
and `report/`.
