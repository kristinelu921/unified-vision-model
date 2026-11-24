# VisualCloze open-weight full v5-pack evaluation

Status: completed on
`kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`.

## Frozen inputs and runtime

- Pack: `test_s64_px256_v5`, pack ID
  `48e062452574ff0ded61c72664445d687ecd1c02`.
- Coverage: 15 tasks x 64 queries x four conditions (`real`, corrected
  `scrambled`, `grey`, `oracle`) = 3,840 episodes.
- Episode construction: live `jflux2.analysis.t16_offline_eval_v1`, P=2,
  same-source demonstrations, joint episode coding, seed
  `[42, task_seed(task), q, P]`. Palette-task scrambled controls use the
  corrected current coding path; this is not the superseded E26 control.
- Model: `VisualCloze/VisualClozePipeline-384`, revision
  `59c469d2772d927ffe55f3543c4d3bd556fd46a4`.
- Diffusers: revision `0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec`.
- Generation: stage 1, 384 px, 30 steps, guidance 30, model seed 0.
  The 384 px output was retained and a 256 px Lanczos copy was scored.
- TPU execution: batch size 2 on a 2 x 4 data/model mesh. The original
  FlashAttention tile 1,536 exceeded v5e scoped VMEM at batch size 2; tile
  1,024 passed the full gate and was recorded in `inference.json`.
- Timing: 2026-09-23 21:31:55 UTC to 2026-09-24 04:49:04 UTC. The 1,920
  batches used 6.374 device-hours in total and averaged 11.7935 seconds after
  cache warmup.

## Integrity

- 3,840 manifest episodes and 3,840 unique inference results.
- 7,680 prediction PNGs: raw 384 px plus scored 256 px for every episode.
- 11,520 SHA-256 checks passed: every episode NPZ and both output PNGs for
  every result.
- Exact condition counts: 960 each for `real`, `scrambled`, `grey`, and
  `oracle`; exact task counts: 256 episodes for each of 15 tasks.
- Both score reports contain 3,840 unique rows and 60 task-condition cells.
  There are no active partial inference or score journals. The archived
  37-episode batch-one capacity trial is retained separately and is not
  included in either final report.

## Primary results

The decision-primary table below follows the current ladder contract:
LPIPS-AlexNet for the five appearance tasks and the v3 task metric for all
others. `scores.json` preserves the distortion-primary PSNR view; the table is
from `scores_perceptual.json`. Every cell is n=64 except face-landmark NME,
whose finite counts are real 36, scrambled 47, grey 8, and oracle 32.

| task | primary metric | real | scrambled | grey | oracle | direction |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| colorization | lpips_alex | 0.3011 | 0.3172 | 0.9187 | 0.1267 | lower |
| deblurring | lpips_alex | 0.1779 | 0.1772 | 0.8749 | 0.1384 | lower |
| denoising | lpips_alex | 0.2617 | 0.2804 | 0.8019 | 0.2098 | lower |
| edge_detection | ois_f | 0.7177 | 0.6741 | 0.2056 | 0.7419 | higher |
| face_landmarks | nme_bbox_matched | 0.4605 | 0.4393 | 1.1548 | 0.4808 | lower |
| human_pose_2d | pck_bbox_reject | 0.0156 | 0.0063 | 0.0000 | 0.0234 | higher |
| inpainting | lpips_alex | 0.1206 | 0.1233 | 0.8606 | 0.0720 | lower |
| instance_segmentation | pq_novoid | 0.0643 | 0.0638 | 0.0050 | 0.2387 | higher |
| object_keypoints | pck_bbox_reject | 0.0156 | 0.0234 | 0.0156 | 0.1130 | higher |
| panoptic_segmentation | pq_novoid | 0.0634 | 0.0630 | 0.0051 | 0.2456 | higher |
| relative_depth | delta1_aligned_ssc | 0.5290 | 0.5336 | 0.2422 | 0.7417 | higher |
| saliency_detection | max_fbeta | 0.5814 | 0.6496 | 0.3058 | 0.9300 | higher |
| semantic_segmentation | miou | 0.0455 | 0.0446 | 0.0117 | 0.3352 | higher |
| super_resolution | lpips_alex | 0.2153 | 0.2014 | 0.8898 | 0.1755 | lower |
| surface_normals | mean_deg | 59.4434 | 59.7481 | 125.4602 | 25.2384 | lower |

## Readout cautions

- `real` beats the copy-input baseline on 12 of 15 task means. The exceptions
  are colorization, human pose, and object keypoints.
- `real` is directionally better than corrected `scrambled` on 9 of 15 task
  means, but these raw mean comparisons are not paired significance tests.
  In particular, semantic, instance, and panoptic real-versus-scrambled gaps
  are very small after the corrected palette-control construction.
- `oracle` is a strong positive control on almost every task; `grey` is a
  strong negative control. Face-landmark NME must be interpreted with its
  differing finite counts rather than as a fully paired n=64 comparison.
- These results do not restore the superseded E25a parity claim and do not
  reuse the invalid pre-correction E26 palette scrambled controls.

Artifacts are under `logs/visualcloze_v5_full_20260923/`: `manifest.json`,
`inference.json`, `scores_perceptual.json`, `scores.json`, `episodes/`, and
`predictions/`.
