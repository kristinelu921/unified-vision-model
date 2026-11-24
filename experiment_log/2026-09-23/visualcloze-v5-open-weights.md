# VisualCloze open-weight v5-pack gate

Status: completed the first gate on
`kmh-tpuvm-v5e-8-spot-kristine-02-08-02-17-N-us-west4-a`.

## Frozen inputs

- Pack: `test_s64_px256_v5`
- Pack ID: `48e062452574ff0ded61c72664445d687ecd1c02`
- Episode builder: live `jflux2.analysis.t16_offline_eval_v1`, conventions set
  to `multi_task_loader_v4`, same-source demonstrations, seed
  `[42, task_seed(task), q, P]`.
- Initial tasks: `edge_detection`, `relative_depth`; query 0; P=2.
- Conditions: `real`, corrected `scrambled`, and `oracle`.
- Model: `VisualCloze/VisualClozePipeline-384`, revision
  `59c469d2772d927ffe55f3543c4d3bd556fd46a4`.
- Diffusers: revision `0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec`.
- Generation: stage 1 only, 384px, 30 steps, guidance 30, model seed 0.
- Scoring copy: 384px output resized to 256px with Lanczos; raw 384px output
  retained. Scorer is `jflux2.fasteval.metrics_v3.score_pair_v3`.

This is a compatibility and signal gate, not a headline comparison. It must
retain every prediction and report the copy-input floor and perfect-target
ceiling. Broader coverage is conditional on this gate successfully compiling
and producing non-degenerate images.

## Outcome

The gate completed at 16:20 UTC. The first 30-step XLA compile plus inference
took 1,481.6 seconds; the last four cache-hot episodes each took 9.1 seconds.
All raw 384px predictions and 256px scoring copies are present under
`logs/visualcloze_v5_open_20260923/predictions/`.

| task | metric | real | scrambled | oracle | copy input | perfect |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| edge_detection | ois_f | 0.8225 | 0.8179 | 0.8880 | 0.2795 | 1.0000 |
| relative_depth | delta1_aligned_ssc | 0.4945 | 0.5495 | 0.7914 | 0.2129 | 1.0000 |

Visual inspection passed: the edge outputs are edge maps of the query bird,
and the depth outputs preserve the query-room geometry. Both real arms beat
the copy-input floor and both oracle arms improve on real. On this single query
per task, real does not cleanly beat scrambled: edge differs by only +0.0046,
and scrambled depth is +0.0551 above real. Therefore this gate establishes
runtime compatibility, task following, and an oracle positive-control signal;
it does not establish demo-pair sensitivity or a dataset-level quality result.

The live jflux2 HEAD moved from `29570d6b731e346b42c360356cb3278df8c9b7c0`
during episode preparation to `a4a56cf980c70202f865ad753660b3687270f98a`
during scoring. The four relevant source files had no diff across those
revisions and were clean; their SHA-256 hashes are frozen in `manifest.json`.
