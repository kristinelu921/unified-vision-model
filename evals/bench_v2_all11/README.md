# Bench v2 — all 11 baseline evaluation runners

Cleaned snapshot of the **selected** evaluation scripts used for the eleven Bench v2 baselines.
Extracted from `reports/baseline_implementations_2026-11-19/baseline_implementation_sources.zip`
(the same archive that backs the implementation reference report).

Heavy caches, completion receipts, site `dist/` score boards, and qualification-launch helpers were dropped.
Recipe JSON (`plan_val.json`, `defaults.json`, `validation_plan.json`, task prompts) is kept.

## Models → entrypoints

| Baseline | Entrypoints |
|---|---|
| VisualCloze | `visualcloze/source/baseline/p2/visualcloze_p2_task_text.py`, `visualcloze/run_visualcloze_tests.py` |
| Painter | `painter/painter_native.py` + `shared/depth_output_adapter.py` |
| VICT | `vict/vict_native.py`, `vict/adaptation.py` |
| GenLV 101 Huge | `genlv/genlv_runtime.py`, `genlv/genlv_test.py` |
| LaVin-DiT | `lavin_dit/lavin_dit_bench_v2.py` |
| PromptDiffusion | `promptdiffusion/source_optimized/engine.py`, `promptdiffusion/run_tests_retry1.py` |
| RelationAdapter | `relation_adapter/plan_val.json`, `relation_adapter/source/runtime_loader.py` |
| InstaManip | `instamanip/instamanip_bench_v2.py` |
| IMProv | `improv/improv_cpu.py` + `shared/edge_contrast_adapter.py` |
| Analogist | `analogist/analogist_q4_v5p16.py` |
| LVM 7B | `lvm/lvm_bench_v2.py`, `lvm/lvm_runtime_config.py` |

Shared episode builder / scorer: `shared/bench_episode_v1.py`, `shared/selected_metric_backend.py`, `shared/signed_depth_metric.py`.

See also: `reports/baseline_implementations_2026-11-19/baseline_model_implementation_reference.md`.
