# FLUX2 Prompt Initialization 10k Full-Shuffle

## Run Summary

| Field | Value |
|---|---|
| Run ID | `eo8i590o` |
| Status | `interrupted` |
| Started | `2026-07-20T02:14:58Z` |
| Finished | `2026-07-20T03:31:54Z` |
| Baseline | `not specified` |
| Git revision | `849027c2ecaf8e083fd67b3bf8dc5613f325a731` with untracked experiment files |
| Command | `env PYTHONUNBUFFERED=1 PYTHONPATH=/tmp/flux2_probe/jflux:/tmp/flux2_probe JAX_PLATFORMS=tpu JAX_COORDINATOR_ADDRESS=10.202.0.194:12641 python -u /tmp/flux2_probe/experiments/flux2_cv_prompt_probe/trainer_optimize_text_token_only_prompt_init.py --config /tmp/flux2_probe/experiments/flux2_cv_prompt_probe/optimize_concise_prompt_init_10k_wandb_conditioned.yaml` |
| Seed | `42` |
| Hardware | `v6e-32` Spot TPU, 8 workers / 32 chips, `us-east5-b`; node was subsequently preempted and deleted |

## Configuration

Crash-run source: `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/optimize_concise_prompt_init_10k_wandb_conditioned.yaml`

Recovery source: `experiments/flux2_cv_prompt_probe/optimize_concise_prompt_init_10k_wandb_conditioned.yaml`

```yaml
trainer: optimize_text_token_only_prompt_init
runtime:
  run_name: flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle
  workdir: /kristine/lvm/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle
  local_output_dir: /home/kristinelu/elf-runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle_resume1000
  gcp_zone: asia-northeast1-b
  seed: 42
  auto_resume_final: false
  save_optimizer_state: false
logging:
  log_every: 10
  save_every: 1000
  validation_every: 1000
  validation_num_steps: 50
  validation_artifacts_per_task: 10
  embedding_snapshot_every: 1000
wandb:
  wandb: true
  wandb_project: VGI
  wandb_entity: kaynelu921-massachusetts-institute-of-technology
  wandb_run_name: flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle
  wandb_mode: online
dataset:
  image_size: 256
  val_visualize_filename_substr: ""
  dataloader:
    train:
      module: experiments.flux2_cv_prompt_probe.latent_npz_loader
      function: load_data
      kwargs:
        split: train
        repeat: true
        shuffle: true
        cache: true
        datasets:
          - {name: monocular_depth, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: surface_normals, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: panoptic_segmentation, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: binary_segmentation, data_dir: /home/kristinelu/elf-data/training_latents_10k}
    val:
      module: experiments.flux2_cv_prompt_probe.latent_npz_loader
      function: load_data
      kwargs:
        split: val
        repeat: false
        shuffle: false
        cache: true
        datasets:
          - {name: monocular_depth, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: surface_normals, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: panoptic_segmentation, data_dir: /home/kristinelu/elf-data/training_latents_10k}
          - {name: binary_segmentation, data_dir: /home/kristinelu/elf-data/training_latents_10k}
model:
  model_variant: 4B
  flux_weights: /home/kristinelu/elf-runs/source/flux2_v4_resume1000/experiments/flux2_cv_prompt_probe/assets/flux2_4b/jflux2_transformer.safetensors
  flux_config: /home/kristinelu/elf-runs/source/flux2_v4_resume1000/experiments/flux2_cv_prompt_probe/assets/flux2_4b/transformer_config.json
  vae_weights: /home/kristinelu/elf-runs/source/flux2_v4_resume1000/experiments/flux2_cv_prompt_probe/assets/flux2_4b/jflux2_vae.safetensors
  vae_config: /home/kristinelu/elf-runs/source/flux2_v4_resume1000/experiments/flux2_cv_prompt_probe/assets/flux2_4b/vae_config.json
  task_init_path: /home/kristinelu/elf-runs/source/flux2_v4_resume1000/experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/embedding_snapshots/task_embeddings_microstep_000001000.npz
  resume_workdir: null
  lora_impl: disabled
  lora_rank: 0
  lora_alpha: 1.0
  task_num_tokens: 512
  gradient_checkpointing: true
  guidance: 4.0
  condition_time_index: 10
optimization:
  batch_size: 32
  epochs: 100000
  max_steps: 10000
  learning_rate: 2.0e-04
  min_learning_rate: 1.0e-06
  use_lr_scheduler: false
  lr_scheduler: constant
  lr_warmup_steps: 0
  gradient_accumulation_steps: 1
  weight_decay: 0.0
  max_grad_norm: 1.0
  loss_type: huber
  huber_c: 0.1
  flow_loss_weighting: cosmap
  timestep_density: logit_normal
  flow_logit_mean: 0.0
  flow_logit_std: 1.0
  flow_mode_scale: 1.29
  flow_beta_alpha: 0.5
  flow_beta_beta: 0.5
distributed:
  use_hsdp: true
  mesh_device_type: tpu
  mesh_data_axis_size: -1
  mesh_fsdp_axis_size: 1
```

## Changed Hyperparameters

No explicit baseline was specified. Recovery preserves all optimization hyperparameters and changes only storage locality, TPU zone, non-fatal W&B handling, and initialization from the step-1000 embedding snapshot.

## Results

| Dataset / split | Metric | Value | Step / epoch | Direction | Source |
|---|---|---:|---:|---|---|
| Fixed held-out, all tasks | decoded validation L1 | `0.640926` | `1000` | min | `train.log` |
| Training mixture | final logged flow-matching loss | `0.063240` | `1690` | min | `train.log` |

## Artifacts

| Artifact | Path or URL | Verified |
|---|---|---|
| Crash-run configuration | `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/optimize_concise_prompt_init_10k_wandb_conditioned.yaml` | yes |
| Recovery configuration | `experiments/flux2_cv_prompt_probe/optimize_concise_prompt_init_10k_wandb_conditioned.yaml` | yes |
| Step-1000 checkpoint | `gs://kmh-gcp-us-east5/kristine/lvm/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/checkpoints/checkpoint_1000` | yes |
| Step-1000 embedding snapshot | `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/embedding_snapshots/task_embeddings_microstep_000001000.npz` | yes |
| Fixed validation images | `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/validation/step_0001000` | yes |
| Tracking run | `https://wandb.ai/kaynelu921-massachusetts-institute-of-technology/VGI/runs/eo8i590o` | yes |
| Detailed output | `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/train.log` | yes |

## Failure Details

- Last known step: `1690`
- Error summary: W&B local service socket reset raised `ConnectionResetError: Connection lost`; the original Spot TPU was later preempted and deleted.
- Detailed output: `experiments/flux2_cv_prompt_probe/optimization_runs/flux2_prompt_init_10k_10000_wandb_conditioned_v4_fullshuffle/train.log`

## Observations

- Full-dataset shuffling removed the prior approximately 313-step task-order sawtooth.
- Step-1000 embeddings, checkpoint, validation images, and W&B identity are preserved.
- The first Asia replacement passed READY/HEALTHY/Spot/device-idle preflight but was preempted during staging before training began.

## Interpretation

- The step-1690 stop was a telemetry robustness bug, not an optimization divergence. W&B calls are now non-fatal.
- Because optimizer state was disabled, resuming from the step-1000 embedding snapshot with a fresh optimizer is equivalent to the available checkpoint contents.

## Updates

- `2026-07-20T03:31:54Z`: Training stopped at step 1690 after a W&B `ConnectionResetError`; step-1000 checkpoint remained valid.
- `2026-07-20T03:55:00Z`: Added non-fatal W&B handling, prepared node-local Asia recovery from microstep 1000, and activated at-least-30-minute persistent supervision. Asia replacement was preempted during staging; recovery monitor remains active.
- `2026-07-20T04:10:00Z`: Reconfirmed the ELF launch-skill monitoring contract. The apparent READY Asia TPU was stale Cloud Asset data and direct preflight returned `NOT_FOUND`; direct TPU API requests remain Spot `CREATING`. No training process is currently running, so the monitor is waiting for a READY/HEALTHY/device-idle Kristine v6e-32 and will then stage node-local data, resume from microstep 1000, and verify the first completed step before reporting the run as live.
- `2026-07-20T04:12:00Z`: Verified the only READY Kristine v6e-32 is occupied on all eight workers by the separate `100k_wandb_lr22_resume2000_fullshuffle` run and left it untouched. Completed all three recovery archives in the Asia, us-east5, and us-central1 same-region staging buckets. Direct API still shows replacement Spot v6e-32 requests in `CREATING`; training has not relaunched yet.
- `2026-07-20T04:15:00Z`: Third consecutive recovery interval remained capacity-blocked. Direct API showed only Spot v6e-32 requests in `CREATING`; the sole READY v6e-32 remained occupied by the preserved separate run. Investigated additional zones: `us-west4-a` does not offer v6e, while `us-east1-d` offers v6e-32 but lacks an ELF-approved regional service-account mapping. No unsafe launch was attempted.
