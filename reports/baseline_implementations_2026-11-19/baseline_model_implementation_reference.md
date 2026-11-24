# Final implemented adjustments for the eleven baselines

The final selected Bench v2 implementations use the settings and changes below. `P` is the number of demonstration input–target pairs. Settings refer to the implemented selected recipes, including their adopted numerical and output-processing changes.

# VisualCloze

- **Final settings:** P2, BF16, 30 FlowMatch Euler steps, guidance 30, and CPU seed 0 reset per query. Use the frozen task prompts with an empty content prompt. Run padded batches of two across eight TPU devices with a `(2,4)` data/model mesh and cached text embeddings.
- **Implemented numerical fix:** During VAE decoding, compute `decoder.up_blocks.3.resnets.2.norm2` in FP32, including its input, weight, and bias; cast the result back to the input dtype. Preserve the existing group count and epsilon. Check the raw decoded tensor for finite values before converting it to pixels.
- **Output:** Save the native 384-pixel prediction and resize it to 256 with LANCZOS for scoring.

Implementation: [P2 runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/visualcloze_recovery/source/baseline/p2/visualcloze_p2_task_text.py), [final decoder fix](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/run_visualcloze_tests.py).

# Painter

- **Final settings:** P1, TPU FP32, padded batch size four. Stack the demonstration input and query vertically into an 896×448 canvas. Resize inputs with bicubic interpolation, demonstration targets with nearest-neighbor interpolation, and apply ImageNet normalization. Mask the lower 784 of 1,568 patches.
- **Output processing:** Extract the lower prediction half, denormalize, clip to `[0,255]`, resize to 256 with nearest-neighbor interpolation, and truncate to `uint8`.
- **Implemented depth adjustment:** During depth scoring, interpret the saved RGB image as `depth = mean(R,G,B)/255`, then take its reciprocal at positive pixels. Preserve exactly black pixels as invalid. This conversion uses only the prediction; copy-input controls retain the generic decoder.

Implementation: [Painter runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/painter_runner/painter_native.py), [depth decoder](/kmh-nfs-ssd-us-mount/code/kristine/lvm/visualcloze-results-site/scripts/depth_output_adapter.py).

# VICT

- **Final adaptation:** P1 with 60 encoder updates per query and no crop/flip augmentation. Train parameters whose names do not begin with `decoder`. Reset model weights and tensor optimizer state for each query. Use the cycle-reconstruction smooth-L1 loss, retaining the gradient through the predicted query target.
- **Optimizer and execution:** XLA syncfree AdamW, learning rate `1e-6`, betas `(0.9,0.999)`, epsilon `1e-8`, and zero weight decay. Use BF16 autocast, activation checkpointing, and a fresh GradScaler initialized to 65,536 per query. Seed Torch and NumPy once at process startup with 0; preserve the continuing random stream across queries.
- **Final output processing:** Clip before resizing and truncate to `uint8`. Resize to 256 using nearest-neighbor for instance segmentation, pose, and facial landmarks; bilinear for depth, semantic segmentation, edges, saliency, and normals; bicubic for appearance tasks. Apply Painter's saved-RGB reciprocal-depth decoder for depth scoring.

Implementation: [VICT runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/vict_runner/vict_native.py), [adaptation step](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/vict_runner/adaptation.py).

# GenLV 101 Huge

- **Final selection:** Use the extended 101-task Huge model with `huge-checkpoint-49.pth`, P1, CPU FP32, batch size one, seed 0, eight intra-op threads, and one inter-op thread. Divide test queries across eight CPU shards.
- **Input and execution:** Convert RGB to BGR, divide by 255, and arrange as NCHW. Call the model with `[demo_input, demo_target, query_input, query_input]`; the fourth image is a placeholder. Use the direct uncached forward.
- **Output:** Multiply by 255, clip, truncate to `uint8`, transpose to HWC, and convert BGR back to RGB. Keep the native 256-pixel resolution and checkpoint's output coordinate conventions.

Implementation: [model runtime](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/runner/source/optimized/genlv_runtime.py), [selected execution](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/runner/genlv_test.py).

# LaVin-DiT

- **Final settings:** P8, BF16 DiT and STVAE, guidance 1.5, and the fixed 20-point time grid giving 19 Euler integration intervals. Use the reference conditional/null guidance implementation.
- **Conditioning and noise:** Resize each frame to 512 with antialiased bicubic interpolation and normalize to `[-1,1]`. Order frames as `x1,y1,…,x8,y8,query`. Sample the VAE posterior and transform its latent as `(sample − 0.032470703125) × 1.08936170213`. Use two separate CPU generators, each initialized to seed 0: one for posterior noise and one for initial flow noise.
- **Implemented execution adjustment:** Serialize host-side checkpoint loading across ranks to limit peak memory. Strictly validate inference weights while permitting the enumerated training-only VAE keys.
- **Output:** Save the native 512-pixel image and resize to 256 with LANCZOS.

Implementation: [selected LaVin-DiT runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/lavin_p8_retry18/lavin_dit_bench_v2.py).

# PromptDiffusion

- **Final input adjustment:** Select A1 with P1 and normalize the demonstration input, demonstration target, and query to `[-1,1]`. Concatenate the demonstration input and target in that order into six channels; supply the query as a separate three-channel condition. Use the frozen task prompts and empty negative text.
- **Final sampler:** BF16, DDIM 50, eta 0, guidance 7.5, conditioning scale 1.0, and batch size four. Derive each query's seed from its frozen namespace, manifest, pack, and record identity; draw noise in CPU FP32 before device casting. Keep `cache_conditioning=False` and `dynamic_coefficients=False`.
- **Execution and output:** Use eight devices with a `(4,2)` data/replica mesh. Pad short batches by repeating the last query and discard padding outputs. Check decoded floats for finiteness, map them to `[0,1]`, clip, multiply by 255, round, and cast to `uint8`.

Implementation: [sampling engine](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/promptdiffusion_runner/source_optimized/engine.py), [selected execution](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/promptdiffusion_runner/run_tests_retry1.py).

# RelationAdapter

- **Final settings:** P1, BF16, 24 FlowMatch Euler steps, guidance 3.5, relation scale 1.0, seed 1000, and 256×256 output. Use the neutral prompt “Apply the demonstrated image transformation.” and cached text embeddings.
- **Adapter implementation:** Load the released relation adapter and LoRA; fuse the LoRA once and unload its loading wrapper. Encode the demonstration input and target with SigLIP, project their pooled features, and concatenate the relation representations in source-then-target order. Supply the query through the FLUX image-conditioning VAE path.
- **Implemented TPU adjustment:** Apply SPMD/FSDP wrapping to all 19 dual-stream and 38 single-stream FLUX blocks plus the root. The selected backend uses eight global devices across two workers. Round final pixels when converting to RGB `uint8`.

Implementation: [selected settings](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/relation/plan_val.json), [runtime loader](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/relation/source/runtime_loader.py).

# InstaManip

- **Final selection:** P3 with the released 3shot checkpoint, BF16, 50 Euler steps, text guidance 7.5, image guidance 1.5, and 1024×1024 output. Greedy agent generation uses 64 tokens per image, 30 edit tokens, and a 750-token cap.
- **Input and noise:** Demonstrations pass through a 1024-pixel resize before the 448-pixel visual transform; query visual encoding goes directly to 448, with a separate 1024-pixel query for SDXL. Derive the seed from `42|task|q|P` and create explicit CPU FP32 noise of shape `[1,4,128,128]` before casting to BF16.
- **Implemented TPU adjustments:** Use fixed-capacity KV storage, a fused causally masked block for processor-forced image tokens, device timestep/sigma scalars for Euler sampling, and a VAE boundary cast from FP32 images to BF16. Keep the agent batch size one; replicate SDXL weights and shard its diffusion batch across 32 devices on eight workers.
- **Output:** Keep batches task-homogeneous, repeat the last query when padding, and discard padding outputs. Save native 1024-pixel images and bicubic-resize to 256 for scoring.

Implementation: [selected InstaManip runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/instamanip_p3/instamanip_bench_v2.py).

# IMProv

- **Final selection:** P1, CPU FP32, batch size four, empty text, one masked-prediction step, guidance 1, and temperature 0 implemented as argmax.
- **Canvas and decoding:** Place 111-pixel demonstration and query tiles in a 224-pixel canvas with a two-pixel internal gap. Predict the missing bottom-right region. Decode the complete 14×14 code grid, composite it with the visible canvas, clip floats, truncate to `uint8`, crop the bottom-right tile, and bicubic-resize to 256.
- **Implemented edge adjustment:** For edge scoring only, convert RGB to `[0,1]`, compute `luma = 0.299R + 0.587G + 0.114B`, and use `sqrt(clamp(luma,0,1))` as confidence. Retain the 99-threshold edge scorer. Other tasks and copy-input controls retain their existing processing.

Implementation: [CPU runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/improv_runner/improv_cpu.py), [edge adjustment](/kmh-nfs-ssd-us-mount/code/kristine/lvm/visualcloze-results-site/scripts/edge_contrast_adapter.py).

# Analogist

- **Final conditioning:** P1 with a frozen per-query vision-caption cache built from permitted input images. Use a 512-pixel inpainting grid with 253-pixel tiles and copy the query quadrant into the answer quadrant. Set self-attention control scale to 1.3 and apply self-attention control/cross-attention masking to layers 3–10.
- **Final sampler:** BF16, eight candidates in a batch over eight devices. Use 50 nominal DDIM steps: strength 0.7 gives 35 executed steps for denoising/deblurring; strength 0.98 gives 49 for the other tasks. Guidance is 7.5 for inpainting and 15 otherwise.
- **Implemented random draws:** Use separate candidate-specific seeds for diffusion, initial-image VAE noise, and masked-image VAE noise. Draw each on CPU in FP32 before device casting.
- **Output and aggregation:** Decode the full canvas, extract its bottom-right 256×256 quadrant, and round to RGB `uint8`. Retain all eight candidate images and average their individual scores for the query.

Implementation: [selected Analogist wrapper](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/analogist_selected/analogist_q4_v5p16.py).

# LVM 7B

- **Final settings:** P1, TPU FP16 for the causal model, and CPU VQ encoding/decoding. Use temperature 1.0, top-p 0.9, top-k 50, seed 0, and one candidate. Concatenate 256 tokens each for the demonstration input, demonstration target, and query; generate exactly 256 answer tokens.
- **Implemented token handling:** Insert no BOS token. Suppress logits at IDs 8192 and above, and set EOS/padding to suppressed ID 8192. This preserves code 2 as a valid image token and prevents premature image termination. Validate output length and the `[0,8191]` code range before decoding.
- **Implemented TPU adjustments:** Load converted parameters on CPU, apply EasyLM partition rules with mesh `1,device_count,1`, and wrap generation with `pjit`. Use 1,024-token query/key scan-attention chunks for P1 and disable scan-MLP during cached single-token decoding.

Implementation: [LVM runner](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/lvm/lvm_bench_v2.py), [runtime configuration](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/recovered/lvm/lvm_runtime_config.py).

# Shared final evaluation adjustments

- Use Bench v2 supervision v6 and the real demonstration condition. Construct episodes with `default_rng([42, task_seed(task), q, P])`, drawing demonstrations from the query's source while excluding the query. Apply episode palette coding consistently and keep query targets outside the inference inputs.
- Use the frozen corrected-v3 scorer with signed-scale relative-depth alignment. Fit scale and shift against ground truth at scoring time, allowing a negative scale, then compute depth δ1. Painter/VICT's depth conversion and IMProv's edge confidence adjustment are applied as described above.
- Compute edge OIS per image over 99 thresholds, then average across queries. Preserve copy-input controls and each method's declared candidate aggregation.

Implementation: [episode builder](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/runner/tools/bench_episode_v1.py), [selected scoring backend](/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-11-10/bench_v2_all11_test/selected_metric_backend.py).
