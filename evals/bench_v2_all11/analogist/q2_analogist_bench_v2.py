#!/usr/bin/env python3
"""Direct-native P=1 Analogist producer for the supervision-v6 Bench v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import types
import resource

import numpy as np

from analogist_loader import (
    MODEL_REPOSITORY, MODEL_REVISION, NEGATIVE_PROMPT, PROMPT, SOURCE_COMMIT,
    load_runtime_components,
)
import bench_v2_native_p1_common as C


IMAGE_SIZE = 512
LATENT_SIZE = 64
BATCH = 4
SMOKE_EPISODES = 16
AUTHORIZED_TPU = "kmh-tpuvm-v5p-8-us-east5-spot-lyy-10151630uqf"
EXPECTED_ACCELERATOR = "v5p-8"
EXPECTED_TOPOLOGY = "2x2x1"
EXPECTED_GLOBAL_DEVICES = 4
EXPECTED_ADDRESSABLE_DEVICES = 4
EXPECTED_WORKERS = 1
INFERENCE_STEPS = 50
STRENGTH = 0.98
GUIDANCE_SCALE = 15.0
SCALE_SAC = 1.3
NOISE_NAMESPACE = "analogist_bench_v2_native_p1_v1"
SOURCE_HASHES = {
    "models/attn_utils.py": "04be20cd9f124b55c0656f1785140e627992c25bb6f9b9f6750912cdcf601954",
    "models/scheduling_ddim.py": "b785d63183c2e93265d02534e066bda06fb52b224b186b7fb70b6195b426efdf",
    "analogist.py": "f50b1ac8b7e2d8798c14e2494ab1bc4059041fbe106ded744c7704a4d12d3aca",
    "visual_prompting.py": "7274241b2571734e7084a0871bfc5b48ad2ec7603501e9a936d24c950b241bb5",
    "LICENSE": "e8cf4a8793b0c951cac356a62b659405632e21e88a13c654b9d4cdf80304e772",
    "assets/index_sa.pkl": "836b5507f90455f160d8b4c115db34c2e7a4131124be6fbf0f7aa8464133bc7e",
    "assets/mask_image_inpainting.png": "67621b44198e0c154a0d105cd6dd4604bcb764b49a773cef626580e63e18ab8f",
}
LEGACY_HELPER_HASHES = {
    "analogist_loader.py": "cd6a83e1758fbdb60c8b977282234d21bfe3e75dcb426f69ccf939d4cd8e5603",
    "analogist_compat.py": "0ed53237652213a1ab16ff05589358a0174e7fd1b4db787beaeb64456547e2b1",
    "cpu_preflight.py": "dc35c86d16c3b3d340db78adc1d3f1f08bc502b64f33e37eaefc1298d33b75fb",
}
BUNDLED_LAZY_LOADER_SHA256 = "6bcdbb8aadfb95f9a42e26d7448bca5e07e9167b8a94f27c82e56539de55a0f6"
ADAPTER = {
    "model": "Analogist", "source_repository": "https://github.com/edward3862/Analogist",
    "source_commit": SOURCE_COMMIT, "base_model": MODEL_REPOSITORY,
    "base_model_revision": MODEL_REVISION, "native_prompt_count": 1,
    "episode_draw_p0": 1, "demo_selection": "direct same-source CL9 P=1 episode",
    "condition": "real only", "grid": "official A/A-prime/B/B 512px inpainting canvas",
    "text_prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
    "text_policy": "fixed task-neutral adaptation; no GPT-4V and no task/class names",
    "paper_scope": "native structural Analogist inference, not the paper's GPT-4V prompt-generation arm",
    "scheduler": "official Analogist DDIM", "steps": INFERENCE_STEPS, "strength": STRENGTH,
    "guidance_scale": GUIDANCE_SCALE, "scale_sac": SCALE_SAC,
    "sac_layers": [3, 11], "cam_layers": [3, 11], "outputs_per_query": 1,
    "ensemble": False, "runtime_precision": "TPU bfloat16; explicit CPU float32 noise",
    "rng_namespace": NOISE_NAMESPACE,
    "hardware_variant": "v5p-8 one-worker four-device SPMD data mesh",
}
CONFIG = {
    "plan_schema": "analogist_bench_v2_native_p1_plan_v1",
    "run_manifest_schema": "analogist_bench_v2_native_p1_run_manifest_v1",
    "producer_schema": "analogist_bench_v2_native_p1_corrected_v3_deferred_v1",
    "episode_schema": "bench_v2_episode_v26", "arm": "Analogist", "readout_tag": "native_p1",
    "readout": "Analogist@native_p1", "adapter": ADAPTER,
    "inference_marker_schema": "analogist_bench_v2_native_p1_inference_complete_v1",
    "validation_receipt_schema": "analogist_bench_v2_corrected_v3_validation_receipt_v1",
}


def verify_runtime(source_dir: Path, model_dir: Path, embeddings: Path,
                   helper_dir: Path) -> dict:
    source = {name: C.sha256_file(source_dir / name) for name in SOURCE_HASHES}
    legacy_helpers = {
        name: C.sha256_file(helper_dir / name) for name in LEGACY_HELPER_HASHES
    }
    bundled_loader = Path(__file__).resolve().parent / "analogist_loader.py"
    bundled_loader_sha256 = C.sha256_file(bundled_loader)
    if source != SOURCE_HASHES or legacy_helpers != LEGACY_HELPER_HASHES:
        raise ValueError("pinned official Analogist source/helper identity differs")
    if bundled_loader_sha256 != BUNDLED_LAZY_LOADER_SHA256:
        raise ValueError("bundled lazy Analogist loader identity differs")
    if bundled_loader.resolve() == (helper_dir / "analogist_loader.py").resolve():
        raise ValueError("bundled lazy loader and legacy helper must be distinct snapshots")
    manifest = helper_dir / "assets/RUNTIME_SHA256SUMS"
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip()
        entries[relative] = digest
        path = helper_dir / relative
        if C.sha256_file(path) != digest:
            raise ValueError(f"Analogist asset differs: {relative}")
    expected_model = (helper_dir / "assets/model").resolve()
    if model_dir.resolve() != expected_model or embeddings.resolve() != (helper_dir / "assets/fixed_prompt_embeddings.safetensors").resolve():
        raise ValueError("Analogist asset paths are outside the authenticated snapshot")
    return {"source_commit": SOURCE_COMMIT, "source_sha256": source,
            "legacy_helper_sha256": legacy_helpers,
            "bundled_lazy_loader_sha256": bundled_loader_sha256,
            "runtime_manifest_sha256": C.sha256_file(manifest),
            "asset_sha256": entries, "legacy_run_bundle_sha256":
                "deaafbb04d5e52192e779c5aae59a749184fbd70666a9108ca068dc353bf64b5",
            "legacy_model_assets_sha256":
                "3417743348d4b32bc8c217be606af9acc03e0fabbfb98fb640b03536373ef287"}


def seed_for(episode: dict, domain: str) -> int:
    material = "|".join((NOISE_NAMESPACE, C.MANIFEST_SHA256, episode["pack_id"],
                         episode["query_record"], domain))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little") & 0x7fff_ffff_ffff_ffff


def cpu_noise(shape: tuple[int, ...], seed: int):
    import torch
    return torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed),
                       dtype=torch.float32)


def make_grid(a: np.ndarray, a_prime: np.ndarray, b: np.ndarray) -> np.ndarray:
    from PIL import Image

    process_size = 253
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    canvas.paste(Image.fromarray(a, "RGB").resize((process_size, process_size)), (2, 2))
    canvas.paste(Image.fromarray(a_prime, "RGB").resize((process_size, process_size)), (257, 2))
    canvas.paste(Image.fromarray(b, "RGB").resize((process_size, process_size)), (2, 257))
    canvas.paste(Image.new("RGB", (process_size, process_size), (0, 0, 0)), (257, 257))
    pasted = canvas.copy()
    pasted.paste(canvas.crop((0, 256, 256, 512)), (256, 256))
    return np.asarray(pasted, np.uint8)


def load_batch(rows: list[tuple[dict, dict]]):
    import torch
    padded = list(rows)
    while len(padded) < BATCH:
        padded.append(rows[-1])
    grids, diffusion, init_eps, masked_eps, metadata = [], [], [], [], []
    for episode, rebuilt in padded:
        grids.append(make_grid(rebuilt["demo_input"], rebuilt["demo_target"], rebuilt["query_input"]))
        noise_row = {}
        for domain, destination in (("diffusion", diffusion), ("init_vae", init_eps),
                                    ("masked_vae", masked_eps)):
            seed = seed_for(episode, domain)
            value = cpu_noise((1, 4, LATENT_SIZE, LATENT_SIZE), seed)
            destination.append(value)
            noise_row[f"{domain}_seed"] = seed
            noise_row[f"{domain}_sha256"] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        metadata.append(noise_row)
    images = np.stack(grids).astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    return (images, torch.cat(diffusion).numpy(), torch.cat(init_eps).numpy(),
            torch.cat(masked_eps).numpy(), metadata)


def timestep_rows(scheduler):
    scheduler.set_timesteps(INFERENCE_STEPS, device="cpu")
    start = max(INFERENCE_STEPS - min(int(INFERENCE_STEPS * STRENGTH), INFERENCE_STEPS), 0)
    timesteps = [int(value) for value in scheduler.timesteps[start:].tolist()]
    if len(timesteps) != 49:
        raise ValueError("Analogist strength/timestep contract changed")
    alphas = scheduler.alphas_cumprod.float().cpu()
    final_alpha = float(scheduler.final_alpha_cumprod)
    stride = scheduler.config.num_train_timesteps // INFERENCE_STEPS
    return [(step, float(alphas[step]), float(alphas[step - stride]) if step >= stride else final_alpha)
            for step in timesteps]


def step_latents(latents, noise, alpha: float, alpha_previous: float, scheduler):
    """Value-equivalent eta=0 DDIM step for the pinned official scheduler."""
    if str(scheduler.config.prediction_type) != "epsilon":
        raise ValueError("official Analogist scheduler prediction type changed")
    if bool(scheduler.config.thresholding):
        raise ValueError("official Analogist scheduler unexpectedly enables thresholding")
    original = (latents - math.sqrt(1.0 - alpha) * noise) / math.sqrt(alpha)
    if bool(scheduler.config.clip_sample):
        clip = float(scheduler.config.clip_sample_range)
        original = original.clamp(-clip, clip)
    return math.sqrt(alpha_previous) * original + math.sqrt(1.0 - alpha_previous) * noise


def register_attention(source_dir: Path, unet):
    from analogist_compat import install as install_attention_compat

    if str(source_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(source_dir.resolve()))
    from models.attn_utils import AnalogistAttentionEdit, regiter_attention_editor_diffusers
    install_attention_compat(AnalogistAttentionEdit)
    previous = Path.cwd()
    try:
        os.chdir(source_dir)
        editor = AnalogistAttentionEdit(sac_start_layer=3, sac_end_layer=11,
                                        cam_start_layer=3, cam_end_layer=11,
                                        scale_sac=SCALE_SAC)
    finally:
        os.chdir(previous)
    regiter_attention_editor_diffusers(types.SimpleNamespace(unet=unet), editor)
    return editor


def predict_batch(components, arrays, embeddings, mask_np, device, mesh, editor):
    import torch
    import torch.nn.functional as functional
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    unet, vae, scheduler = components
    image_np, diffusion_np, init_eps_np, masked_eps_np, noise_rows = arrays
    dtype = torch.bfloat16
    image = torch.from_numpy(image_np).to(device=device, dtype=dtype)
    mask = torch.from_numpy(mask_np).repeat(BATCH, 1, 1, 1).to(device=device, dtype=dtype)
    masked = image * (mask < 0.5)
    diffusion = torch.from_numpy(diffusion_np).to(device=device, dtype=dtype)
    init_eps_tensor = torch.from_numpy(init_eps_np).to(device=device, dtype=dtype)
    masked_eps_tensor = torch.from_numpy(masked_eps_np).to(device=device, dtype=dtype)
    for tensor in (image, mask, masked, diffusion, init_eps_tensor, masked_eps_tensor):
        xs.mark_sharding(tensor, mesh, ("data",) + (None,) * (tensor.ndim - 1))
    positive, negative = embeddings
    positive = positive.repeat(BATCH, 1, 1).to(device=device, dtype=dtype)
    negative = negative.repeat(BATCH, 1, 1).to(device=device, dtype=dtype)
    prompt = torch.cat((negative, positive), dim=0)
    xs.mark_sharding(prompt, mesh, ("data", None, None))
    editor.reset()
    with torch.no_grad():
        init_dist = vae.encode(image, return_dict=False)[0]
        image_latents = (init_dist.mean + init_dist.std * init_eps_tensor) * float(vae.config.scaling_factor)
        masked_dist = vae.encode(masked, return_dict=False)[0]
        masked_latents = (masked_dist.mean + masked_dist.std * masked_eps_tensor) * float(vae.config.scaling_factor)
        mask_latents = functional.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE))
        mask_guided = torch.cat((mask_latents, mask_latents), dim=0)
        masked_guided = torch.cat((masked_latents, masked_latents), dim=0)
        for tensor in (image_latents, masked_latents, mask_latents, mask_guided, masked_guided):
            xs.mark_sharding(tensor, mesh, ("data",) + (None,) * (tensor.ndim - 1))
        schedule = timestep_rows(scheduler)
        _, first_alpha, _ = schedule[0]
        latents = math.sqrt(first_alpha) * image_latents + math.sqrt(1.0 - first_alpha) * diffusion
        for timestep, alpha, alpha_previous in schedule:
            guided = torch.cat((latents, latents), dim=0)
            latent_input = torch.cat((guided, mask_guided, masked_guided), dim=1)
            xs.mark_sharding(latent_input, mesh, ("data", None, None, None))
            noise_guided = unet(latent_input, torch.tensor(timestep, device=device, dtype=torch.int64),
                                encoder_hidden_states=prompt, return_dict=False)[0]
            unconditioned, conditioned = noise_guided.chunk(2)
            noise = unconditioned + GUIDANCE_SCALE * (conditioned - unconditioned)
            latents = step_latents(latents, noise, alpha, alpha_previous, scheduler)
            xs.mark_sharding(latents, mesh, ("data", None, None, None)); xm.mark_step()
        decoded = vae.decode(latents / float(vae.config.scaling_factor), return_dict=False)[0]
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1); xm.mark_step()
        decoded = decoded.float().cpu()[:, :, 256:, 256:]
        images = (decoded * 255.0).round().clamp(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy(), noise_rows


def prepare(args) -> int:
    source_identity = verify_runtime(args.source_dir, args.model_dir, args.prompt_embeddings,
                                     args.helper_dir)
    plan = C.prepare_plan(phase=args.phase, out=args.out, pack_cache=args.pack_cache,
                          config=CONFIG, source_identity=source_identity,
                          val_receipt=args.val_receipt)
    print(json.dumps({"plan": str(args.out), "episodes": len(plan["episodes"]),
                      "identity": plan["plan_identity"]}, sort_keys=True))
    return 0


def select_smoke_episodes(plan: dict) -> list[dict]:
    by_task = {}
    for episode in plan["episodes"]:
        by_task.setdefault(episode["task"], []).append(episode)
    selected = [rows[0] for rows in by_task.values()]
    for rows in by_task.values():
        for episode in rows[1:]:
            if len(selected) == SMOKE_EPISODES:
                break
            selected.append(episode)
        if len(selected) == SMOKE_EPISODES:
            break
    if len(by_task) != 13 or len(selected) != SMOKE_EPISODES:
        raise ValueError("task-diverse smoke coverage changed")
    return selected


def run(args) -> int:
    if args.smoke and args.smoke_marker is None:
        raise ValueError("task-diverse smoke requires --smoke-marker")
    if not args.smoke and args.smoke_marker is not None:
        raise ValueError("--smoke-marker is valid only with --smoke")
    source_identity = verify_runtime(args.source_dir, args.model_dir, args.prompt_embeddings,
                                     args.helper_dir)
    plan = C.load_plan(args.plan, phase=args.phase, config=CONFIG)
    if plan.get("source_identity") != source_identity:
        raise ValueError("runtime source/assets differ from frozen plan")
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from PIL import Image
    from safetensors import safe_open
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False); xr.use_spmd()
    runtime_devices, addressable = int(xr.global_runtime_device_count()), int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if (backend.upper() != "TPU" or runtime_devices != EXPECTED_GLOBAL_DEVICES
            or addressable != EXPECTED_ADDRESSABLE_DEVICES
            or args.expected_devices != EXPECTED_GLOBAL_DEVICES):
        raise ValueError(f"refusing fallback/topology: {backend=} {runtime_devices=} {addressable=}")
    if os.environ.get("EXPECTED_TPU_NAME") != AUTHORIZED_TPU:
        raise ValueError("exact authorized TPU identity differs")
    mesh = xs.Mesh(np.arange(EXPECTED_GLOBAL_DEVICES), (EXPECTED_GLOBAL_DEVICES,), ("data",)); xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {"backend": backend, "device": str(device), "runtime_device_count": runtime_devices,
                "addressable_runtime_device_count": addressable, "torch": torch.__version__,
                "torch_xla": getattr(torch_xla, "__version__", "unknown"), "host": socket.gethostname(),
                "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
                "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE")}
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    with safe_open(str(args.prompt_embeddings), framework="pt", device="cpu") as handle:
        if handle.metadata() != {"prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT}:
            raise ValueError("fixed prompt metadata differs")
        embeddings = (handle.get_tensor("positive"), handle.get_tensor("negative"))
    if any(tuple(value.shape) != (1, 77, 768) for value in embeddings):
        raise ValueError("fixed prompt embedding shape differs")
    if args.mask.resolve() != (args.source_dir / "assets/mask_image_inpainting.png").resolve():
        raise ValueError("mask path is outside the authenticated official source")
    mask_image = Image.open(args.mask).convert("L")
    if mask_image.size != (512, 512):
        raise ValueError("official Analogist mask dimensions differ")
    mask_np = (np.asarray(mask_image, np.float32)[None, None] / 255.0 >= 0.5).astype(np.float32)
    components = load_runtime_components(args.source_dir, args.model_dir, dtype=torch.bfloat16)
    components = (components[0].eval().to(device), components[1].eval().to(device), components[2])
    editor = register_attention(args.source_dir, components[0]); xm.mark_step()
    cache = C.PackCache(plan)
    limit_remaining = args.limit
    unit_rows = []
    if args.smoke:
        qualification_run_id = os.environ.get("QUALIFICATION_RUN_ID")
        if not qualification_run_id:
            raise ValueError("task-diverse smoke requires QUALIFICATION_RUN_ID")
        if args.phase != "val" or len(plan["packs"]) != 1:
            raise ValueError("task-diverse smoke requires the validation plan")
        selected_smoke = select_smoke_episodes(plan)
        pack_work = [(plan["packs"][0], selected_smoke)]
    else:
        pack_work = [(pack, plan["episodes"][pack["plan_range"][0]:pack["plan_range"][1]])
                     for pack in plan["packs"]]
    smoke_batch_seconds = []
    smoke_prediction_batches = []
    smoke_memory = []
    for pack, selected in pack_work:
        writer = C.UnitWriter(root=args.units_root, plan=plan, pack=pack, config=CONFIG)
        for offset in range(0, len(selected), BATCH):
            batch_eps = selected[offset:offset + BATCH]
            present = [writer.has(ep["key"]) for ep in batch_eps]
            if args.smoke and any(present):
                raise ValueError("task-diverse smoke requires a fresh result root")
            if all(present):
                continue
            if any(present):
                if not writer.discard_incomplete_batch([ep["key"] for ep in batch_eps]):
                    raise ValueError(f"{pack['rel']}: partial native batch rollback failed")
            if limit_remaining and limit_remaining < len(batch_eps):
                break
            rebuilt = [(ep, cache.rebuild(ep)) for ep in batch_eps]
            started = time.monotonic()
            predictions, noise_rows = predict_batch(components, load_batch(rebuilt), embeddings,
                                                     mask_np, device, mesh, editor)
            elapsed = time.monotonic() - started
            try:
                memory = {str(key): int(value) for key, value in xm.get_memory_info(device).items()}
            except Exception as error:
                memory = {"error": f"{type(error).__name__}: {error}"}
            output_rows = []
            for (episode, values), prediction, noise in zip(rebuilt, predictions, noise_rows):
                output_rows.append((episode, values, prediction, {**noise, **evidence,
                    "seconds_per_native_batch": elapsed, "batch": BATCH,
                    "native_prompt_count": 1, "grid": "official_A_Aprime_B_B_512",
                    "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
                    "steps": INFERENCE_STEPS, "strength": STRENGTH,
                    "guidance_scale": GUIDANCE_SCALE, "scale_sac": SCALE_SAC}))
                if limit_remaining:
                    limit_remaining -= 1
            writer.add_batch(output_rows)
            print(f"PROGRESS {pack['rel']} {len(writer.data['episodes'])}/{len(selected)}", flush=True)
            if args.smoke:
                smoke_batch_seconds.append(elapsed)
                smoke_prediction_batches.append(predictions[:len(rebuilt)].astype(np.float32))
                smoke_memory.append(memory)
            if args.limit and limit_remaining == 0:
                print("SMOKE_LIMIT_REACHED", flush=True)
                return 0
        if args.smoke:
            if (len(writer.data["episodes"]) != SMOKE_EPISODES
                    or len(smoke_batch_seconds) != math.ceil(SMOKE_EPISODES / BATCH)):
                raise ValueError("task-diverse smoke batch set is incomplete")
            values = np.concatenate(smoke_prediction_batches, axis=0)
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            smoke_summary = {
                "schema": "analogist_v5p8_q2r5_task_diverse_smoke_v1",
                "qualification_run_id": qualification_run_id,
                "plan_identity": plan["plan_identity"],
                "adapter_identity": C.identity(CONFIG["adapter"]),
                "expected_tpu": AUTHORIZED_TPU,
                "accelerator": EXPECTED_ACCELERATOR,
                "topology": EXPECTED_TOPOLOGY,
                "global_devices": runtime_devices,
                "addressable_devices": addressable,
                "workers": EXPECTED_WORKERS,
                "episodes": SMOKE_EPISODES,
                "tasks": [episode["task"] for episode in selected],
                "keys": [episode["key"] for episode in selected],
                "batch_seconds": smoke_batch_seconds,
                "last_batch_seconds": smoke_batch_seconds[-1],
                "prediction_min": int(values.min()),
                "prediction_max": int(values.max()),
                "prediction_std_min": min(float(row.std()) for row in values),
                "prediction_dynamic_range_min": min(float(np.percentile(row, 99) - np.percentile(row, 1)) for row in values),
                "prediction_mean_min": min(float(row.mean()) for row in values),
                "prediction_mean_max": max(float(row.mean()) for row in values),
                "memory_after_batches": smoke_memory,
                "rlimit_memlock_soft": "infinity" if soft == resource.RLIM_INFINITY else int(soft),
                "rlimit_memlock_hard": "infinity" if hard == resource.RLIM_INFINITY else int(hard),
            }
            C.atomic_json(args.smoke_marker, smoke_summary)
            print("SMOKE_COMPLETE " + json.dumps(smoke_summary, sort_keys=True), flush=True)
            return 0
        _, digest = writer.finalize()
        unit_rows.append({"pack_rel": pack["rel"], "unit": str(writer.root),
                          "episodes": len(pack["expected_episode_keys"]), "producer_sha256": digest,
                          "run_identity": writer.data["run_identity"]})
    marker = C.write_phase_marker(path=args.marker, plan=plan, config=CONFIG, units=unit_rows)
    print(json.dumps({"marker": str(args.marker), "episodes": marker["episodes"]}, sort_keys=True))
    return 0


def receipt(args) -> int:
    value = C.create_validation_receipt(plan_path=args.plan, marker_path=args.marker,
                                        unit=args.unit, out=args.out, config=CONFIG)
    print(json.dumps({"receipt": str(args.out), "sha256": C.sha256_file(args.out),
                      "identity": value["receipt_identity"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare-plan")
    plan.add_argument("--phase", choices=("val", "test"), required=True)
    plan.add_argument("--out", type=Path, required=True); plan.add_argument("--pack-cache", type=Path, required=True)
    plan.add_argument("--source-dir", type=Path, required=True); plan.add_argument("--model-dir", type=Path, required=True)
    plan.add_argument("--prompt-embeddings", type=Path, required=True); plan.add_argument("--helper-dir", type=Path, required=True)
    plan.add_argument("--val-receipt", type=Path); plan.set_defaults(function=prepare)
    execute = sub.add_parser("run")
    execute.add_argument("--phase", choices=("val", "test"), required=True)
    execute.add_argument("--plan", type=Path, required=True); execute.add_argument("--units-root", type=Path, required=True)
    execute.add_argument("--marker", type=Path, required=True); execute.add_argument("--source-dir", type=Path, required=True)
    execute.add_argument("--model-dir", type=Path, required=True); execute.add_argument("--prompt-embeddings", type=Path, required=True)
    execute.add_argument("--mask", type=Path, required=True); execute.add_argument("--helper-dir", type=Path, required=True)
    execute.add_argument("--cache-dir", type=Path, required=True); execute.add_argument("--expected-devices", type=int, default=EXPECTED_GLOBAL_DEVICES)
    execute.add_argument("--limit", type=int, default=0)
    execute.add_argument("--smoke", action="store_true")
    execute.add_argument("--smoke-marker", type=Path)
    execute.set_defaults(function=run)
    gate = sub.add_parser("validation-receipt")
    gate.add_argument("--plan", type=Path, required=True); gate.add_argument("--marker", type=Path, required=True)
    gate.add_argument("--unit", type=Path, required=True); gate.add_argument("--out", type=Path, required=True)
    gate.set_defaults(function=receipt)
    return root


if __name__ == "__main__":
    try:
        arguments = parser().parse_args()
        raise SystemExit(arguments.function(arguments))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Analogist Bench-v2 refused: {error}") from error
