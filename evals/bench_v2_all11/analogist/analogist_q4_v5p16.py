#!/usr/bin/env python3
"""Two-worker v5p-16 Analogist validation producer with eight candidates.

The episode plan remains the frozen q2 P=1 Bench-v2 plan.  This runner changes
only the parts required by the released Analogist evaluation contract: an
episode-specific B-prime text embedding, task-dependent strength/guidance, and
eight independently seeded candidates retained for independent scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import socket
import sys
import time

import numpy as np

import bench_v2_native_p1_common as C
import q2_analogist_bench_v2 as Q2
from analogist_loader import NEGATIVE_PROMPT, load_runtime_components


AUTHORIZED_TPU = "kmh-tpuvm-v5p-16-kaiminghe-8a2526"
EXPECTED_ACCELERATOR = "v5p-16"
EXPECTED_TOPOLOGY = "2x2x2"
EXPECTED_GLOBAL_DEVICES = 8
EXPECTED_ADDRESSABLE_DEVICES = 4
EXPECTED_WORKERS = 2
BATCH = 8
CANDIDATES = 8
INFERENCE_STEPS = 50
SCALE_SAC = 1.3
LATENT_SIZE = 64
PLAN_IDENTITY = "4ec5c77b046c6daf9eebde53a0004ed357d626c7a647d70e5f312fea550d01e5"
PROMPT_CACHE_IDENTITY = "1a4a19c801c230be7d6954ac4ee3e9dd1361a81e3ffa7781229483d6154ce37c"
PROMPT_CACHE_SHA256 = "44b7f5c794562da9f6e077b7849d550707df53e9ccd5b59757c744b71333c141"
PROMPT_INDEX_IDENTITY = "bf54996a13da864d7171dad5c12a7e90a2108f006be34c7c63856e33b893b84c"
PROMPT_INDEX_SHA256 = "5279bdc1f4f0deb4b4923cc9bd9df6c8045ce06d7bd2f1759c8f7bd8ddace626"
PROMPT_EMBEDDINGS_SHA256 = "e1e75186199abc2e5b67b989002274508e7b12c35ebb30d16a0991d1f4bb4e0a"
PROMPT_INPUTS_SHA256 = "5637621999336c2a2ebdfc10dc5e95ab867e1d22d162c1ddb7a27894d7b76d26"
PACK_JSON_SHA256 = "7a130a55c9928b3509d95d03f3a3c9a63ffb5331c03335b2bee211374bb59322"
PACK_NPZ_SHA256 = "d36685bbd8b95e720f03980b910712cd99d51f4933a4d1e9cf2e118d2235548e"
NOISE_NAMESPACE = "analogist_official_q3_v1"
SMOKE_KEYS = ("saliency_detection@q0", "denoising@q0", "inpainting@q0")
TASKS = {
    "colorization", "deblurring", "denoising", "edge_detection", "face_landmarks",
    "human_pose_2d", "inpainting", "instance_segmentation", "relative_depth",
    "saliency_detection", "semantic_segmentation", "super_resolution", "surface_normals",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_png(path: Path, value: np.ndarray) -> dict:
    from PIL import Image

    value = np.asarray(value, np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"invalid RGB artifact shape: {value.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(value, "RGB").save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "pixels_sha256": hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest(),
        "shape": list(value.shape),
        "dtype": "uint8",
    }


def settings_for_task(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"unsupported Analogist task: {task}")
    return {
        "num_inference_steps": INFERENCE_STEPS,
        "strength": 0.7 if task in {"deblurring", "denoising"} else 0.98,
        "guidance_scale": 7.5 if task == "inpainting" else 15.0,
        "scale_sac": SCALE_SAC,
        "sac_layers": [3, 11],
        "cam_layers": [3, 11],
        "candidates": CANDIDATES,
    }


def candidate_seed(episode_id: str, candidate: int, domain: str) -> int:
    if not 0 <= candidate < CANDIDATES or domain not in {"diffusion", "init_vae", "masked_vae"}:
        raise ValueError("candidate seed request is outside the released contract")
    material = f"{NOISE_NAMESPACE}|{episode_id}|candidate={candidate}|{domain}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little") & 0x7fff_ffff_ffff_ffff


def cpu_noise(shape: tuple[int, ...], seed: int):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def load_prompt_assets(cache_path: Path, index_path: Path, embedding_path: Path,
                       prompt_inputs_path: Path, plan: dict):
    from safetensors import safe_open

    if sha256_file(cache_path) != PROMPT_CACHE_SHA256:
        raise ValueError("current-vision prompt cache bytes differ")
    if sha256_file(index_path) != PROMPT_INDEX_SHA256:
        raise ValueError("current-vision prompt index bytes differ")
    if sha256_file(prompt_inputs_path) != PROMPT_INPUTS_SHA256:
        raise ValueError("prompt-input receipt bytes differ")
    cache = json.loads(cache_path.read_text())
    unsigned_cache = dict(cache); unsigned_cache.pop("cache_identity", None)
    if (cache.get("cache_identity") != PROMPT_CACHE_IDENTITY
            or cache["cache_identity"] != identity(unsigned_cache)
            or cache.get("plan_identity") != PLAN_IDENTITY
            or cache.get("exact_official_prompt_route") is not False
            or cache.get("source_kind") != "codex_current_vision_cache_v1"
            or cache.get("uses_query_target") is not False):
        raise ValueError("current-vision prompt cache contract differs")
    index = json.loads(index_path.read_text())
    unsigned_index = dict(index); unsigned_index.pop("index_identity", None)
    if (index.get("index_identity") != PROMPT_INDEX_IDENTITY
            or index["index_identity"] != identity(unsigned_index)
            or index.get("prompt_cache_identity") != PROMPT_CACHE_IDENTITY
            or index.get("embedding_file_sha256") != PROMPT_EMBEDDINGS_SHA256
            or sha256_file(embedding_path) != PROMPT_EMBEDDINGS_SHA256):
        raise ValueError("current-vision embedding index contract differs")
    plan_ids = [row["episode_id"] for row in plan["episodes"]]
    cache_ids = [row["episode_id"] for row in cache["entries"]]
    index_ids = [row["episode_id"] for row in index["entries"]]
    if plan_ids != cache_ids or plan_ids != index_ids or len(plan_ids) != 221:
        raise ValueError("prompt/plan episode ordering differs")
    for cache_row, index_row in zip(cache["entries"], index["entries"]):
        if (cache_row["conditioned_prompt_sha256"] != index_row["conditioned_prompt_sha256"]
                or cache_row["uses_query_target"] is not False
                or not 1 <= len(cache_row["raw_bprime_prompt"].split()) <= 5):
            raise ValueError(f"prompt entry differs: {cache_row['episode_id']}")
    with safe_open(str(embedding_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        expected_metadata = {
            key: str(index[key]) for key in (
                "schema", "prompt_cache_identity", "ordered_episode_ids_sha256",
                "model_repository", "model_revision", "asset_manifest_sha256", "dtype")
        }
        if metadata != expected_metadata:
            raise ValueError("prompt embedding metadata differs")
        positive = handle.get_tensor("positive")
        negative = handle.get_tensor("negative")
    if tuple(positive.shape) != (221, 77, 768) or tuple(negative.shape) != (1, 77, 768):
        raise ValueError("prompt embedding shapes differ")
    return cache, index, positive, negative


def load_candidate_arrays(episode: dict, rebuilt: dict):
    import torch

    grid = Q2.make_grid(rebuilt["demo_input"], rebuilt["demo_target"], rebuilt["query_input"])
    grids = np.repeat(grid[None], CANDIDATES, axis=0)
    noise_arrays: dict[str, list] = {"diffusion": [], "init_vae": [], "masked_vae": []}
    rows = []
    for candidate in range(CANDIDATES):
        row = {"candidate": candidate}
        for domain in noise_arrays:
            seed = candidate_seed(episode["episode_id"], candidate, domain)
            value = cpu_noise((1, 4, LATENT_SIZE, LATENT_SIZE), seed)
            noise_arrays[domain].append(value)
            row[f"{domain}_seed"] = seed
            row[f"{domain}_sha256"] = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        rows.append(row)
    image = grids.astype(np.float32).transpose(0, 3, 1, 2) / 127.5 - 1.0
    return image, *(torch.cat(noise_arrays[name]).numpy()
                    for name in ("diffusion", "init_vae", "masked_vae")), rows, grid


def timestep_rows(scheduler, strength: float):
    scheduler.set_timesteps(INFERENCE_STEPS, device="cpu")
    start = max(INFERENCE_STEPS - min(int(INFERENCE_STEPS * strength), INFERENCE_STEPS), 0)
    timesteps = [int(value) for value in scheduler.timesteps[start:].tolist()]
    expected = 35 if strength == 0.7 else 49
    if len(timesteps) != expected:
        raise ValueError(f"Analogist strength/timestep contract changed: {strength=} {len(timesteps)=}")
    alphas = scheduler.alphas_cumprod.float().cpu()
    final_alpha = float(scheduler.final_alpha_cumprod)
    stride = scheduler.config.num_train_timesteps // INFERENCE_STEPS
    return [(step, float(alphas[step]), float(alphas[step - stride]) if step >= stride else final_alpha)
            for step in timesteps]


def process_rss_bytes() -> int:
    # Linux ru_maxrss is KiB and remains a useful fail-closed fallback when the
    # SPMD virtual device refuses the torch-xla HBM query.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def hbm_evidence(xm, device) -> dict:
    attempts = []
    candidates = [str(device), "TPU:0"]
    try:
        candidates.extend(str(value) for value in xm.xla_real_devices())
    except Exception as error:
        attempts.append({"device": "xla_real_devices", "error": f"{type(error).__name__}: {error}"})
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            values = {str(key): int(value) for key, value in xm.get_memory_info(candidate).items()}
            if values.get("bytes_limit", 0) > 0 and values.get("bytes_in_use", -1) >= 0:
                return {"supported": True, "device": candidate, **values, "attempts": attempts}
            attempts.append({"device": candidate, "error": "non-numeric or empty memory report"})
        except Exception as error:
            attempts.append({"device": candidate, "error": f"{type(error).__name__}: {error}"})
    return {"supported": False, "attempts": attempts,
            "fallback": "process_peak_rss_bytes", "process_peak_rss_bytes": process_rss_bytes()}


def predict(components, arrays, positive, negative, mask_np, settings, device, mesh, editor):
    import torch
    import torch.nn.functional as functional
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    unet, vae, scheduler = components
    image_np, diffusion_np, init_eps_np, masked_eps_np, noise_rows, grid = arrays
    dtype = torch.bfloat16
    image = torch.from_numpy(image_np).to(device=device, dtype=dtype)
    mask = torch.from_numpy(mask_np).repeat(BATCH, 1, 1, 1).to(device=device, dtype=dtype)
    masked = image * (mask < 0.5)
    diffusion = torch.from_numpy(diffusion_np).to(device=device, dtype=dtype)
    init_eps = torch.from_numpy(init_eps_np).to(device=device, dtype=dtype)
    masked_eps = torch.from_numpy(masked_eps_np).to(device=device, dtype=dtype)
    for tensor in (image, mask, masked, diffusion, init_eps, masked_eps):
        xs.mark_sharding(tensor, mesh, ("data",) + (None,) * (tensor.ndim - 1))
    positive = positive.repeat(BATCH, 1, 1).to(device=device, dtype=dtype)
    negative = negative.repeat(BATCH, 1, 1).to(device=device, dtype=dtype)
    prompt = torch.cat((negative, positive), dim=0)
    xs.mark_sharding(prompt, mesh, ("data", None, None))
    editor.reset()
    step_rss = []
    with torch.no_grad():
        init_dist = vae.encode(image, return_dict=False)[0]
        image_latents = (init_dist.mean + init_dist.std * init_eps) * float(vae.config.scaling_factor)
        masked_dist = vae.encode(masked, return_dict=False)[0]
        masked_latents = (masked_dist.mean + masked_dist.std * masked_eps) * float(vae.config.scaling_factor)
        mask_latents = functional.interpolate(mask, size=(LATENT_SIZE, LATENT_SIZE))
        mask_guided = torch.cat((mask_latents, mask_latents), dim=0)
        masked_guided = torch.cat((masked_latents, masked_latents), dim=0)
        for tensor in (image_latents, masked_latents, mask_latents, mask_guided, masked_guided):
            xs.mark_sharding(tensor, mesh, ("data",) + (None,) * (tensor.ndim - 1))
        schedule = timestep_rows(scheduler, settings["strength"])
        _, first_alpha, _ = schedule[0]
        latents = math.sqrt(first_alpha) * image_latents + math.sqrt(1.0 - first_alpha) * diffusion
        for timestep, alpha, alpha_previous in schedule:
            guided = torch.cat((latents, latents), dim=0)
            latent_input = torch.cat((guided, mask_guided, masked_guided), dim=1)
            xs.mark_sharding(latent_input, mesh, ("data", None, None, None))
            noise_guided = unet(
                latent_input, torch.tensor(timestep, device=device, dtype=torch.int64),
                encoder_hidden_states=prompt, return_dict=False)[0]
            unconditioned, conditioned = noise_guided.chunk(2)
            noise = unconditioned + settings["guidance_scale"] * (conditioned - unconditioned)
            latents = Q2.step_latents(latents, noise, alpha, alpha_previous, scheduler)
            xs.mark_sharding(latents, mesh, ("data", None, None, None))
            xm.mark_step()
            step_rss.append(process_rss_bytes())
        decoded = vae.decode(latents / float(vae.config.scaling_factor), return_dict=False)[0]
        decoded = (decoded / 2.0 + 0.5).clamp(0, 1)
        xm.mark_step()
        decoded = decoded.float().cpu()[:, :, 256:, 256:]
        images = (decoded * 255.0).round().clamp(0, 255).to(torch.uint8)
    return (images.permute(0, 2, 3, 1).numpy(), noise_rows, grid,
            {"steps": len(step_rss), "max_process_rss_bytes": max(step_rss)})


def write_episode(root: Path, episode: dict, rebuilt: dict, predictions: np.ndarray,
                  noise_rows: list[dict], grid: np.ndarray, cache_row: dict,
                  settings: dict, elapsed: float, runtime: dict) -> dict:
    epdir = root / "episodes" / f"{episode['task']}_q{episode['q']:02d}"
    artifacts = {
        "demo_input": atomic_png(epdir / "demo_input.png", rebuilt["demo_input"]),
        "demo_target": atomic_png(epdir / "demo_target.png", rebuilt["demo_target"]),
        "query_input": atomic_png(epdir / "query_input.png", rebuilt["query_input"]),
        "query_target": atomic_png(epdir / "query_target.png", rebuilt["query_target"]),
        "model_grid": atomic_png(epdir / "model_grid.png", grid),
        "candidates": [],
    }
    for candidate, prediction in enumerate(predictions):
        artifacts["candidates"].append(
            atomic_png(epdir / f"candidate_{candidate:02d}.png", prediction))
    row = {
        "schema": "analogist_current_vision_candidate_episode_v1",
        "episode_id": episode["episode_id"], "key": episode["key"],
        "task": episode["task"], "q": episode["q"], "p": 1,
        "query_record": episode["query_record"], "demo_records": episode["demo_records"],
        "episode_pixels_sha256": episode["episode_pixels_sha256"],
        "prompt_source_kind": "codex_current_vision_cache_v1",
        "exact_official_prompt_route": False,
        "raw_bprime_prompt": cache_row["raw_bprime_prompt"],
        "conditioned_prompt": cache_row["conditioned_prompt"],
        "conditioned_prompt_sha256": cache_row["conditioned_prompt_sha256"],
        "marked_grid_pixels_sha256": cache_row["marked_grid_pixels_sha256"],
        "uses_query_target_for_prompt": False,
        "preprocessing": "uint8 RGB / 127.5 - 1.0",
        "output_crop": [256, 256, 512, 512],
        "negative_prompt": NEGATIVE_PROMPT,
        "settings": settings,
        "candidate_seeds": noise_rows,
        "seconds": elapsed,
        "runtime": runtime,
        "artifacts": artifacts,
    }
    row["episode_identity"] = identity(row)
    return row


def run(args: argparse.Namespace) -> int:
    plan = C.load_plan(args.plan, phase="val", config=Q2.CONFIG)
    if plan["plan_identity"] != PLAN_IDENTITY:
        raise ValueError("frozen validation plan identity differs")
    if (sha256_file(args.pack_dir / "samples.json") != PACK_JSON_SHA256
            or sha256_file(args.pack_dir / "samples.npz") != PACK_NPZ_SHA256):
        raise ValueError("bundled validation pack differs")
    if len(plan.get("packs", [])) != 1:
        raise ValueError("frozen validation plan pack count differs")
    # The authenticated q2 plan identity includes its historical staging path.
    # Rebind only after validating that frozen identity; the bundled pack bytes
    # above are the authoritative runtime copy.
    plan["packs"][0]["local_dir"] = str(args.pack_dir.resolve())
    source_identity = Q2.verify_runtime(
        args.source_dir, args.model_dir, args.legacy_prompt_embeddings, args.helper_dir)
    if plan.get("source_identity") != source_identity:
        raise ValueError("runtime source/assets differ from frozen plan")
    cache, index, all_positive, negative = load_prompt_assets(
        args.prompt_cache, args.prompt_index, args.prompt_embeddings,
        args.prompt_inputs, plan)
    cache_by_id = {row["episode_id"]: row for row in cache["entries"]}
    embedding_row = {row["episode_id"]: int(row["row"]) for row in index["entries"]}

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from PIL import Image

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    global_devices = int(xr.global_runtime_device_count())
    addressable = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if (backend.upper() != "TPU" or global_devices != EXPECTED_GLOBAL_DEVICES
            or addressable != EXPECTED_ADDRESSABLE_DEVICES
            or os.environ.get("EXPECTED_TPU_NAME") != AUTHORIZED_TPU):
        raise ValueError(
            f"refusing fallback/topology: {backend=} {global_devices=} {addressable=}")
    mesh = xs.Mesh(np.arange(EXPECTED_GLOBAL_DEVICES), (EXPECTED_GLOBAL_DEVICES,), ("data",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    runtime = {
        "backend": backend, "device": str(device), "global_devices": global_devices,
        "addressable_devices": addressable, "workers": EXPECTED_WORKERS,
        "accelerator": EXPECTED_ACCELERATOR, "topology": EXPECTED_TOPOLOGY,
        "host": socket.gethostname(), "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
        "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE"), "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(runtime, sort_keys=True), flush=True)
    if args.mask.resolve() != (args.source_dir / "assets/mask_image_inpainting.png").resolve():
        raise ValueError("mask path is outside authenticated Analogist source")
    mask_image = Image.open(args.mask).convert("L")
    if mask_image.size != (512, 512):
        raise ValueError("official Analogist mask dimensions differ")
    mask_np = (np.asarray(mask_image, np.float32)[None, None] / 255.0 >= 0.5).astype(np.float32)
    components = load_runtime_components(args.source_dir, args.model_dir, dtype=torch.bfloat16)
    components = (components[0].eval().to(device), components[1].eval().to(device), components[2])
    editor = Q2.register_attention(args.source_dir, components[0])
    xm.mark_step()
    cache_pack = C.PackCache(plan)
    selected = ([next(row for row in plan["episodes"] if row["key"] == key)
                 for key in SMOKE_KEYS] if args.smoke else plan["episodes"])
    args.results_root.mkdir(parents=True, exist_ok=True)
    partial_path = args.results_root / "results.partial.json"
    final_path = args.results_root / "results.json"
    if final_path.exists():
        raise ValueError("result root is already finalized")
    if partial_path.exists():
        result = json.loads(partial_path.read_text())
    else:
        result = {
            "schema": "analogist_current_vision_v5p16_results_v1",
            "phase": "qualification" if args.smoke else "validation",
            "run_id": os.environ.get("PHASE_RUN_ID"),
            "plan_identity": PLAN_IDENTITY,
            "prompt_cache_identity": PROMPT_CACHE_IDENTITY,
            "prompt_embedding_sha256": PROMPT_EMBEDDINGS_SHA256,
            "source_identity": source_identity,
            "runtime": runtime,
            "candidate_readout": {
                "count": 8, "retain_all": True,
                "target_based_selection_forbidden": True,
                "pixel_averaging_forbidden": True,
                "metric_aggregation": "score each candidate independently, then arithmetic mean per episode",
            },
            "expected_keys": [row["key"] for row in selected],
            "episodes": {}, "complete": False,
        }
        result["run_identity"] = identity(result)
        atomic_json(partial_path, result)
    if (result.get("plan_identity") != PLAN_IDENTITY
            or result.get("prompt_cache_identity") != PROMPT_CACHE_IDENTITY
            or result.get("expected_keys") != [row["key"] for row in selected]):
        raise ValueError("stale result root identity")
    hbm_rows = []
    batch_seconds = []
    for episode_number, episode in enumerate(selected, 1):
        if episode["key"] in result["episodes"]:
            continue
        rebuilt = cache_pack.rebuild(episode)
        settings = settings_for_task(episode["task"])
        row_index = embedding_row[episode["episode_id"]]
        started = time.monotonic()
        predictions, noise_rows, grid, step_memory = predict(
            components, load_candidate_arrays(episode, rebuilt),
            all_positive[row_index:row_index + 1], negative,
            mask_np, settings, device, mesh, editor)
        elapsed = time.monotonic() - started
        memory = hbm_evidence(xm, device)
        memory["process_peak_rss_bytes"] = process_rss_bytes()
        memory["diffusion_step_memory"] = step_memory
        hbm_rows.append(memory); batch_seconds.append(elapsed)
        output = write_episode(
            args.results_root, episode, rebuilt, predictions, noise_rows, grid,
            cache_by_id[episode["episode_id"]], settings, elapsed,
            {**runtime, "memory_after_batch": memory})
        # Make artifact paths portable relative to the result root.
        portable_artifacts = [
            output["artifacts"][name]
            for name in ("demo_input", "demo_target", "query_input", "query_target", "model_grid")
        ]
        portable_artifacts.extend(output["artifacts"]["candidates"])
        for artifact in portable_artifacts:
            artifact["path"] = str(Path(artifact["path"]).relative_to(args.results_root))
        # write_episode authenticates the just-written images, but its first
        # identity necessarily contains host-local absolute paths.  Rebind the
        # episode after those paths have been made portable so the identity in
        # the finalized result is independently reproducible.
        output["episode_identity"] = identity(
            {key: value for key, value in output.items() if key != "episode_identity"})
        result["episodes"][episode["key"]] = output
        atomic_json(partial_path, result)
        print(f"PROGRESS {episode_number}/{len(selected)} key={episode['key']} seconds={elapsed:.3f}", flush=True)
    if set(result["episodes"]) != set(result["expected_keys"]):
        raise ValueError("result coverage differs")
    batch_seconds = [float(result["episodes"][key]["seconds"])
                     for key in result["expected_keys"]]
    hbm_rows = [result["episodes"][key]["runtime"]["memory_after_batch"]
                for key in result["expected_keys"]]
    result["complete"] = True
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["batch_seconds"] = batch_seconds
    result["memory_after_batches"] = hbm_rows
    result["result_identity"] = identity({key: value for key, value in result.items()
                                          if key != "result_identity"})
    atomic_json(final_path, result)
    partial_path.unlink()
    marker = {
        "schema": "analogist_current_vision_v5p16_complete_v1",
        "status": "PASS", "phase": result["phase"], "run_id": result["run_id"],
        "host": socket.gethostname(), "episodes": len(result["episodes"]),
        "candidates": len(result["episodes"]) * CANDIDATES,
        "plan_identity": PLAN_IDENTITY, "prompt_cache_identity": PROMPT_CACHE_IDENTITY,
        "result_sha256": sha256_file(final_path), "result_identity": result["result_identity"],
        "batch_seconds": batch_seconds, "memory_after_batches": hbm_rows,
        "prediction_min": min(int(np.asarray(Image.open(args.results_root / row["artifacts"]["candidates"][0]["path"])).min())
                              for row in result["episodes"].values()),
        "prediction_max": max(int(np.asarray(Image.open(args.results_root / row["artifacts"]["candidates"][0]["path"])).max())
                              for row in result["episodes"].values()),
    }
    marker["marker_identity"] = identity(marker)
    atomic_json(args.marker, marker)
    print("RUN_COMPLETE " + json.dumps(marker, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--plan", type=Path, required=True)
    root.add_argument("--results-root", type=Path, required=True)
    root.add_argument("--marker", type=Path, required=True)
    root.add_argument("--source-dir", type=Path, required=True)
    root.add_argument("--model-dir", type=Path, required=True)
    root.add_argument("--legacy-prompt-embeddings", type=Path, required=True)
    root.add_argument("--helper-dir", type=Path, required=True)
    root.add_argument("--mask", type=Path, required=True)
    root.add_argument("--prompt-cache", type=Path, required=True)
    root.add_argument("--prompt-index", type=Path, required=True)
    root.add_argument("--prompt-embeddings", type=Path, required=True)
    root.add_argument("--prompt-inputs", type=Path, required=True)
    root.add_argument("--pack-dir", type=Path, required=True)
    root.add_argument("--cache-dir", type=Path, required=True)
    root.add_argument("--smoke", action="store_true")
    return root


if __name__ == "__main__":
    try:
        raise SystemExit(run(parser().parse_args()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"Analogist q4 refused: {error}") from error
