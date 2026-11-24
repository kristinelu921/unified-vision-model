#!/usr/bin/env python3
"""Frozen task-text VisualCloze P=2 runner for Bench v2."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import statistics
import sys
import time
import types
from pathlib import Path


LIVE_REPO = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
PACK_URI_ROOT = "gs://kmh-gcp-us-west4/data/xtiange/mt_eval_packs"
TRUSTED_MANIFEST = LIVE_REPO / "docs/bench_v2_audit/manifest_v1.json"
TRUSTED_MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"
MODEL_ID = "VisualCloze/VisualClozePipeline-384"
MODEL_REVISION = "59c469d2772d927ffe55f3543c4d3bd556fd46a4"
DIFFUSERS_REVISION = "0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec"
TASK_PROMPTS = json.loads((Path(__file__).parent / 'task_prompts.json').read_text())
CONTENT_PROMPT = ""
READOUT = "VC@f30"
ARM = "VC"
READOUT_TAG = "f30"
SEED = 42
P = 2
STEPS = 30
GUIDANCE = 30.0
MODEL_SEED = 0
SCHEMA = "visualcloze_bench_v2_p2_task_text_v1_inference_v1"
PLAN_SCHEMA = "visualcloze_bench_v2_p2_task_text_v1_plan_v1"

SOURCE_FILES = {
    "runner": Path(__file__).resolve(),
    "task_prompts": Path(__file__).parent / "task_prompts.json",
    "runtime_v1": LIVE_REPO / "tools/bench_v2_runtime_v1.py",
    "runtime_v2": LIVE_REPO / "tools/bench_v2_runtime_v2.py",
    "episode_builder": LIVE_REPO / "jflux2/analysis/capacity_ladder_v9.py",
    "episode_base": LIVE_REPO / "jflux2/analysis/capacity_ladder_v6.py",
    "episode_contract": LIVE_REPO / "jflux2/analysis/t16_offline_eval_v1.py",
    "episode_side_effect": LIVE_REPO / "jflux2/analysis/ttt_ladder_v1.py",
    "conventions": LIVE_REPO / "jflux2/dataloaders/multi_task_loader_v4.py",
    "numpy_episode_replay": LIVE_REPO / "tools/bench_episode_v1.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array) -> str:
    import numpy as np

    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def episode_sha256(demo_inputs, demo_targets, query_input, query_target) -> str:
    import numpy as np

    h = hashlib.sha256()
    for array in (*demo_inputs, *demo_targets, query_input, query_target):
        h.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_FILES.items()}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o666)


def atomic_png(path: Path, image) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)
    os.chmod(path, 0o666)


def descriptor(path: Path, root: Path) -> dict:
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        pixels = sha256_array(__import__("numpy").asarray(rgb))
        shape = [rgb.height, rgb.width, 3]
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "pixels_sha256": pixels,
        "shape": shape,
        "dtype": "uint8",
        "mode": "RGB",
    }


def task_rows(pack: dict, task: str) -> list[int]:
    return [i for i, row in enumerate(pack["records"]) if row["task"] == task]


def pack_slug(rel: str) -> str:
    return rel.replace("/", "__")


def verify_model_assets(model_dir: Path, sums_path: Path) -> str:
    listed = {}
    for line in sums_path.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip("*").removeprefix("./")
        if rel in listed or not rel:
            raise SystemExit("invalid duplicate/empty model checksum entry")
        listed[rel] = digest
    actual = {
        str(p.relative_to(model_dir))
        for p in model_dir.rglob("*")
        if p.is_file() and p.resolve() != sums_path.resolve()
    }
    if actual != set(listed):
        raise SystemExit(
            f"model asset file set mismatch: missing={sorted(set(listed)-actual)[:5]} "
            f"extra={sorted(actual-set(listed))[:5]}"
        )
    for rel, expected in sorted(listed.items()):
        got = sha256_file(model_dir / rel)
        if got != expected:
            raise SystemExit(f"model asset hash mismatch: {rel}: {got} != {expected}")
    return sha256_file(sums_path)


class PackCache:
    def __init__(self, plan: dict):
        self.by_rel = {p["rel"]: p for p in plan["packs"]}
        self.rel = None
        self.images = self.masks = None

    def load(self, rel: str):
        import numpy as np

        if rel == self.rel:
            return self.images, self.masks
        self.images = self.masks = None
        gc.collect()
        info = self.by_rel[rel]
        with np.load(Path(info["local_dir"]) / "samples.npz", allow_pickle=False) as z:
            self.images = z["images"]
            self.masks = z["masks"]
        self.images.setflags(write=False)
        self.masks.setflags(write=False)
        self.rel = rel
        print(f"[infer] loaded {rel}: {len(self.images)} rows", flush=True)
        return self.images, self.masks


def build_grid(ep: dict, packs: PackCache):
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(LIVE_REPO))
    from tools import bench_episode_v1 as BE

    images, masks = packs.load(ep["pack_rel"])
    demo_inputs = [np.asarray(images[i], np.uint8) for i in ep["demo_global"]]
    raw_targets = [np.asarray(masks[i], np.uint8) for i in ep["demo_global"]]
    raw_targets.append(np.asarray(masks[ep["query_global"]], np.uint8))
    coded = BE.code_episode(raw_targets, ep["task"], int(ep["cid"]))
    demo_targets, query_target = coded[:-1], coded[-1]
    query_input = np.asarray(images[ep["query_global"]], np.uint8)
    got = episode_sha256(demo_inputs, demo_targets, query_input, query_target)
    if got != ep["episode_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: replayed episode pixels changed: {got}")
    if sha256_array(query_input) != ep["query_input_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: query input changed")
    if sha256_array(query_target) != ep["query_target_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: coded query target changed")
    rows = [[Image.fromarray(a), Image.fromarray(b)] for a, b in zip(demo_inputs, demo_targets)]
    rows.append([Image.fromarray(query_input), None])
    return rows, query_input, query_target


def initial_view(pack: dict, plan: dict) -> dict:
    return {
        "schema": SCHEMA,
        "protocol": SCHEMA,
        "args": {
            "pack": pack["local_dir"],
            "p": P,
            "seed": SEED,
            "steps": STEPS,
            "guidance": GUIDANCE,
            "model_seed": MODEL_SEED,
            "condition": "real",
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusers_revision": DIFFUSERS_REVISION,
        },
        "data_identity": pack["data_identity"],
        "data_identity_sha256": pack["data_identity_sha256"],
        "plan_identity": plan["plan_identity"],
        "source_sha256": plan["source_sha256"],
        "expected_episode_keys": pack["expected_episode_keys"],
        "episodes": {},
        "complete": False,
    }


def checked_views(results_root: Path, plan: dict) -> tuple[dict[str, dict], set[str]]:
    views, completed = {}, set()
    for pack in plan["packs"]:
        root = results_root / "views" / pack["slug"]
        path = root / "results.json"
        view = json.loads(path.read_text()) if path.exists() else initial_view(pack, plan)
        if view.get("plan_identity") != plan["plan_identity"]:
            raise SystemExit(f"{path}: plan identity differs")
        if view.get("data_identity") != pack["data_identity"]:
            raise SystemExit(f"{path}: pack identity differs")
        bad_missing = []
        for key, ep in list(view["episodes"].items()):
            arts = ep.get("artifacts", {})
            entries = [arts.get("query_input"), arts.get("query_target")]
            entries.append(((arts.get("readouts", {}).get(READOUT) or {}).get("real")))
            missing = any(not d or not (root / d["path"]).is_file() for d in entries)
            if missing:
                bad_missing.append(key)
                del view["episodes"][key]
                continue
            for d in entries:
                got = sha256_file(root / d["path"])
                if got != d["sha256"]:
                    raise SystemExit(f"{root / d['path']}: saved artifact hash mismatch")
            completed.add(f"{pack['rel']}|{key}")
        if bad_missing:
            view["complete"] = False
            atomic_json(path, view)
            print(f"[resume] {pack['rel']}: discarded {len(bad_missing)} incomplete journal entries", flush=True)
        views[pack["rel"]] = view
    return views, completed


def save_episode(
    ep: dict,
    raw_output,
    query_input,
    query_target,
    pack: dict,
    view: dict,
    results_root: Path,
    batch_seconds: float,
    model_assets_sha256: str,
) -> None:
    from PIL import Image

    root = results_root / "views" / pack["slug"]
    episode_dir = root / f"{ep['task']}_q{ep['q']}"
    query_input_path = episode_dir / "query_input.png"
    query_target_path = episode_dir / "query_target.png"
    prediction_path = episode_dir / f"{ARM}_real_{READOUT_TAG}_query.png"
    raw_path = episode_dir / f"{ARM}_real_{READOUT_TAG}_raw384.png"
    atomic_png(query_input_path, Image.fromarray(query_input))
    atomic_png(query_target_path, Image.fromarray(query_target))
    atomic_png(raw_path, raw_output)
    atomic_png(prediction_path, raw_output.resize((256, 256), Image.Resampling.LANCZOS))
    record = {
        "task": ep["task"],
        "q": int(ep["q"]),
        "p": P,
        "cid": int(ep["cid"]),
        "query_record": ep["query_record"],
        "demo_records": ep["demo_records"],
        "prediction_complete": True,
        "complete": True,
        "data_identity": pack["data_identity"],
        "readouts": {
            READOUT: {
                "real": {
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "steps": STEPS,
                    "guidance": GUIDANCE,
                    "model_seed": MODEL_SEED,
                    "task_prompt": TASK_PROMPTS[ep["task"]],
                    "content_prompt": CONTENT_PROMPT,
                    "batch_seconds": batch_seconds,
                    "native_prompt_count": P,
                    "ensemble": False,
                    "model_assets_sha256s_sha256": model_assets_sha256,
                }
            }
        },
        "artifacts": {
            "query_input": descriptor(query_input_path, root),
            "query_target": descriptor(query_target_path, root),
            "readouts": {READOUT: {"real": descriptor(prediction_path, root)}},
            "raw384": descriptor(raw_path, root),
        },
    }
    view["episodes"][ep["key"]] = record


def memory_info(xm, device) -> dict:
    try:
        value = xm.get_memory_info(device)
        return {str(k): int(v) for k, v in value.items()}
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"error": f"{type(exc).__name__}: {exc}"}


def infer(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.experimental.custom_kernel import FlashAttention
    from diffusers import VisualClozeGenerationPipeline

    plan = json.loads(args.plan.read_text())
    if plan.get("schema") != PLAN_SCHEMA or plan.get("protocol") != SCHEMA:
        raise SystemExit("unsupported plan")
    expected_identity = plan.pop("plan_identity")
    got_identity = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    plan["plan_identity"] = expected_identity
    if got_identity != expected_identity:
        raise SystemExit(f"plan identity mismatch: {got_identity} != {expected_identity}")
    if plan["source_sha256"] != source_hashes():
        raise SystemExit("source bytes changed after plan creation")
    if plan["task_prompts"] != TASK_PROMPTS or plan["content_prompt"] != CONTENT_PROMPT:
        raise SystemExit("frozen task text differs")
    if len(plan["packs"]) != 1 or plan["packs"][0]["rel"] != "bench_v2_val":
        raise SystemExit("validation only")
    if sum(not e["padding"] for e in plan["episodes"]) != 221:
        raise SystemExit("expected exactly 221 queries")
    if any(int(ep["p"]) != P for ep in plan["episodes"]):
        raise SystemExit("plan contains a non-P2 episode")
    if len(plan["episodes"]) % args.batch_size:
        raise SystemExit("planned episode count is not batch divisible")
    if args.batch_size != 2 or args.expected_devices != 8:
        raise SystemExit("the frozen eight-chip mesh requires batch size 2 and eight devices")
    if os.environ.get("EXPECTED_TPU_NAME") != args.expected_tpu:
        raise SystemExit("TPU name environment does not match the launch contract")
    if os.environ.get("EXPECTED_TPU_ZONE") != args.expected_zone:
        raise SystemExit("TPU zone environment does not match the launch contract")

    model_assets_sha256 = verify_model_assets(args.model_dir, args.model_sha256s)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    compiler_cache = args.cache_dir / "xla_compile_cache"
    compiler_cache.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(compiler_cache), readonly=False)
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    addressable_devices = xr.addressable_runtime_device_count()
    backend = str(xr.device_type())
    if backend.upper() != "TPU" or num_devices != 8 or addressable_devices != 8:
        raise SystemExit(
            f"refusing fallback: backend={backend!r}, devices={num_devices}, addressable={addressable_devices}"
        )
    mesh = xs.Mesh(np.arange(num_devices), (2, 4), ("data", "model"))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {
        "backend": backend,
        "device": str(device),
        "runtime_device_count": int(num_devices),
        "addressable_runtime_device_count": int(addressable_devices),
        "spmd_mesh_shape": [2, 4],
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "tpu_name": args.expected_tpu,
        "tpu_zone": args.expected_zone,
        "spot": True,
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    results_root = args.results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    views, completed = checked_views(results_root, plan)
    packs_by_rel = {p["rel"]: p for p in plan["packs"]}
    cache = PackCache(plan)
    first = next(ep for ep in plan["episodes"] if not ep["padding"])
    first_grid, _, _ = build_grid(first, cache)
    first_grids = [first_grid, first_grid]

    text_pipe = VisualClozeGenerationPipeline.from_pretrained(
        str(args.model_dir),
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        resolution=384,
    ).to("cpu")
    embedding_cache = {}
    text_audit = {}
    for task, task_text in TASK_PROMPTS.items():
        processed = text_pipe.image_processor.preprocess(
            [task_text] * 2, [CONTENT_PROMPT] * 2, first_grids,
            vae_scale_factor=text_pipe.vae_scale_factor,
        )
        with torch.no_grad():
            encoded = text_pipe.encode_prompt(
                layout_prompt=processed["layout_prompt"],
                task_prompt=processed["task_prompt"],
                content_prompt=processed["content_prompt"],
                device=torch.device("cpu"), max_sequence_length=512,
            )
        embedding_cache[task] = tuple(x.detach().clone() for x in encoded)
        assert torch.equal(encoded[0][0], encoded[0][1])
        assert torch.equal(encoded[1][0], encoded[1][1])
        text_audit[task] = {
            "task_prompt": task_text, "content_prompt": CONTENT_PROMPT,
            "processed": {k: processed[k] for k in ("layout_prompt", "task_prompt", "content_prompt")},
            "embedding_sha256": [hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in encoded],
            "embedding_shapes": [list(x.shape) for x in encoded],
        }
        print("TEXT_EMBEDDED " + task, flush=True)
    atomic_json(results_root / "text_embeddings_audit.json", text_audit)
    del processed, text_pipe, first_grids, first_grid, encoded
    gc.collect()

    pipe = VisualClozeGenerationPipeline.from_pretrained(
        str(args.model_dir),
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        resolution=384,
    )
    print("TPU_STAGE model_to_device_lazy_begin", flush=True)
    pipe = pipe.to(device)
    # Keep the CPU->TPU transfers lazy until every large transformer parameter
    # carries its 4-way model-sharding annotation.  Materializing here first
    # would transiently replicate the full model on each 16 GiB v5e device.
    pipe.set_progress_bar_config(disable=True)
    sharded = 0
    for _, param in pipe.transformer.named_parameters():
        if param.dim() >= 2:
            spec = [None] * param.dim()
            largest = max(range(param.dim()), key=lambda d: param.shape[d])
            spec[largest] = "model"
            xs.mark_sharding(param, mesh, tuple(spec))
            sharded += 1
    print(f"TPU_STAGE lazy_sharding_annotations_complete count={sharded}", flush=True)
    pipe.transformer.enable_xla_flash_attention(
        partition_spec=("data", None, None, None), is_flux=True
    )
    FlashAttention.DEFAULT_BLOCK_SIZES = {
        key: int(args.flash_block_size)
        for key in (
            "block_q",
            "block_k_major",
            "block_k",
            "block_b",
            "block_q_major_dkv",
            "block_k_major_dkv",
            "block_q_dkv",
            "block_k_dkv",
            "block_q_dq",
            "block_k_dq",
            "block_k_major_dq",
        )
    }
    prompt_embeds = pooled_prompt_embeds = text_ids = None

    def fixed_encode_prompt(_self, *unused_args, **unused_kwargs):
        return prompt_embeds, pooled_prompt_embeds, text_ids

    pipe.encode_prompt = types.MethodType(fixed_encode_prompt, pipe)
    torch.manual_seed(MODEL_SEED)
    xm.set_rng_state(MODEL_SEED, device)
    total_real = sum(not ep["padding"] for ep in plan["episodes"])
    start_completed = len(completed)
    batch_times = []
    batches = [plan["episodes"][i : i + 2] for i in range(0, len(plan["episodes"]), 2)]
    for batch_index, batch in enumerate(batches):
        real_batch = [ep for ep in batch if not ep["padding"]]
        ids = [f"{ep['pack_rel']}|{ep['key']}" for ep in real_batch]
        if ids and all(i in completed for i in ids):
            continue
        grids, pixels = [], []
        for ep in batch:
            grid, query_input, query_target = build_grid(ep, cache)
            grids.append(grid)
            pixels.append((query_input, query_target))
        batch_tasks = [ep["task"] for ep in batch]
        prompt_embeds = torch.cat([embedding_cache[t][0][0:1] for t in batch_tasks], dim=0).to(device)
        pooled_prompt_embeds = torch.cat([embedding_cache[t][1][0:1] for t in batch_tasks], dim=0).to(device)
        text_ids = embedding_cache[batch_tasks[0]][2].to(device)
        assert all(torch.equal(embedding_cache[t][2], embedding_cache[batch_tasks[0]][2]) for t in batch_tasks)
        xs.mark_sharding(prompt_embeds, mesh, ("data", None, None))
        xs.mark_sharding(pooled_prompt_embeds, mesh, ("data", None))
        started = time.perf_counter()
        outputs = pipe(
            task_prompt=[TASK_PROMPTS[ep["task"]] for ep in batch],
            content_prompt=[CONTENT_PROMPT] * 2,
            image=grids,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            max_sequence_length=512,
            generator=[torch.Generator("cpu").manual_seed(MODEL_SEED) for _ in batch],
        ).images
        if len(outputs) != 2 or any(len(crop) != 1 or crop[0].size != (384, 384) for crop in outputs):
            raise RuntimeError("unexpected output dimensions")
        elapsed = time.perf_counter() - started
        batch_times.append(elapsed)
        touched = set()
        for ep, cropped, (query_input, query_target) in zip(batch, outputs, pixels):
            if ep["padding"]:
                continue
            identity = f"{ep['pack_rel']}|{ep['key']}"
            if identity not in completed:
                pack = packs_by_rel[ep["pack_rel"]]
                view = views[ep["pack_rel"]]
                save_episode(
                    ep,
                    cropped[0],
                    query_input,
                    query_target,
                    pack,
                    view,
                    results_root,
                    elapsed,
                    model_assets_sha256,
                )
                completed.add(identity)
                touched.add(ep["pack_rel"])
        for rel in touched:
            pack = packs_by_rel[rel]
            view = views[rel]
            view["complete"] = len(view["episodes"]) == int(pack["n_episodes"])
            view["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            atomic_json(results_root / "views" / pack["slug"] / "results.json", view)
        remaining = total_real - len(completed)
        recent = batch_times[-20:]
        seconds_per_batch = statistics.median(recent)
        status = {
            "schema": SCHEMA,
            "plan_identity": plan["plan_identity"],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "running",
            "completed": len(completed),
            "completed_this_attempt": len(completed) - start_completed,
            "total": total_real,
            "remaining": remaining,
            "batch_index": batch_index,
            "batch_seconds": elapsed,
            "median_recent_batch_seconds": seconds_per_batch,
            "eta_seconds": (remaining / 2.0) * seconds_per_batch,
            "last_pack": batch[0]["pack_rel"],
            "last_keys": [ep["key"] for ep in real_batch],
            "device_memory": memory_info(xm, device),
            "backend": evidence,
            "model_assets_sha256s_sha256": model_assets_sha256,
        }
        atomic_json(results_root / "status.json", status)
        print(
            f"batch {batch_index + 1}/{len(batches)} {batch[0]['pack_rel']}: "
            f"{len(completed)}/{total_real} committed in {elapsed:.1f}s",
            flush=True,
        )

    for pack in plan["packs"]:
        view = views[pack["rel"]]
        if len(view["episodes"]) != int(pack["n_episodes"]):
            raise SystemExit(f"{pack['rel']}: incomplete after inference")
        view["complete"] = True
        view["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(results_root / "views" / pack["slug"] / "results.json", view)
    final = {
        "schema": SCHEMA,
        "plan_identity": plan["plan_identity"],
        "state": "inference_complete",
        "completed": total_real,
        "total": total_real,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": evidence,
        "model_assets_sha256s_sha256": model_assets_sha256,
    }
    atomic_json(results_root / "status.json", final)
    (results_root / "INFERENCE_COMPLETE_UTC").write_text(final["finished_utc"] + "\n")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    ip = sub.add_parser("infer")
    ip.add_argument("--plan", type=Path, required=True)
    ip.add_argument("--results-root", type=Path, required=True)
    ip.add_argument("--model-dir", type=Path, required=True)
    ip.add_argument("--model-sha256s", type=Path, required=True)
    ip.add_argument("--cache-dir", type=Path, required=True)
    ip.add_argument("--expected-tpu", required=True)
    ip.add_argument("--expected-zone", required=True)
    ip.add_argument("--expected-devices", type=int, default=8)
    ip.add_argument("--batch-size", type=int, default=2)
    ip.add_argument("--flash-block-size", type=int, default=1024)
    ip.set_defaults(func=infer)
    return ap


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
