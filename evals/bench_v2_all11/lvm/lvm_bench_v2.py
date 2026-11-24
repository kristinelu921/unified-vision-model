#!/usr/bin/env python3
"""Frozen released-LVM P=1 runner for verified Bench v2 validation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from functools import partial

import numpy as np
from PIL import Image

from lvm_runtime_config import (
    CONTEXT_FRAMES,
    EPISODE_CONTEXT_TOKENS,
    EPISODE_NEW_TOKENS,
    EPISODE_TOTAL_TOKENS,
    N_DEMOS,
    VISUAL_PAD_EOS_TOKEN_ID,
    VQ_CODEBOOK_SIZE,
    adaptation_record,
    make_vqlm_7b_config,
)


LIVE_REPO = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
PACK_URI_ROOT = "gs://kmh-gcp-us-west4/data/xtiange/mt_eval_packs"
TRUSTED_MANIFEST = LIVE_REPO / "docs/bench_v2_audit/manifest_v1.json"
TRUSTED_MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"
OFFICIAL_SOURCE_COMMIT = "b6de939ef0eb1ee6593445a7f5268145f338749b"
MODEL_ID = "Emma02/LVM_ckpts"
MODEL_REVISION = "aa5a9039273bd4221a0f69f5c943de1c56c39d89"
MODEL_SHA256 = "a5dfb7210a7a8a1a52492f0bf3b45010ec1da66dd5b2c5fd8245d7359f2ba0eb"
VQ_ID = "Emma02/vqvae_ckpts"
VQ_REVISION = "e4b5f5f83d9588ed52b6d1b65b91262f9f2d804b"
VQ_SHA256 = "fed960b9b88968dcaab28fd0cb28b786693b83b21fa93b1a1ad1177b3f7e9fd1"
READOUT = "LVM@native"
ARM = "LVM"
READOUT_TAG = "native"
SEED = 42
P = N_DEMOS
MODEL_SEED = 0
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50
SCHEMA = "lvm_7b_bench_v2_p1_native_generation_v1"
PLAN_SCHEMA = "lvm_7b_bench_v2_p1_plan_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def episode_sha256(demo_inputs, demo_targets, query_input, query_target) -> str:
    h = hashlib.sha256()
    for array in (*demo_inputs, *demo_targets, query_input, query_target):
        h.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return h.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)
    os.chmod(path, 0o666)


def atomic_png(path: Path, pixels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(np.asarray(pixels, np.uint8), "RGB").save(temp, format="PNG")
    os.replace(temp, path)
    os.chmod(path, 0o666)


def descriptor(path: Path, root: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.uint8)
        shape = [rgb.height, rgb.width, 3]
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "pixels_sha256": sha256_array(pixels),
        "shape": shape,
        "dtype": "uint8",
        "mode": "RGB",
    }


def task_rows(pack: dict, task: str) -> list[int]:
    return [index for index, row in enumerate(pack["records"]) if row["task"] == task]


def pack_slug(rel: str) -> str:
    return rel.replace("/", "__")


def source_hashes(official_repo: Path) -> dict[str, str]:
    sources = {
        "runner": Path(__file__).resolve(),
        "runtime_config": Path(__file__).with_name("lvm_runtime_config.py"),
        "runtime_v1": LIVE_REPO / "tools/bench_v2_runtime_v1.py",
        "runtime_v2": LIVE_REPO / "tools/bench_v2_runtime_v2.py",
        "episode_builder": LIVE_REPO / "jflux2/analysis/capacity_ladder_v9.py",
        "episode_base": LIVE_REPO / "jflux2/analysis/capacity_ladder_v6.py",
        "episode_contract": LIVE_REPO / "jflux2/analysis/t16_offline_eval_v1.py",
        "episode_numpy_replay": LIVE_REPO / "tools/bench_episode_v1.py",
        "conventions": LIVE_REPO / "jflux2/dataloaders/multi_task_loader_v4.py",
        "official_model": official_repo / "EasyLM/models/llama/llama_model.py",
        "official_generation": official_repo / "evaluation/vqlm_demo/inference.py",
        "official_preprocess": official_repo / "evaluation/vqlm_demo/utils.py",
        "official_vq": official_repo / "evaluation/vqlm_demo/vqvae_muse.py",
    }
    return {name: sha256_file(path) for name, path in sources.items()}


def prepare_plan(args: argparse.Namespace) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(LIVE_REPO))
    from tools.bench_v2_runtime_v2 import stage_pack  # noqa: PLC0415
    from jflux2.analysis import ttt_ladder_v1 as _ladder_side_effect  # noqa: F401,PLC0415
    from jflux2.analysis import capacity_ladder_v9 as CL9  # noqa: PLC0415

    if sha256_file(TRUSTED_MANIFEST) != TRUSTED_MANIFEST_SHA256:
        raise SystemExit("trusted bench_v2 manifest file changed")
    manifest = json.loads(TRUSTED_MANIFEST.read_text())
    all_rels = [row["rel"] for row in manifest["packs"]]
    rels = []
    if args.include_val:
        rels.append("bench_v2_val")
    if args.include_test:
        rels.extend(sorted(rel for rel in all_rels if rel.startswith("bench_v2_test/")))
    if not rels:
        raise SystemExit("no packs selected")

    plan = {
        "schema": PLAN_SCHEMA,
        "protocol": SCHEMA,
        "trusted_manifest": str(TRUSTED_MANIFEST),
        "trusted_manifest_file_sha256": TRUSTED_MANIFEST_SHA256,
        "pack_uri_root": PACK_URI_ROOT,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "vq_id": VQ_ID,
        "vq_revision": VQ_REVISION,
        "vq_sha256": VQ_SHA256,
        "seed": SEED,
        "p": P,
        "model_seed": MODEL_SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "candidates": 1,
        "readout": READOUT,
        "context_order": "x1,y1,query",
        "preprocessing": "RGB; direct resize to 256x256 only if needed; float32 /255; no crop",
        "source_sha256": source_hashes(args.official_repo),
        "packs": [],
        "episodes": [],
    }

    for rel in rels:
        pack = stage_pack(
            rel,
            uri_root=PACK_URI_ROOT,
            manifest_path=TRUSTED_MANIFEST,
            manifest_sha256=TRUSTED_MANIFEST_SHA256,
            local_root=args.pack_cache,
        )
        tasks = []
        for row in pack["records"]:
            if row["task"] not in tasks:
                tasks.append(row["task"])
        pack_start = len(plan["episodes"])
        pack_keys = []
        for task in tasks:
            indices = task_rows(pack, task)
            images = pack["images"][indices]
            masks = pack["masks"][indices]
            records = [pack["records"][index] for index in indices]
            sources = [str(row.get("source")) for row in records]
            for q in range(len(indices)):
                built = CL9.build_episode(images, masks, sources, task, q, SEED, P)
                if int(built["p0"]) != P:
                    raise SystemExit(f"{rel} {task} q{q}: only {built['p0']} same-source demonstrations")
                cid = int(built["cid"])
                demo_local = [int(index) for index in built["demo_indices"]]
                demo_global = [int(indices[index]) for index in demo_local]
                demo_inputs = [np.asarray(images[index], np.uint8) for index in demo_local]
                demo_targets = [
                    np.rint(np.transpose(pair[1], (1, 2, 0)) * 255).astype(np.uint8)
                    for pair in built["real"]
                ]
                query_input = np.asarray(images[q], np.uint8)
                query_target = np.asarray(built["query_t"], np.uint8)
                key = f"{task}:{q}"
                pack_keys.append(key)
                plan["episodes"].append(
                    {
                        "episode_id": f"{pack_slug(rel)}__{task}__q{q:05d}__P1__real",
                        "key": key,
                        "pack_rel": rel,
                        "pack_slug": pack_slug(rel),
                        "task": task,
                        "q": int(q),
                        "p": P,
                        "cid": cid,
                        "query_global": int(indices[q]),
                        "query_record": str(records[q]["record_id"]),
                        "demo_global": demo_global,
                        "demo_records": [str(records[index]["record_id"]) for index in demo_local],
                        "episode_pixels_sha256": episode_sha256(
                            demo_inputs, demo_targets, query_input, query_target
                        ),
                        "query_input_pixels_sha256": sha256_array(query_input),
                        "query_target_pixels_sha256": sha256_array(query_target),
                    }
                )
        count = len(plan["episodes"]) - pack_start
        plan["packs"].append(
            {
                "rel": rel,
                "slug": pack_slug(rel),
                "local_dir": pack["staged_copy"]["local_dir"],
                "data_identity": pack["data_identity"],
                "data_identity_sha256": pack["data_identity_sha256"],
                "tasks": tasks,
                "expected_episode_keys": pack_keys,
                "n_episodes": count,
                "plan_range": [pack_start, len(plan["episodes"])],
            }
        )
        print(f"[plan] {rel}: {count} queries", flush=True)
        del pack, images, masks
        gc.collect()

    plan["plan_identity"] = hashlib.sha256(canonical_bytes(plan)).hexdigest()
    atomic_json(args.out, plan)
    print(
        f"[plan] wrote {args.out}: {len(plan['packs'])} packs, {len(plan['episodes'])} queries, "
        f"identity={plan['plan_identity']}",
        flush=True,
    )
    return 0


class PackCache:
    def __init__(self, plan: dict):
        self.by_rel = {row["rel"]: row for row in plan["packs"]}
        self.rel = None
        self.images = None
        self.masks = None

    def load(self, rel: str) -> tuple[np.ndarray, np.ndarray]:
        if rel == self.rel:
            return self.images, self.masks
        self.images = self.masks = None
        gc.collect()
        info = self.by_rel[rel]
        with np.load(Path(info["local_dir"]) / "samples.npz", allow_pickle=False) as archive:
            self.images = np.array(archive["images"], copy=True)
            self.masks = np.array(archive["masks"], copy=True)
        self.images.setflags(write=False)
        self.masks.setflags(write=False)
        self.rel = rel
        print(f"[infer] loaded {rel}: {len(self.images)} rows", flush=True)
        return self.images, self.masks


def build_context(ep: dict, packs: PackCache):
    sys.path.insert(0, str(LIVE_REPO))
    from tools import bench_episode_v1 as BE  # noqa: PLC0415

    images, masks = packs.load(ep["pack_rel"])
    demo_inputs = [np.asarray(images[index], np.uint8) for index in ep["demo_global"]]
    raw_targets = [np.asarray(masks[index], np.uint8) for index in ep["demo_global"]]
    raw_targets.append(np.asarray(masks[ep["query_global"]], np.uint8))
    coded = BE.code_episode(raw_targets, ep["task"], int(ep["cid"]))
    demo_targets, query_target = coded[:-1], coded[-1]
    query_input = np.asarray(images[ep["query_global"]], np.uint8)
    if episode_sha256(demo_inputs, demo_targets, query_input, query_target) != ep["episode_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: reconstructed episode pixels changed")
    if sha256_array(query_input) != ep["query_input_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: query input changed")
    if sha256_array(query_target) != ep["query_target_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: query target changed")
    frames = []
    for demo_input, demo_target in zip(demo_inputs, demo_targets):
        frames.extend((demo_input, demo_target))
    frames.append(query_input)
    context = np.stack(frames).astype(np.float32) / np.float32(255.0)
    if context.shape != (CONTEXT_FRAMES, 256, 256, 3):
        raise SystemExit(f"{ep['episode_id']}: context shape {context.shape} differs")
    return context, query_input, query_target


def initial_view(pack: dict, plan: dict) -> dict:
    return {
        "schema": SCHEMA,
        "protocol": SCHEMA,
        "args": {
            "pack": pack["local_dir"],
            "p": P,
            "seed": SEED,
            "condition": "real",
            "model_seed": MODEL_SEED,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "candidates": 1,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "vq_id": VQ_ID,
            "vq_revision": VQ_REVISION,
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
    views = {}
    completed = set()
    for pack in plan["packs"]:
        root = results_root / "views" / pack["slug"]
        path = root / "results.json"
        view = json.loads(path.read_text()) if path.exists() else initial_view(pack, plan)
        if view.get("plan_identity") != plan["plan_identity"]:
            raise SystemExit(f"{path}: plan identity differs")
        if view.get("data_identity") != pack["data_identity"]:
            raise SystemExit(f"{path}: pack identity differs")
        for key, episode in list(view["episodes"].items()):
            artifacts = episode.get("artifacts", {})
            entries = [artifacts.get("query_input"), artifacts.get("query_target")]
            entries.append(((artifacts.get("readouts", {}).get(READOUT) or {}).get("real")))
            if any(not item or not (root / item["path"]).is_file() for item in entries):
                del view["episodes"][key]
                view["complete"] = False
                continue
            for item in entries:
                if sha256_file(root / item["path"]) != item["sha256"]:
                    raise SystemExit(f"{root / item['path']}: saved artifact hash mismatch")
            completed.add(f"{pack['rel']}|{key}")
        atomic_json(path, view)
        views[pack["rel"]] = view
    return views, completed


def save_episode(ep, prediction, query_input, query_target, pack, view, results_root, timings):
    root = results_root / "views" / pack["slug"]
    episode_dir = root / f"{ep['task']}_q{ep['q']}"
    input_path = episode_dir / "query_input.png"
    target_path = episode_dir / "query_target.png"
    prediction_path = episode_dir / f"{ARM}_real_{READOUT_TAG}_query.png"
    atomic_png(input_path, query_input)
    atomic_png(target_path, query_target)
    atomic_png(prediction_path, prediction)
    view["episodes"][ep["key"]] = {
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
                    "vq_revision": VQ_REVISION,
                    "model_seed": MODEL_SEED,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "candidates": 1,
                    "native_prompt_count": P,
                    "context_frames": CONTEXT_FRAMES,
                    "generated_tokens": EPISODE_NEW_TOKENS,
                    "ensemble": False,
                    **timings,
                }
            }
        },
        "artifacts": {
            "query_input": descriptor(input_path, root),
            "query_target": descriptor(target_path, root),
            "readouts": {READOUT: {"real": descriptor(prediction_path, root)}},
        },
    }


def infer(args: argparse.Namespace) -> int:
    plan = json.loads(args.plan.read_text())
    claimed = plan.pop("plan_identity")
    if hashlib.sha256(canonical_bytes(plan)).hexdigest() != claimed:
        raise SystemExit("plan identity differs")
    plan["plan_identity"] = claimed
    if plan.get("schema") != PLAN_SCHEMA or plan.get("p") != P or plan.get("readout") != READOUT:
        raise SystemExit("refusing a plan outside the frozen LVM P=1 contract")
    if sha256_file(args.checkpoint) != MODEL_SHA256:
        raise SystemExit("converted LVM checkpoint SHA256 differs")
    if sha256_file(args.vq_checkpoint / "pytorch_model.bin") != VQ_SHA256:
        raise SystemExit("VQ checkpoint SHA256 differs")

    sys.path.insert(0, str(args.official_repo))
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from flax.core import freeze  # noqa: F401,PLC0415
    from jax.experimental.pjit import pjit  # noqa: PLC0415
    from jax.sharding import PartitionSpec as PS  # noqa: PLC0415
    from transformers import FlaxLogitsProcessorList, GenerationConfig  # noqa: PLC0415
    from EasyLM.checkpoint import StreamingCheckpointer  # noqa: PLC0415
    from EasyLM.jax_utils import (  # noqa: PLC0415
        get_float_dtype_by_name,
        make_shard_and_gather_fns,
        match_partition_rules,
        tree_apply,
    )
    from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM  # noqa: PLC0415
    from evaluation.vqlm_demo.vqvae_muse import VQGANModel  # noqa: PLC0415

    devices = jax.devices()
    evidence = {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [str(device) for device in devices],
        "jax_version": jax.__version__,
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "hostname": socket.gethostname(),
        "expected_tpu": args.expected_tpu,
        "expected_zone": args.expected_zone,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    if evidence["backend"] != "tpu" or len(devices) != args.expected_devices:
        raise SystemExit(f"refusing nonconforming TPU backend: {evidence}")
    if any("TPU" not in str(device).upper() for device in devices):
        raise SystemExit(f"refusing non-TPU devices: {evidence}")
    args.results_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_root / "backend.json", evidence)
    atomic_json(args.results_root / "runtime_config.json", adaptation_record())

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(4, args.torch_threads))
    vq = VQGANModel.from_pretrained(str(args.vq_checkpoint)).to("cpu").eval()

    config = make_vqlm_7b_config(LLaMAConfig)
    with jax.default_device(jax.devices("cpu")[0]):
        _, host_params = StreamingCheckpointer.load_trainstate_checkpoint(
            f"params::{args.checkpoint}", disallow_trainstate=True
        )
        model = FlaxLLaMAForCausalLM(
            config,
            input_shape=(1, EPISODE_TOTAL_TOKENS),
            seed=MODEL_SEED,
            dtype=jnp.float16,
            _do_init=False,
        )
    model_ps = match_partition_rules(LLaMAConfig.get_partition_rules(), host_params)
    shard_fns, _ = make_shard_and_gather_fns(model_ps, get_float_dtype_by_name("fp16"))
    mesh = LLaMAConfig.get_jax_mesh(f"1,{args.expected_devices},1")
    shard_started = time.monotonic()
    with mesh:
        params = tree_apply(shard_fns, host_params)
    del host_params
    print(f"PARAMS_SHARDED seconds={time.monotonic() - shard_started:.2f}", flush=True)

    class SuppressNonVQTokens:
        def __call__(self, input_ids, scores, cur_len):
            del input_ids, cur_len
            return scores.at[:, VQ_CODEBOOK_SIZE:].set(jnp.finfo(scores.dtype).min)

    generation_config = GenerationConfig(
        max_new_tokens=EPISODE_NEW_TOKENS,
        pad_token_id=VISUAL_PAD_EOS_TOKEN_ID,
        bos_token_id=1,
        eos_token_id=VISUAL_PAD_EOS_TOKEN_ID,
        do_sample=True,
        num_beams=1,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )
    processors = FlaxLogitsProcessorList([SuppressNonVQTokens()])

    @partial(pjit, in_shardings=(model_ps, PS(), PS()), out_shardings=PS())
    def generate(params_tree, input_tokens, key):
        sequences = model.generate(
            input_tokens,
            attention_mask=jnp.ones_like(input_tokens),
            params=params_tree["params"],
            prng_key=key,
            logits_processor=processors,
            generation_config=generation_config,
        ).sequences
        return sequences[:, input_tokens.shape[1]:]

    views, completed = checked_views(args.results_root, plan)
    packs = PackCache(plan)
    packs_by_rel = {row["rel"]: row for row in plan["packs"]}
    pending = [ep for ep in plan["episodes"] if f"{ep['pack_rel']}|{ep['key']}" not in completed]
    print(f"RUN total={len(plan['episodes'])} complete={len(completed)} pending={len(pending)}", flush=True)
    for ordinal, episode in enumerate(pending, 1):
        started = time.monotonic()
        context, query_input, query_target = build_context(episode, packs)
        encode_started = time.monotonic()
        with torch.no_grad():
            tensor = torch.from_numpy(context).permute(0, 3, 1, 2).contiguous()
            _, token_ids = vq.encode(tensor)
        input_ids = np.asarray(token_ids.reshape(1, -1).cpu(), dtype=np.int32)
        if input_ids.shape != (1, EPISODE_CONTEXT_TOKENS):
            raise SystemExit(f"{episode['episode_id']}: token shape {input_ids.shape} differs")
        encode_seconds = time.monotonic() - encode_started

        generation_started = time.monotonic()
        with mesh:
            generated = generate(params, input_ids, jax.random.PRNGKey(MODEL_SEED))
            generated = np.asarray(jax.device_get(generated), dtype=np.int64)
        generation_seconds = time.monotonic() - generation_started
        if generated.shape != (1, EPISODE_NEW_TOKENS):
            raise SystemExit(f"{episode['episode_id']}: generated shape {generated.shape} differs")
        if generated.min() < 0 or generated.max() >= VQ_CODEBOOK_SIZE:
            raise SystemExit(f"{episode['episode_id']}: nonvisual output token")

        decode_started = time.monotonic()
        with torch.no_grad():
            decoded = vq.decode_code(torch.from_numpy(generated).long())
        # Match the released demo/batch-generation code: clip, multiply, cast.
        prediction = (
            np.clip(decoded[0].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        decode_seconds = time.monotonic() - decode_started
        total_seconds = time.monotonic() - started
        pack = packs_by_rel[episode["pack_rel"]]
        view = views[episode["pack_rel"]]
        save_episode(
            episode,
            prediction,
            query_input,
            query_target,
            pack,
            view,
            args.results_root,
            {
                "encode_seconds": encode_seconds,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "total_seconds": total_seconds,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        view["complete"] = len(view["episodes"]) == int(pack["n_episodes"])
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
        if not (args.results_root / "SMOKE_PASSED.json").exists():
            atomic_json(
                args.results_root / "SMOKE_PASSED.json",
                {"episode_id": episode["episode_id"], "prediction_sha256": sha256_array(prediction), **evidence},
            )
            print(f"SMOKE_PASS episode={episode['episode_id']}", flush=True)
        print(
            f"[{ordinal}/{len(pending)}] {episode['episode_id']} total={total_seconds:.2f}s "
            f"encode={encode_seconds:.2f}s generate={generation_seconds:.2f}s decode={decode_seconds:.2f}s",
            flush=True,
        )

    for pack in plan["packs"]:
        view = views[pack["rel"]]
        if set(view["episodes"]) != set(pack["expected_episode_keys"]):
            raise SystemExit(f"{pack['rel']}: terminal episode coverage differs")
        view["complete"] = True
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
    atomic_json(
        args.results_root / "INFERENCE_COMPLETE.json",
        {
            "schema": "lvm_bench_v2_p1_inference_complete_v1",
            "episodes": len(plan["episodes"]),
            "packs": len(plan["packs"]),
            "plan_identity": plan["plan_identity"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("prepare-plan")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--pack-cache", type=Path, required=True)
    plan.add_argument("--official-repo", type=Path, required=True)
    plan.add_argument("--include-val", type=int, choices=(0, 1), default=1)
    plan.add_argument("--include-test", type=int, choices=(0, 1), default=1)
    run = subparsers.add_parser("infer")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--results-root", type=Path, required=True)
    run.add_argument("--official-repo", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--vq-checkpoint", type=Path, required=True)
    run.add_argument("--expected-tpu", required=True)
    run.add_argument("--expected-zone", required=True)
    run.add_argument("--expected-devices", type=int, default=8)
    run.add_argument("--torch-threads", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare-plan":
        return prepare_plan(args)
    if args.command == "infer":
        return infer(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
