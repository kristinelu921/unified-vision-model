#!/usr/bin/env python3
"""Frozen InstaManip-17B-3shot native-P=3 producer for Bench v2."""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image

JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
MANIFEST = JFLUX / "docs/bench_v2_audit/manifest_v1.json"
MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"
PACK_URI_ROOT = "gs://kmh-gcp-us-west4/data/xtiange/mt_eval_packs"
SOURCE_COMMIT = "be992025e226211352cbb6c0734f5243fe1972c4"
MODEL_REVISION = "c63e249238beba8cc09342782a63322f0833d282"
SEED_X_REVISION = "534d2ff6669f36e4241a3d3fbbfa2d40adc04cdf"
QWEN_REVISION = "f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"
SOURCE_HASHES = {
    "official_run": "eac813df81e50e73321e7cd2d9d0226b9556c500be1f849de424ba5a0fb81d68",
    "official_eval": "f63b7201ddd8ff58b25d45d4e0a5d7514df398e129826bf99153ecd098a1613f",
    "agent": "e817dd052adce47e3b7fbb935815e7edd917fb2985d624848dc6294c52e52c85",
    "llama_xla_diagnostic_patch": "28b14376e0e1dc9487fb7ae8ee4d77c752ba56be6d1746c01fc1a55d4fdb0781",
    "custom_diffusion_pipeline": "b7efc22c4feaae1ef153f6664b49c9141d322b26701a2a2af89734ff333a425c",
    "adapter_modules": "6ef598a373f4d7bcafae21b43f8ab678b6c873f618894d5c2398cccb2f71e926",
    "transforms": "0bc05062cf0bbf57b4085de35b248761aaa849f20120453607e86b91cc67cf73",
    "added_tokens": "db3fb6023cc94f2a858b12da53856a16be467d87d2c1c0d6769f0cc88f6dd98f",
    "runtime_v2": "537368acfe898edf4e8213b802ad9a2c496cae12865818ce5dc846865254b132",
    "episode_builder": "fa0b36a4dfa95073cfe4d4a1ce23a5d0e177c8667cc6d276b9df2c6ad1317373",
    "episode_replay": "efc614de568df9eaa907f5d946a257a8f1f7ef915d47e604d10f671b13e8d81b",
    "conventions": "d4f9bce12ff179c0f85c86b312ebf2d85e2bc55a25bf5095956705a263e675f9",
    "producer_validator": "53a5270a1f4a1c38c7d1ac02e2ff59239ab6c5e7f27d44f9880395ee51696862",
    "corrected_backend": "fcfabc83dc0f7d0020cb7b92526318832d28714956c444218c2e7d49035e0844",
}
# The accepted q6 plan predates the separately audited diffusion-pipeline
# provenance field. Verify that file above, but do not rewrite the frozen plan
# identity by inserting a new key into its serialized source map.
PLAN_SOURCE_HASHES = {
    name: digest for name, digest in SOURCE_HASHES.items()
    if name != "custom_diffusion_pipeline"
}
P = 3
SEED = 42
TASK_INSTRUCTIONS = {
    "colorization": "Colorize the grayscale image with realistic colors while preserving its exact content, structure, and lighting.",
    "deblurring": "Remove the blur and restore a sharp image without changing the scene, objects, colors, or composition.",
    "denoising": "Remove the image noise while preserving all real details, colors, edges, and scene content.",
    "edge_detection": "Convert the image into an edge map containing only white object and region boundaries on a black background.",
    "face_landmarks": "Locate the facial landmarks and output only the colored landmark points on a black background, following the exact point and color convention shown in the examples.",
    "human_pose_2d": "Locate the person’s 2D body joints and output only the colored joint markers on a black background, following the exact joint and color convention shown in the examples.",
    "inpainting": "Fill in the black missing regions with content consistent with the surrounding image, while leaving every unmasked pixel unchanged.",
    "instance_segmentation": "Segment every individual object instance and output only a flat color mask, using one solid color per instance and the exact color convention shown in the examples.",
    "relative_depth": "Predict a dense relative-depth map for the entire image, following the exact grayscale depth direction and intensity convention shown in the examples.",
    "saliency_detection": "Identify the most visually salient foreground object and output a binary mask with the salient region in white and the background in black.",
    "semantic_segmentation": "Assign every pixel its semantic class and output only a flat color segmentation mask, using the exact class-color convention shown in the examples with no photographic texture or shading.",
    "super_resolution": "Restore a high-resolution version of the image with faithful fine detail while preserving the exact objects, geometry, identity, colors, and composition.",
    "surface_normals": "Predict a dense per-pixel surface-normal map encoded as RGB colors, following the exact coordinate and color convention shown in the examples without retaining photographic texture.",
}
TASK_INSTRUCTIONS_IDENTITY = "frozen_user_supplied_task_instruction_map_v1"
PROMPT_POLICY = "user-supplied deterministic task instruction selected by Bench-v2 task label"
NUM_IMAGE_TOKENS = 64
NUM_EDIT_TOKENS = 30
MAX_NEW_TOKENS = 450 + 100 * P
DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_GUIDANCE_SCALE = 1.5
READOUT = "InstaManip@native_p3_text_task_v1"
READOUT_TAG = "native_p3_text_task_v1"
EVALUATION_CONTRACT = "instamanip_official_p3_text_task_v1_evaluation_cli_v2_v5e32_spmd_fused_image_block_v1_batched_diffusion_v1_vae_dtype_adapter_dynamic_euler_device_scalars_v1"
PORT_QUALIFICATION_EVALUATION_CONTRACT = EVALUATION_CONTRACT
PLAN_SCHEMA = "instamanip_bench_v2_native_p3_text_task_v1_v5e32_plan_v1"
VIEW_SCHEMA = "instamanip_bench_v2_native_p3_text_task_v1_v5e32_generation_v1"
QUALIFICATION_SCHEMA = "instamanip_bench_v2_v5e32_actual_p_repeat_qualification_v1"
QUALIFICATION_PROJECTION_MODE = "cold_first_batch_plus_12_warm_validation_task_batches_v1"
SMOKE_SCHEMA = "instamanip_bench_v2_v5e32_actual_p_smoke_v1"
XLA_GENERATION_CACHE = "instamanip_xla_fused_image_block_v1"
XLA_DIFFUSION_EXECUTION = "instamanip_xla_dynamic_euler_device_scalars_v1"
XLA_COMPILATION_CACHE_LIMIT = 4
VAE_INPUT_ADAPTER = "instamanip_vae_input_to_model_dtype_v1"
VALIDATION_GATE_SCHEMA = "instamanip_bench_v2_native_p3_validation_gate_v1"
CORRECTED_PROTOCOL = "bench_v2_PP2_CSA_seed1_AP_FIRST_v1"
CORRECTED_RELEASE = "bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792"
EXPECTED_ACCELERATOR = "v5litepod-32"
EXPECTED_TOPOLOGY = "4x8"
EXPECTED_GLOBAL_DEVICES = 32
EXPECTED_ADDRESSABLE_DEVICES = 4
EXPECTED_WORKERS = 8
INFERENCE_BATCH_SIZE = EXPECTED_GLOBAL_DEVICES
SMOKE_QUERY_COUNT = 17
VALIDATION_TASK_GROUPS = 13
TOTAL_VALIDATION_QUERIES = 221
TOTAL_BENCH_QUERIES = 17732
MAX_TOTAL_SECONDS = 5 * 24 * 60 * 60
MAX_PROCESS_RSS_BYTES = 150_000_000_000
MIN_HOST_MEM_AVAILABLE_BYTES = 20_000_000_000
Q6_CANONICAL_PACK_CACHE = Path(
    "/dev/shm/instamanip_bench_v2_native_p2_v5e32_q6_3f0211_20261016/packs")
FROZEN_TEST_PLAN_SHA256 = "8a2b7a833e9b80a0088408f77751f0e4c71cd163ced4faf904c04fc303737176"
FROZEN_TEST_PLAN_IDENTITY = "fdd3e62754a70fd8f54c96d219211fc238680b917ed898ccb12fa8e655e4e331"
FROZEN_TEST_PREPARATION_PACK_ROOT = Path(
    "/tmp/kristine_bench_all11_scoring_20261110/packs/89f2501aa722cd12")
FROZEN_TEST_BUILDER_SHA256 = "fa9f5fddd0ceca9e0232bb0307fee1cd986df6ccabc39f8ad0801564ee35044d"
FROZEN_VALIDATION_PLAN_IDENTITY = "b2494def1af0155f13787076362f6070d428713abf530a593e52ba0cc68d9feb"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


TASK_INSTRUCTIONS_SHA256 = identity(TASK_INSTRUCTIONS)


def instruction_for_task(task):
    try:
        return TASK_INSTRUCTIONS[task]
    except KeyError as exc:
        raise ValueError(f"no frozen text-conditioned instruction for task {task!r}") from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_png(path, pixels):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(np.asarray(pixels, np.uint8), "RGB").save(temporary, format="PNG", optimize=False)
    os.replace(temporary, path)


def descriptor(path, root, size):
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (size, size):
            raise ValueError(f"invalid InstaManip artifact: {path}")
        pixels = np.asarray(image, np.uint8)
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path),
            "pixels_sha256": sha256_array(pixels), "shape": [size, size, 3],
            "dtype": "uint8", "mode": "RGB"}


def verify_descriptor(root, item, where, size=256):
    expected = [size, size, 3]
    if not isinstance(item, dict) or item.get("shape") != expected or item.get("dtype") != "uint8" or item.get("mode") != "RGB":
        raise ValueError(f"{where}: descriptor contract differs")
    rel = Path(item.get("path", "")); path = root / rel
    if rel.is_absolute() or ".." in rel.parts or not path.is_file() or descriptor(path, root, size) != item:
        raise ValueError(f"{where}: artifact bytes/pixels differ")


def source_paths(source):
    return {
        "official_run": source / "src/inference/run_model_multishot.py",
        "official_eval": source / "src/inference/eval_model_multishot.py",
        "agent": source / "src/models/mllm/instamanip.py",
        "llama_xla_diagnostic_patch": source / "src/models/mllm/modeling_llama_xformer.py",
        "custom_diffusion_pipeline": source / "src/models/detokenizer/pipeline_stable_diffusion_xl_t2i_edit.py",
        "adapter_modules": source / "src/models/detokenizer/adapter_modules.py",
        "transforms": source / "src/processer/transforms.py",
        "added_tokens": source / "pretrained/added_tokens.json",
        "runtime_v2": JFLUX / "tools/bench_v2_runtime_v2.py",
        "episode_builder": JFLUX / "jflux2/analysis/capacity_ladder_v9.py",
        "episode_replay": JFLUX / "tools/bench_episode_v1.py",
        "conventions": JFLUX / "jflux2/dataloaders/multi_task_loader_v4.py",
        "producer_validator": JFLUX / "tools/score_bench_outputs_v1.py",
        "corrected_backend": JFLUX / "tools/bench_eval_v3_backend_v1.py",
    }


def verify_sources(source):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True).splitlines()
    got = {name: sha256_file(path) for name, path in source_paths(source).items()}
    expected_status = [
        " M src/models/detokenizer/pipeline_stable_diffusion_xl_t2i_edit.py",
        " M src/models/mllm/modeling_llama_xformer.py",
    ]
    if commit != SOURCE_COMMIT or status != expected_status or got != SOURCE_HASHES:
        raise SystemExit("official/shared source identity differs from preregistration")
    return got


def pack_slug(rel):
    return rel.replace("/", "__")


def phase_rels(manifest, phase):
    rels = ["bench_v2_val"] if phase == "validation" else sorted(
        row["rel"] for row in manifest["packs"] if row["rel"].startswith("bench_v2_test/"))
    by_rel = {row["rel"]: int(row["n"]) for row in manifest["packs"]}
    expected = (1, 221) if phase == "validation" else (26, 17_511)
    if len(rels) != expected[0] or sum(by_rel[rel] for rel in rels) != expected[1]:
        raise SystemExit("Bench v2 phase cardinality differs")
    return rels


def replay_episode(images, masks, records, task, q, demo, cid):
    sys.path.insert(0, str(JFLUX))
    from tools import bench_episode_v1 as BE
    sources = [str(row.get("source")) for row in records]
    candidates = [i for i, source in enumerate(sources) if i != q and source == sources[q]]
    demo = [int(i) for i in demo]
    if len(demo) != P or len(set(demo)) != P or any(i not in candidates for i in demo):
        raise ValueError(f"{task}:{q}: invalid same-source native prompt")
    coded = BE.code_episode([masks[i] for i in demo] + [masks[q]], task, int(cid))
    values = {"demo_inputs": np.asarray([images[i] for i in demo], np.uint8),
              "demo_targets": np.asarray(coded[:-1], np.uint8),
              "query_input": np.asarray(images[q], np.uint8),
              "query_target": np.asarray(coded[-1], np.uint8)}
    digest = hashlib.sha256()
    for name in ("demo_inputs", "demo_targets", "query_input", "query_target"):
        digest.update(memoryview(np.ascontiguousarray(values[name])).cast("B"))
    return {**values, "candidates": candidates, "demo_local": demo, "cid": int(cid),
            "episode_pixels_sha256": digest.hexdigest()}


def build_episode(images, masks, records, task, q):
    sys.path.insert(0, str(JFLUX))
    from jflux2.analysis import ttt_ladder_v1 as _side_effect  # noqa: F401
    from jflux2.analysis import capacity_ladder_v9 as CL9
    built = CL9.build_episode(images, masks, [str(r.get("source")) for r in records], task, q, SEED, P)
    if int(built["p0"]) != P:
        raise ValueError(f"{task}:{q}: no native same-source P=3 prompt")
    replay = replay_episode(images, masks, records, task, q, built["demo_indices"], built["cid"])
    if not np.array_equal(replay["query_target"], built["query_t"]):
        raise ValueError(f"{task}:{q}: authoritative episode replay differs")
    return replay


def prepare_plan(args):
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("trusted Bench v2 manifest differs")
    hashes = verify_sources(args.source_dir)
    sys.path.insert(0, str(JFLUX))
    from tools.bench_v2_runtime_v2 import stage_pack
    manifest = json.loads(MANIFEST.read_text())
    plan = {"schema": PLAN_SCHEMA, "phase": args.phase, "manifest_sha256": MANIFEST_SHA256,
            "pack_uri_root": PACK_URI_ROOT, "source_commit": SOURCE_COMMIT,
            "source_sha256": {name: hashes[name] for name in PLAN_SOURCE_HASHES},
            "asset_manifest_sha256": args.asset_manifest_sha256,
            "p": P, "seed": SEED, "condition": "real", "readout": READOUT,
            "evaluation_contract": EVALUATION_CONTRACT,
            "task_instructions": TASK_INSTRUCTIONS,
            "task_instructions_sha256": TASK_INSTRUCTIONS_SHA256,
            "task_instructions_identity": TASK_INSTRUCTIONS_IDENTITY,
            "prompt_policy": PROMPT_POLICY,
            "max_new_tokens": MAX_NEW_TOKENS, "diffusion_steps": DIFFUSION_STEPS,
            "guidance_scale": GUIDANCE_SCALE, "image_guidance_scale": IMAGE_GUIDANCE_SCALE,
            "exemplar_preprocessing": "RGB 256 -> Pillow implicit resize 1024 -> official CLIP transform 448",
            "query_preprocessing": "RGB 256 -> official CLIP transform 448",
            "latent_dtype_contract": "CPU float32 deterministic noise is cast to pipeline BF16; the upstream forced-FP32 VAE image tensor is recast to the pinned BF16 VAE dtype at encode",
            "vae_input_adapter": VAE_INPUT_ADAPTER,
            "backend_contract": "PyTorch/XLA v5litepod-32 eight-worker SPMD-FSDP; no CUDA parity claim",
            "packs": [], "episodes": []}
    for rel in phase_rels(manifest, args.phase):
        pack = stage_pack(rel, uri_root=PACK_URI_ROOT, manifest_path=MANIFEST,
                          manifest_sha256=MANIFEST_SHA256, local_root=args.pack_cache)
        tasks = list(dict.fromkeys(row["task"] for row in pack["records"]))
        start, keys = len(plan["episodes"]), []
        for task in tasks:
            rows = [i for i, row in enumerate(pack["records"]) if row["task"] == task]
            images, masks = pack["images"][rows], pack["masks"][rows]
            records = [pack["records"][i] for i in rows]
            for q in range(len(rows)):
                built = build_episode(images, masks, records, task, q)
                key = f"{task}:{q}"; keys.append(key)
                plan["episodes"].append({"episode_id": f"{pack_slug(rel)}__{task}__q{q:05d}__P3__real",
                    "key": key, "pack_rel": rel, "pack_slug": pack_slug(rel), "task": task, "q": q,
                    "p": P, "cid": built["cid"], "query_global": rows[q],
                    "query_record": str(records[q]["record_id"]), "query_source": str(records[q].get("source")),
                    "demo_global": [rows[i] for i in built["demo_local"]],
                    "demo_records": [str(records[i]["record_id"]) for i in built["demo_local"]],
                    "candidates_global": [rows[i] for i in built["candidates"]],
                    "episode_pixels_sha256": built["episode_pixels_sha256"]})
        staged_dir = Path(pack["staged_copy"]["local_dir"])
        try:
            staged_rel = staged_dir.resolve().relative_to(args.pack_cache.resolve())
        except ValueError as exc:
            raise SystemExit("staged pack escaped the runtime pack cache") from exc
        logical_dir = args.plan_pack_cache / staged_rel
        plan["packs"].append({"rel": rel, "slug": pack_slug(rel),
            "local_dir": str(logical_dir), "data_identity": pack["data_identity"],
            "data_identity_sha256": pack["data_identity_sha256"], "tasks": tasks,
            "expected_episode_keys": keys, "n_episodes": len(plan["episodes"]) - start,
            "plan_range": [start, len(plan["episodes"])]})
        del pack; gc.collect()
    expected = 221 if args.phase == "validation" else 17_511
    if len(plan["episodes"]) != expected:
        raise SystemExit("planned query count differs")
    plan["plan_identity"] = identity(plan); atomic_json(args.out, plan)


def verify_plan(path, phase):
    plan = json.loads(Path(path).read_text()); claimed = plan.pop("plan_identity", None)
    if claimed != identity(plan):
        raise SystemExit("plan identity differs")
    plan["plan_identity"] = claimed; expected = 221 if phase == "validation" else 17_511
    if (plan.get("schema"), plan.get("phase"), plan.get("p"), plan.get("condition"),
            plan.get("readout"), plan.get("evaluation_contract"), len(plan.get("episodes", []))) != (
            PLAN_SCHEMA, phase, P, "real", READOUT, EVALUATION_CONTRACT, expected):
        raise SystemExit("plan is outside the InstaManip native-P=3 phase contract")
    if (plan.get("max_new_tokens") != MAX_NEW_TOKENS
            or plan.get("task_instructions") != TASK_INSTRUCTIONS
            or plan.get("task_instructions_sha256") != TASK_INSTRUCTIONS_SHA256
            or plan.get("task_instructions_identity") != TASK_INSTRUCTIONS_IDENTITY
            or plan.get("prompt_policy") != PROMPT_POLICY
            or plan.get("exemplar_preprocessing") != "RGB 256 -> Pillow implicit resize 1024 -> official CLIP transform 448"
            or plan.get("latent_dtype_contract") != "CPU float32 deterministic noise is cast to pipeline BF16; the upstream forced-FP32 VAE image tensor is recast to the pinned BF16 VAE dtype at encode"
            or plan.get("vae_input_adapter") != VAE_INPUT_ADAPTER):
        raise SystemExit("plan evaluation preprocessing/token/dtype contract differs")
    if plan.get("source_sha256") != PLAN_SOURCE_HASHES:
        raise SystemExit("plan source hashes differ")
    return plan


class PackCache:
    def __init__(self, plan, runtime_pack_cache=None):
        self.info = {row["rel"]: row for row in plan["packs"]}; self.rel = None; self.value = None
        override = runtime_pack_cache or os.environ.get("INSTAMANIP_RUNTIME_PACK_CACHE")
        self.runtime_pack_cache = Path(override).resolve() if override else None

    def root_for(self, rel):
        logical = Path(self.info[rel]["local_dir"])
        if self.runtime_pack_cache is None:
            return logical
        suffix = None
        for accepted_root in (Q6_CANONICAL_PACK_CACHE, FROZEN_TEST_PREPARATION_PACK_ROOT):
            try:
                suffix = logical.relative_to(accepted_root)
                break
            except ValueError:
                pass
        if suffix is None:
            raise ValueError(f"{rel}: logical pack path is outside the accepted frozen caches")
        root = (self.runtime_pack_cache / suffix).resolve()
        if not root.is_relative_to(self.runtime_pack_cache):
            raise ValueError(f"{rel}: runtime pack mapping escaped its private cache")
        return root

    def load(self, rel):
        if rel == self.rel:
            return self.value
        root = self.root_for(rel)
        expected = self.info[rel]["data_identity"]["file_sha256"]
        for name in ("samples.json", "samples.npz", "build_report.json"):
            if sha256_file(root / name) != expected[name]:
                raise ValueError(f"{rel}: runtime pack {name} differs from the accepted plan")
        metadata = json.loads((root / "samples.json").read_text())
        with np.load(root / "samples.npz", allow_pickle=False) as archive:
            images, masks = np.array(archive["images"], copy=True), np.array(archive["masks"], copy=True)
        self.rel, self.value = rel, (metadata["records"], images, masks)
        return self.value


def reconstruct(ep, cache):
    records, images, masks = cache.load(ep["pack_rel"])
    rows = [i for i, row in enumerate(records) if row["task"] == ep["task"]]
    local = {global_i: i for i, global_i in enumerate(rows)}
    built = replay_episode(images[rows], masks[rows], [records[i] for i in rows], ep["task"], ep["q"],
                           [local[i] for i in ep["demo_global"]], ep["cid"])
    if rows[ep["q"]] != ep["query_global"] or built["episode_pixels_sha256"] != ep["episode_pixels_sha256"]:
        raise ValueError(f"{ep['episode_id']}: frozen replay differs")
    return built


def prompt_text(task, demo_count=P):
    image = "<img>" + "".join(f"<img_{index:05d}>" for index in range(NUM_IMAGE_TOKENS)) + "</img>"
    edit = "<edit>" + "".join(f"<edit_{index:03d}>" for index in range(NUM_EDIT_TOKENS)) + "</edit>"
    pairs = f" source image {image} to target image {image}" * demo_count
    instruction = instruction_for_task(task)
    body = (f'Here is an image manipulation instruction "{instruction}", which can edit{pairs}. '
            f"The editing is embedded in {edit}. Learn from the instruction with the exemplar pairs and "
            f"apply the same manipulation to this image {image}.")
    return f"[INST] {body} [/INST]\n"


def prompt_inputs(tokenizer, device, dtype, task):
    import torch
    import xformers.ops as xops
    prompt = prompt_text(task, P); ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, device=device, dtype=torch.long)
    boi_id = tokenizer.encode("<img>", add_special_tokens=False)[0]
    eoi_id = tokenizer.encode("</img>", add_special_tokens=False)[0]
    boi = torch.where(input_ids == boi_id)[0].cpu().tolist(); eoi = torch.where(input_ids == eoi_id)[0].cpu().tolist()
    if len(boi) != 2 * P + 1 or len(eoi) != len(boi):
        raise RuntimeError("released multishot prompt image-token layout changed")
    cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for start, end in zip(boi, eoi): cmp_mask[start + 1:end] = True
    for start, end in zip(boi[1:-1:2], eoi[1:-1:2]): target_mask[start + 1:end] = True
    scope = xops.LowerTriangularMask().materialize((len(ids), len(ids)), dtype=dtype, device=device)
    boe_id = tokenizer.encode("<edit>", add_special_tokens=False)[0]
    eoe_id = tokenizer.encode("</edit>", add_special_tokens=False)[0]
    for _, end in zip(torch.where(input_ids == boe_id)[0].cpu().tolist(),
                      torch.where(input_ids == eoe_id)[0].cpu().tolist()):
        scope[end + 1:, :eoi[1:-1:2][-1] + 1] = -float("inf")
    return {"input_ids": input_ids.unsqueeze(0), "ids_cmp_mask": cmp_mask.unsqueeze(0),
            "ids_exemplar_target_mask": target_mask.unsqueeze(0), "scope_mask": scope.unsqueeze(0),
            "prompt": prompt}


def noise_seed(ep):
    key = f"{SEED}|{ep['task']}|{int(ep['q'])}|{P}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def recursive_shard_output(output, mesh):
    import torch
    import torch_xla.distributed.spmd as xs
    def visit(value):
        if isinstance(value, torch.Tensor) and value.ndim:
            spec = [None] * value.ndim
            candidates = [axis for axis, size in enumerate(value.shape) if axis > 0 and size % mesh.size() == 0]
            axis = candidates[0] if candidates else next((axis for axis, size in enumerate(value.shape)
                                                           if size % mesh.size() == 0), None)
            if axis is not None: spec[axis] = "fsdp"; xs.mark_sharding(value, mesh, tuple(spec))
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif hasattr(value, "to_tuple"): visit(value.to_tuple())
    visit(output)


def install_vae_input_dtype_adapter(vae, expected_dtype):
    """Undo the released pipeline's unconditional image.float() at the VAE boundary."""
    if getattr(vae, "_instamanip_vae_input_adapter", None) == VAE_INPUT_ADAPTER:
        return vae
    floating = [parameter for parameter in vae.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != expected_dtype for parameter in floating):
        raise RuntimeError("VAE parameters do not match the audited pipeline dtype")
    original_encode = vae.encode

    @functools.wraps(original_encode)
    def encode(sample, *args, **kwargs):
        if not sample.is_floating_point():
            raise RuntimeError("VAE encode input must be floating point")
        return original_encode(sample.to(dtype=expected_dtype), *args, **kwargs)

    vae.encode = encode
    vae._instamanip_vae_input_adapter = VAE_INPUT_ADAPTER
    return vae


def xla_memory_evidence(get_memory_info, device):
    """Record real counters, or only the known SPMD virtual-device limitation."""
    try:
        values = {key: int(value) for key, value in get_memory_info(device).items()}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if "MemoryInfo not supported for SPMD virtual device" not in message:
            raise RuntimeError("mandatory XLA memory evidence unavailable") from exc
        return {"error": message}
    if not values or any(value < 0 for value in values.values()) or not any(values.values()):
        raise RuntimeError("mandatory XLA memory evidence is empty or invalid")
    return values


def load_runtime(args, device, mesh):
    import torch
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    from instamanip_loader import build_agent, build_diffusion, build_llm, build_tokenizer, build_visual
    from src.models.mllm.modeling_llama_xformer import (
        LlamaAttention, LlamaDecoderLayer, apply_rotary_pos_emb)
    from src.processer.transforms import get_transform
    from xla_static_generation import (
        STATIC_CACHE_SCHEMA, bind_static_generate, install_static_attention)
    from xla_dynamic_euler import (
        DYNAMIC_EULER_SCHEMA, dynamic_euler_audit, install_dynamic_xla_euler)
    dtype = torch.bfloat16
    if STATIC_CACHE_SCHEMA != XLA_GENERATION_CACHE:
        raise RuntimeError("XLA static generation schema differs")
    install_static_attention(LlamaAttention, apply_rotary_pos_emb)
    tokenizer = build_tokenizer(); visual, visual_audit = build_visual(dtype)
    llm = build_llm(dtype); agent, agent_audit = build_agent(llm, dtype)
    bind_static_generate(agent)
    adapter, vae, scheduler, diffusion_audit = build_diffusion(dtype)
    if DYNAMIC_EULER_SCHEMA != XLA_DIFFUSION_EXECUTION:
        raise RuntimeError("XLA dynamic Euler schema differs")
    install_dynamic_xla_euler(scheduler, expected_steps=DIFFUSION_STEPS)
    visual = visual.to(device).eval()
    auto_wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    agent = SpmdFullyShardedDataParallel(agent.eval(), mesh=mesh, auto_wrap_policy=auto_wrap,
        shard_output=recursive_shard_output)
    adapter = adapter.to(device).eval(); vae = vae.to(device).eval()
    install_vae_input_dtype_adapter(vae, dtype)
    transform = get_transform(type="clip", image_size=448, keep_ratio=False)
    adapter.init_pipe(vae=vae, scheduler=scheduler, visual_encoder=visual,
                      image_transform=transform, dtype=dtype, device=device)
    xm.mark_step(); xm.wait_device_ops()
    return {"tokenizer": tokenizer, "visual": visual, "agent": agent, "adapter": adapter,
            "pipe": adapter.sdxl_pipe, "image_transform": transform, "dtype": dtype,
            "mesh": mesh,
            "xla_generation_cache": STATIC_CACHE_SCHEMA,
            "xla_dynamic_euler": dynamic_euler_audit(scheduler),
            "vae_input_adapter": VAE_INPUT_ADAPTER,
            "audits": {"visual": visual_audit.as_dict(), "agent": agent_audit.as_dict(),
                       "diffusion": diffusion_audit.as_dict()}}


def pil(value):
    array = np.asarray(value, np.uint8)
    if array.shape != (256, 256, 3): raise ValueError("InstaManip tile shape differs")
    return Image.fromarray(array, "RGB")


def official_visual_inputs(sources, targets, query):
    """Match eval_model_multishot.py: exemplars visit 1024 before the 448 transform."""
    if len(sources) != P or len(targets) != P or query.mode != "RGB":
        raise ValueError("official P=3 visual input contract differs")
    exemplars = []
    for image in [*sources, *targets]:
        if image.mode != "RGB":
            raise ValueError("InstaManip visual input must be RGB")
        # Deliberately omit resample: this is the exact implicit Pillow call in the official evaluator.
        exemplars.append(image.resize((1024, 1024)))
    return exemplars + [query]


def prepare_diffusion_noise(torch, ep, device, dtype):
    seed = noise_seed(ep)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn((1, 4, 128, 128), generator=generator, dtype=torch.float32)
    source_sha256 = hashlib.sha256(source.numpy().tobytes()).hexdigest()
    pipeline = source.to(device=device, dtype=dtype)
    if pipeline.dtype != dtype or tuple(pipeline.shape) != (1, 4, 128, 128):
        raise RuntimeError("diffusion latent did not enter the pipeline in its audited BF16 contract")
    return pipeline, {"seed": seed, "sha256": source_sha256, "source_dtype": "float32",
                      "pipeline_dtype": str(dtype).removeprefix("torch.")}


def _mark_batch_sharding(value, runtime):
    import torch_xla.distributed.spmd as xs
    if int(value.shape[0]) % INFERENCE_BATCH_SIZE:
        raise RuntimeError("TPU batch axis is not divisible by the global device count")
    xs.mark_sharding(value, runtime["mesh"], ("fsdp",) + (None,) * (value.ndim - 1))
    return value


def _memory_value(path, key):
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == key + ":":
            return int(fields[1]) * 1024
    raise RuntimeError(f"{key} missing from {path}")


def host_memory_snapshot():
    return {
        "process_rss_bytes": _memory_value("/proc/self/status", "VmRSS"),
        "host_mem_available_bytes": _memory_value("/proc/meminfo", "MemAvailable"),
    }


def make_xla_diffusion_step_callback(xm, xr, log_path, batch_index):
    records = []

    def callback(step_index, _timestep, _latents):
        # The pinned InstaManip custom pipeline invokes its legacy callback
        # immediately after scheduler.step for every denoising step.
        # Materializing here bounds the lazy XLA graph without changing
        # latents, scheduler state, prompts, noise, weights, or step count.
        xm.mark_step()
        xm.wait_device_ops()
        memory = host_memory_snapshot()
        cached_graphs = int(xr.get_num_cached_compilation_graph())
        record = {
            "schema": "instamanip_xla_diffusion_step_memory_v2",
            "batch_index": int(batch_index),
            "step_index": int(step_index),
            "execution_contract": XLA_DIFFUSION_EXECUTION,
            "compilation_cache_limit": XLA_COMPILATION_CACHE_LIMIT,
            "cached_compilation_graphs": cached_graphs,
            **memory,
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
        print("XLA_DIFFUSION_STEP " + json.dumps(record, sort_keys=True), flush=True)
        records.append(record)
        if not 0 < cached_graphs <= XLA_COMPILATION_CACHE_LIMIT:
            raise RuntimeError("in-memory XLA compilation cache exceeded its fail-closed bound")
        if memory["process_rss_bytes"] > MAX_PROCESS_RSS_BYTES:
            raise RuntimeError("process RSS exceeded the fail-closed diffusion memory guard")
        if memory["host_mem_available_bytes"] < MIN_HOST_MEM_AVAILABLE_BYTES:
            raise RuntimeError("host MemAvailable fell below the fail-closed diffusion memory guard")

    return callback, records


def run_episode_batch(runtime, items, device, step_log_path, batch_index):
    """Run one fixed 16-way diffusion batch while preserving per-query agents.

    The released agent remains batch-one and globally FSDP-sharded.  Its forced
    image-token block is evaluated once per query.  SDXL is weight-replicated,
    so its batch axis is explicitly sharded over the same 16-device mesh; this
    turns otherwise duplicated replicas into one query per device without
    changing any query's official prompt, noise, or 50-step sampler contract.
    """
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm
    if not items or len(items) > INFERENCE_BATCH_SIZE:
        raise RuntimeError("InstaManip batch size is outside the audited range")
    tasks = {ep["task"] for ep, _ in items}
    if len(tasks) != 1:
        raise RuntimeError("text-conditioned batches must contain exactly one task")
    task = next(iter(tasks))
    real_count = len(items)
    padded = list(items) + [items[-1]] * (INFERENCE_BATCH_SIZE - real_count)
    transform, dtype = runtime["image_transform"], runtime["dtype"]
    queries, all_visual_inputs = [], []
    for ep, built in padded:
        del ep
        sources = [pil(value) for value in built["demo_inputs"]]
        targets = [pil(value) for value in built["demo_targets"]]
        query = pil(built["query_input"])
        queries.append(query)
        all_visual_inputs.extend(official_visual_inputs(sources, targets, query))
    visual_batch = torch.stack([transform(image) for image in all_visual_inputs], 0).to(
        device=device, dtype=dtype)
    _mark_batch_sharding(visual_batch, runtime)
    with torch.no_grad():
        embeddings = runtime["visual"](visual_batch); xm.mark_step()
        mask = torch.tensor([True], device=device, dtype=torch.bool)
        prompt = prompt_inputs(runtime["tokenizer"], device, dtype, task)
        image_features, texts = [], []
        visual_inputs_per_episode = 2 * P + 1
        for index in range(INFERENCE_BATCH_SIZE):
            offset = index * visual_inputs_per_episode
            output = runtime["agent"].generate(tokenizer=runtime["tokenizer"], input_ids=prompt["input_ids"],
                image_embeds={"new": embeddings[offset + 2 * P:offset + 2 * P + 1],
                              "exemplar_source": embeddings[offset:offset + P],
                              "exemplar_target": embeddings[offset + P:offset + 2 * P]},
                embeds_cmp_mask={"new": mask, "exemplar_source": [mask] * P,
                                 "exemplar_target": [mask] * P},
                ids_cmp_mask=prompt["ids_cmp_mask"], ids_exemplar_target_mask=prompt["ids_exemplar_target_mask"],
                scope_mask=prompt["scope_mask"], patch_positions=None, max_new_tokens=MAX_NEW_TOKENS,
                num_img_gen_tokens=NUM_IMAGE_TOKENS, device=device)
            xm.mark_step()
            if not output["has_img_output"] or output["num_gen_imgs"] < 1:
                raise RuntimeError(f"model produced no image embedding at batch index {index}: {output['text']!r}")
            image_features.append(output["img_gen_feat"][0:1])
            texts.append(re.sub("<[^>]*>", "", output["text"]))
        positive_features = torch.cat(image_features, 0)
        _mark_batch_sharding(positive_features, runtime)
        negative_pixels = torch.zeros(
            INFERENCE_BATCH_SIZE, 3, 448, 448, device=device, dtype=dtype)
        _mark_batch_sharding(negative_pixels, runtime)
        negative = runtime["visual"](negative_pixels)
        negative = F.avg_pool1d(negative.permute(0, 2, 1), kernel_size=4, stride=4).permute(0, 2, 1)
        resampler_inputs = torch.cat([positive_features, negative], 0)
        _mark_batch_sharding(resampler_inputs, runtime)
        prompts, pooled = runtime["adapter"].resampler(resampler_inputs)
        positive_prompt, negative_prompt = prompts.chunk(2)
        positive_pooled, negative_pooled = pooled.chunk(2)
        noise_rows, noise_audits = [], []
        for ep, _ in padded:
            noise, audit = prepare_diffusion_noise(torch, ep, device, dtype)
            noise_rows.append(noise); noise_audits.append(audit)
        noise = torch.cat(noise_rows, 0)
        _mark_batch_sharding(noise, runtime)
        for value in (positive_prompt, negative_prompt, positive_pooled, negative_pooled):
            _mark_batch_sharding(value, runtime)
        # Deliberately omit resample to mirror the official evaluator's query-to-SDXL resize call.
        image_tensor = runtime["pipe"].image_processor.preprocess(
            [query.resize((1024, 1024)) for query in queries], height=1024, width=1024).to(
                device=device, dtype=dtype)
        _mark_batch_sharding(image_tensor, runtime)
        step_callback, step_memory = make_xla_diffusion_step_callback(
            xm, runtime["xr"], step_log_path, batch_index)
        images = runtime["pipe"](image=image_tensor,
            prompt_embeds=positive_prompt, negative_prompt_embeds=negative_prompt,
            pooled_prompt_embeds=positive_pooled, negative_pooled_prompt_embeds=negative_pooled,
            guidance_scale=GUIDANCE_SCALE, image_guidance_scale=IMAGE_GUIDANCE_SCALE,
            num_inference_steps=DIFFUSION_STEPS, generator=None, latents=noise,
            height=1024, width=1024,
            callback=step_callback, callback_steps=1).images
        xm.mark_step(); xm.wait_device_ops()
        if len(step_memory) != DIFFUSION_STEPS:
            raise RuntimeError("diffusion step-boundary coverage differs")
        runtime["last_diffusion_step_memory"] = step_memory
    if len(images) != INFERENCE_BATCH_SIZE:
        raise RuntimeError("batched diffusion returned the wrong image count")
    return [(images[index].convert("RGB"), noise_audits[index], texts[index], prompt["prompt"])
            for index in range(real_count)]


def validate_qualification(path, expected_sha, validation_plan_identity=None,
                           asset_manifest_sha256=None):
    if not path or not expected_sha or not path.is_file() or sha256_file(path) != expected_sha:
        raise SystemExit("a hash-pinned v5litepod-32 qualification receipt is required")
    receipt = json.loads(path.read_text())
    expected = {"schema": QUALIFICATION_SCHEMA, "accepted": True,
                "evaluation_contract": PORT_QUALIFICATION_EVALUATION_CONTRACT, "manifest_sha256": MANIFEST_SHA256,
                "source_commit": SOURCE_COMMIT,
                "qualified_accelerator": EXPECTED_ACCELERATOR,
                "qualified_devices": EXPECTED_GLOBAL_DEVICES,
                "qualified_addressable_devices": EXPECTED_ADDRESSABLE_DEVICES,
                "qualified_workers": EXPECTED_WORKERS, "topology": EXPECTED_TOPOLOGY,
                "backend": "TPU", "xla_generation_cache": XLA_GENERATION_CACHE,
                "xla_diffusion_execution": XLA_DIFFUSION_EXECUTION,
                "xla_compilation_cache_limit": XLA_COMPILATION_CACHE_LIMIT,
                "xla_diffusion_execution": XLA_DIFFUSION_EXECUTION,
                "vae_input_adapter": VAE_INPUT_ADAPTER,
                "repeat_pixel_identical": True, "repeat_png_identical": True,
                "cross_topology_reference_claimed": False,
                "sensible_output_gate": "PASS",
                "qualification_batch_size": INFERENCE_BATCH_SIZE,
                "qualification_unique_queries": SMOKE_QUERY_COUNT,
                "validation_queries": TOTAL_VALIDATION_QUERIES,
                "validation_task_groups": VALIDATION_TASK_GROUPS,
                "projection_mode": QUALIFICATION_PROJECTION_MODE,
                "qualification_scope": "actual_p_exact_model_same_topology_repeat_v1",
                "maximum_total_seconds": MAX_TOTAL_SECONDS,
                "under_five_days": True,
                "tpu_native_no_cuda_parity_accepted": True, "cuda_parity_claimed": False}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SystemExit("TPU qualification receipt contract differs")
    if (asset_manifest_sha256 is not None
            and receipt.get("asset_manifest_sha256") != asset_manifest_sha256):
        raise SystemExit("TPU qualification receipt asset binding differs")
    if receipt.get("required_scoped_vmem_kib") != 32768:
        raise SystemExit("TPU qualification receipt has the wrong scoped-VMEM contract")
    projected = receipt.get("projected_total_seconds")
    if (isinstance(projected, bool) or not isinstance(projected, (int, float))
            or not 0 < projected <= MAX_TOTAL_SECONDS):
        raise SystemExit("TPU qualification receipt fails the five-day projection")
    bound = receipt.get("validation_plan_identity")
    if not isinstance(bound, str) or len(bound) != 64:
        raise SystemExit("TPU qualification receipt lacks its validation-plan identity")
    if validation_plan_identity is not None and validation_plan_identity != bound:
        raise SystemExit("TPU qualification receipt is not bound to this validation plan")
    return receipt


def validate_validation_gate(path, expected_sha, qualification_sha, validation_plan_identity):
    if not path or not expected_sha or not path.is_file() or sha256_file(path) != expected_sha:
        raise SystemExit("a hash-pinned corrected-V3 validation receipt is required")
    gate = json.loads(path.read_text())
    expected = {"schema": VALIDATION_GATE_SCHEMA, "status": "PASS", "phase": "validation",
                "evaluation_contract": EVALUATION_CONTRACT, "manifest_sha256": MANIFEST_SHA256,
                "validation_episodes": 221, "p": P, "readout": READOUT,
                "metric_protocol": CORRECTED_PROTOCOL, "release_sha256": CORRECTED_RELEASE,
                "qualification_receipt_sha256": qualification_sha,
                "validation_plan_identity": validation_plan_identity,
                "asset_manifest_sha256": "32ce1fb1f79086bb0cccc5eb6da50a0e843197ce836690db961ffe13ca487ea5"}
    if any(gate.get(key) != value for key, value in expected.items()):
        raise SystemExit("validation receipt does not admit this InstaManip TEST")
    if gate.get("plan_identity") != validation_plan_identity:
        raise SystemExit("validation receipt plan binding differs")
    units_sha, packs = gate.get("units_manifest_sha256"), gate.get("packs")
    if (not isinstance(units_sha, str) or len(units_sha) != 64 or not isinstance(packs, list)
            or len(packs) != 1 or packs[0].get("pack_rel") != "bench_v2_val"
            or packs[0].get("episodes") != 221):
        raise SystemExit("validation receipt corrected-score coverage differs")
    return gate


def validate_topology_contract(expected_tpu, expected_zone, expected_accelerator, expected_devices):
    named_authorization = os.environ.get("NAMED_RESOURCE_PYTHON_AUTHORIZED", "")
    authorized_name = ("kristine" in expected_tpu.lower() or
                       named_authorization == expected_tpu)
    if (not authorized_name or expected_zone != "us-west4-a"
            or expected_accelerator != EXPECTED_ACCELERATOR
            or expected_devices != EXPECTED_GLOBAL_DEVICES):
        raise SystemExit("text-conditioned InstaManip requires an authorized west4 v5litepod-32 TPU")


def initial_view(pack, plan):
    return {"schema": VIEW_SCHEMA, "phase": plan["phase"], "condition": "real", "p": P,
            "evaluation_contract": EVALUATION_CONTRACT,
            "plan_identity": plan["plan_identity"], "asset_manifest_sha256": plan["asset_manifest_sha256"],
            "data_identity": pack["data_identity"], "data_identity_sha256": pack["data_identity_sha256"],
            "expected_episode_keys": pack["expected_episode_keys"], "episodes": {}, "complete": False}


def load_views(root, plan):
    views, done = {}, set()
    for pack in plan["packs"]:
        view_root = root / "views" / pack["slug"]; path = view_root / "results.json"
        view = json.loads(path.read_text()) if path.exists() else initial_view(pack, plan)
        if (view.get("plan_identity"), view.get("asset_manifest_sha256"), view.get("data_identity"),
                view.get("evaluation_contract")) != (plan["plan_identity"], plan["asset_manifest_sha256"],
                pack["data_identity"], EVALUATION_CONTRACT):
            raise SystemExit(f"{path}: resume identity differs")
        for key, episode in view.get("episodes", {}).items():
            artifacts = episode.get("artifacts", {}); verify_descriptor(view_root, artifacts.get("query_input"), key + ".input")
            verify_descriptor(view_root, artifacts.get("query_target"), key + ".target")
            verify_descriptor(view_root, artifacts.get("readouts", {}).get(READOUT, {}).get("real"), key + ".prediction")
            verify_descriptor(view_root, artifacts.get("raw_prediction"), key + ".raw", 1024)
            done.add(f"{pack['rel']}|{key}")
        views[pack["rel"]] = view; atomic_json(path, view)
    return views, done


def infer(args):
    plan_phase = "validation" if args.phase in ("smoke", "validation") else "test"
    plan = verify_plan(args.plan, plan_phase); verify_sources(args.source_dir)
    from asset_contract import release_verified_assets, verify_assets
    assets = verify_assets(args.assets_dir)
    if plan.get("asset_manifest_sha256") != assets["manifest_sha256"]:
        raise SystemExit("plan asset registration differs")
    qualification = None
    if args.phase in ("validation", "test"):
        qualification = validate_qualification(
            args.qualification_receipt, args.qualification_receipt_sha256,
            plan["plan_identity"], plan["asset_manifest_sha256"])
    if args.phase == "test":
        validate_validation_gate(args.validation_gate, args.validation_gate_sha256,
                                 args.qualification_receipt_sha256,
                                 qualification["validation_plan_identity"])
    validate_topology_contract(args.expected_tpu, args.expected_zone,
                               args.expected_accelerator, args.expected_devices)
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        raise SystemExit("refusing non-TPU backend")
    if "--xla_tpu_scoped_vmem_limit_kib=32768" not in os.environ.get("LIBTPU_INIT_ARGS", "").split():
        raise SystemExit("audited 32MiB scoped-VMEM compiler flag is required")
    if os.environ.get("XLA_COMPILATION_CACHE_SIZE") != str(XLA_COMPILATION_CACHE_LIMIT):
        raise SystemExit("exact bounded in-memory XLA compilation cache is required")
    args.results_root.mkdir(parents=True, exist_ok=True); args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["INSTAMANIP_SOURCE_DIR"] = str(args.source_dir.resolve())
    os.environ["INSTAMANIP_ASSETS_DIR"] = str(args.assets_dir.resolve())
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    if not callable(getattr(xr, "get_num_cached_compilation_graph", None)):
        raise SystemExit("pinned Torch/XLA compilation-cache evidence API is unavailable")
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False); xr.use_spmd()
    count, local, backend = int(xr.global_runtime_device_count()), int(xr.addressable_runtime_device_count()), str(xr.device_type())
    if (backend.upper() != "TPU" or count != EXPECTED_GLOBAL_DEVICES
            or local != EXPECTED_ADDRESSABLE_DEVICES):
        raise SystemExit(
            f"refusing backend={backend} global={count} addressable={local}; "
            f"exact {EXPECTED_GLOBAL_DEVICES}/{EXPECTED_ADDRESSABLE_DEVICES} required")
    mesh = xs.Mesh(np.arange(EXPECTED_GLOBAL_DEVICES), (EXPECTED_GLOBAL_DEVICES,), ("fsdp",))
    xs.set_global_mesh(mesh); device = xm.xla_device()
    evidence = {"backend": backend, "global_devices": count, "addressable_devices": local,
        "device": str(device), "expected_tpu": args.expected_tpu, "expected_zone": args.expected_zone,
        "expected_accelerator": args.expected_accelerator, "host": socket.gethostname(), "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"), "phase": args.phase,
        "backend_label": "PyTorch/XLA v5litepod-32 eight-worker SPMD-FSDP port; official CUDA numerical parity is not established",
        "cuda_parity_claimed": False, "scoped_vmem_kib": 32768,
        "xla_generation_cache": XLA_GENERATION_CACHE,
        "xla_diffusion_execution": XLA_DIFFUSION_EXECUTION,
        "xla_compilation_cache_limit": XLA_COMPILATION_CACHE_LIMIT,
        "vae_input_adapter": VAE_INPUT_ADAPTER}
    atomic_json(args.results_root / "backend.json", evidence)
    runtime = load_runtime(args, device, mesh)
    runtime["xr"] = xr
    gc.collect()
    asset_release = release_verified_assets(
        args.assets_dir, assets, args.results_root / "ASSETS_RELEASED_AFTER_TPU_LOAD.json")
    if (asset_release.get("status") != "PASS"
            or asset_release.get("released_bytes") != assets["bytes"]):
        raise RuntimeError("authenticated run-local asset release failed")
    views, done = load_views(args.results_root, plan)
    cache, by_pack = PackCache(plan), {row["rel"]: row for row in plan["packs"]}
    pending = [ep for ep in plan["episodes"] if f"{ep['pack_rel']}|{ep['key']}" not in done]
    if args.phase == "smoke": pending = pending[:SMOKE_QUERY_COUNT]
    if args.max_task_groups:
        if args.phase != "validation" or args.max_task_groups != 1:
            raise SystemExit("process-isolated inference admits exactly one validation task group")
    smoke_batch_seconds = None
    run_tag = hashlib.sha256(str(args.results_root).encode()).hexdigest()[:12]
    grouped = []
    for episode in pending:
        if not grouped or grouped[-1][0]["task"] != episode["task"]:
            grouped.append([])
        grouped[-1].append(episode)
    if args.max_task_groups:
        grouped = grouped[:args.max_task_groups]
    batches = [group[offset:offset + INFERENCE_BATCH_SIZE]
               for group in grouped
               for offset in range(0, len(group), INFERENCE_BATCH_SIZE)]
    for batch_index, episodes in enumerate(batches):
        items = [(ep, reconstruct(ep, cache)) for ep in episodes]
        began = time.monotonic()
        outputs = run_episode_batch(
            runtime, items, device,
            args.results_root / "DIFFUSION_STEP_MEMORY.jsonl", batch_index)
        batch_seconds = time.monotonic() - began
        xm.rendezvous(f"instamanip_materialized_{run_tag}_{batch_index}")
        if args.phase == "smoke": smoke_batch_seconds = batch_seconds
        for (ep, built), (image, noise, text, prompt) in zip(items, outputs):
            pack, view = by_pack[ep["pack_rel"]], views[ep["pack_rel"]]
            root = args.results_root / "views" / pack["slug"]
            episode_dir = root / f"{ep['task']}_q{ep['q']}"
            input_path, target_path = episode_dir / "query_input.png", episode_dir / "query_target.png"
            raw_path = episode_dir / f"InstaManip_real_{READOUT_TAG}_query_raw1024.png"
            pred_path = episode_dir / f"InstaManip_real_{READOUT_TAG}_query.png"
            atomic_png(input_path, built["query_input"]); atomic_png(target_path, built["query_target"])
            atomic_png(raw_path, np.asarray(image, np.uint8))
            atomic_png(pred_path, np.asarray(image.resize((256, 256), Image.Resampling.BICUBIC), np.uint8))
            view["episodes"][ep["key"]] = {"task": ep["task"], "q": ep["q"], "p": P, "cid": ep["cid"],
                "query_record": ep["query_record"], "demo_records": ep["demo_records"], "prediction_complete": True,
                "complete": True, "readouts": {READOUT: {"real": {"model": "InstaManip-17B-3shot",
                    "source_commit": SOURCE_COMMIT, "asset_manifest_sha256": assets["manifest_sha256"],
                    "evaluation_contract": EVALUATION_CONTRACT,
                    "instruction": instruction_for_task(ep["task"]),
                    "task_instructions_sha256": TASK_INSTRUCTIONS_SHA256,
                    "prompt_policy": PROMPT_POLICY,
                    "native_prompt_count": P, "max_new_tokens": MAX_NEW_TOKENS,
                    "diffusion_steps": DIFFUSION_STEPS, "guidance_scale": GUIDANCE_SCALE,
                    "image_guidance_scale": IMAGE_GUIDANCE_SCALE, "noise_seed": noise["seed"],
                    "noise_sha256": noise["sha256"], "noise_source_dtype": noise["source_dtype"],
                    "pipeline_latent_dtype": noise["pipeline_dtype"],
                    "exemplar_preprocessing": "RGB 256 -> Pillow implicit resize 1024 -> official CLIP transform 448",
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "generated_text": text, "seconds": batch_seconds / len(episodes),
                    "batch_wall_seconds": batch_seconds, "batch_size": len(episodes),
                    "backend": "PyTorch/XLA v5litepod-32 eight-worker SPMD-FSDP plus device-sharded SDXL batch",
                    "cuda_parity_claimed": False}}}, "artifacts": {"query_input": descriptor(input_path, root, 256),
                    "query_target": descriptor(target_path, root, 256), "raw_prediction": descriptor(raw_path, root, 1024),
                    "readouts": {READOUT: {"real": descriptor(pred_path, root, 256)}}}}
            atomic_json(root / "results.json", view)
        xm.rendezvous(f"instamanip_artifacts_{run_tag}_{batch_index}")
    if args.phase == "smoke":
        if smoke_batch_seconds is None or len(pending) != SMOKE_QUERY_COUNT:
            raise RuntimeError("fresh full-device smoke batch did not complete")
        memory = xla_memory_evidence(xm.get_memory_info, device)
        samples = []
        for ep in pending:
            view_root = args.results_root / "views" / by_pack[ep["pack_rel"]]["slug"]
            artifacts = views[ep["pack_rel"]]["episodes"][ep["key"]]["artifacts"]
            pred = artifacts["readouts"][READOUT]["real"]
            query = artifacts["query_input"]
            samples.append({"episode_id": ep["episode_id"],
                "prediction_path": str((view_root / pred["path"]).relative_to(args.results_root)),
                "prediction_sha256": pred["sha256"], "prediction_pixels_sha256": pred["pixels_sha256"],
                "query_path": str((view_root / query["path"]).relative_to(args.results_root)),
                "query_sha256": query["sha256"], "query_pixels_sha256": query["pixels_sha256"]})
        marker = {"schema": SMOKE_SCHEMA, "evaluation_contract": EVALUATION_CONTRACT,
            "manifest_sha256": MANIFEST_SHA256,
            "source_commit": SOURCE_COMMIT, "asset_manifest_sha256": assets["manifest_sha256"],
            "validation_plan_identity": plan["plan_identity"],
            "device_count": count, "addressable_device_count": local,
            "worker_count": EXPECTED_WORKERS, "topology": EXPECTED_TOPOLOGY,
            "accelerator": EXPECTED_ACCELERATOR,
            "backend": "TPU", "expected_tpu": args.expected_tpu, "cuda_parity_claimed": False,
            "xla_generation_cache": runtime["xla_generation_cache"],
            "xla_diffusion_execution": XLA_DIFFUSION_EXECUTION,
            "xla_compilation_cache_limit": XLA_COMPILATION_CACHE_LIMIT,
            "vae_input_adapter": runtime["vae_input_adapter"],
            "batch_size": INFERENCE_BATCH_SIZE, "batch_wall_seconds": smoke_batch_seconds,
            "amortized_seconds_per_query": smoke_batch_seconds / INFERENCE_BATCH_SIZE,
            "memory_after_first_sample": memory,
            "diffusion_step_memory": {
                "steps": len(runtime["last_diffusion_step_memory"]),
                "max_process_rss_bytes": max(
                    row["process_rss_bytes"] for row in runtime["last_diffusion_step_memory"]),
                "min_host_mem_available_bytes": min(
                    row["host_mem_available_bytes"] for row in runtime["last_diffusion_step_memory"]),
                "process_rss_limit_bytes": MAX_PROCESS_RSS_BYTES,
                "host_mem_available_floor_bytes": MIN_HOST_MEM_AVAILABLE_BYTES,
                "compilation_cache_limit": XLA_COMPILATION_CACHE_LIMIT,
                "max_cached_compilation_graphs": max(
                    row["cached_compilation_graphs"]
                    for row in runtime["last_diffusion_step_memory"]),
            },
            "samples": samples}
        xm.rendezvous(f"instamanip_before_marker_{run_tag}")
        atomic_json(args.results_root / "SMOKE_COMPLETE.json", marker)
        xm.rendezvous(f"instamanip_marker_written_{run_tag}")
        return
    completed = sum(len(view["episodes"]) for view in views.values())
    if args.max_task_groups and completed != len(plan["episodes"]):
        if not batches or completed <= len(done):
            raise RuntimeError("process-isolated validation made no durable progress")
        atomic_json(args.results_root / "PARTIAL_INFERENCE_PROGRESS.json", {
            "schema": "instamanip_q50_process_isolated_progress_v1",
            "phase": args.phase,
            "evaluation_contract": EVALUATION_CONTRACT,
            "plan_identity": plan["plan_identity"],
            "asset_manifest_sha256": assets["manifest_sha256"],
            "completed_episodes": completed,
            "expected_episodes": len(plan["episodes"]),
            "completed_task": grouped[0][0]["task"],
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        return
    for pack in plan["packs"]:
        view = views[pack["rel"]]
        if set(view["episodes"]) != set(pack["expected_episode_keys"]):
            raise SystemExit(f"{pack['rel']}: incomplete phase coverage")
        view["complete"] = True; atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
    atomic_json(args.results_root / "INFERENCE_COMPLETE.json", {"schema": "instamanip_bench_v2_native_p3_inference_complete_v1",
        "phase": args.phase, "evaluation_contract": EVALUATION_CONTRACT,
        "plan_identity": plan["plan_identity"], "asset_manifest_sha256": assets["manifest_sha256"],
        "qualification_receipt_sha256": args.qualification_receipt_sha256,
        "validation_gate_sha256": args.validation_gate_sha256 or None, "episodes": len(plan["episodes"]),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare-plan"); prep.add_argument("--phase", choices=("validation", "test"), required=True)
    prep.add_argument("--out", type=Path, required=True); prep.add_argument("--pack-cache", type=Path, required=True)
    prep.add_argument("--plan-pack-cache", type=Path, required=True,
                      help="logical pack-cache path serialized into the plan identity")
    prep.add_argument("--source-dir", type=Path, required=True)
    prep.add_argument("--asset-manifest-sha256", required=True)
    run = subs.add_parser("infer"); run.add_argument("--phase", choices=("smoke", "validation", "test"), required=True)
    for name in ("plan", "results_root", "source_dir", "assets_dir", "cache_dir"):
        run.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    run.add_argument("--expected-tpu", required=True); run.add_argument("--expected-zone", required=True)
    run.add_argument("--expected-accelerator", required=True); run.add_argument("--expected-devices", type=int, required=True)
    run.add_argument("--qualification-receipt", type=Path); run.add_argument("--qualification-receipt-sha256", default="")
    run.add_argument("--validation-gate", type=Path); run.add_argument("--validation-gate-sha256", default="")
    run.add_argument("--max-task-groups", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args(); return prepare_plan(args) if args.command == "prepare-plan" else infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
