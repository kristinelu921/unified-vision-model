#!/usr/bin/env python3
"""Frozen RelationAdapter native-P=1 runner for all Bench v2 packs."""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image

LIVE_REPO = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
TRUSTED_MANIFEST = LIVE_REPO / "docs/bench_v2_audit/manifest_v1.json"
MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"
PACK_URI_ROOT = "gs://kmh-gcp-us-east5/data/xtiange/mt_eval_packs"
PLAN_SCHEMA = "relation_adapter_bench_v2_native_p1_v5p16_plan_v1"
SCHEMA = "relation_adapter_bench_v2_native_p1_v5p16_inference_v1"
QUALIFICATION_SCHEMA = "relation_adapter_v5p16_tpu_qualification_v1"
SMOKE_SCHEMA = "relation_adapter_bench_v2_v5p16_smoke_v1"
SOURCE_COMMIT = "0ebdebcf5697f5881b5bf3d2fd58d2938733a4e7"
FLUX_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
ADAPTER_REVISION = "7f65ecd6c7718c6de92782dade849a4872aa986a"
SIGLIP_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
ASSET_MANIFESTS = {
    "FLUX.1-dev": "5d974c9898c86e35f0e786b7551a0d2914b37603beb8a940c3b49b96a9a0b892",
    "RelationAdapter": "3d00a444eab4964011a19212c9a5b53cd6bad7413fbd663c4ae2ec60e2e28621",
    "siglip-so400m-patch14-384": "bd890cec873e059c1989854386b388c1c12db4e66bd11a021af8a4147b9535ff",
}
PROMPT = "Apply the demonstrated image transformation."
FROZEN_P = 1
NATIVE_P = 1
EPISODE_SEED = 42
MODEL_SEED = 1000
STEPS = 24
GUIDANCE = 3.5
ALPHA = 1.0
IMAGE_SIZE = 256
READOUT = "RelationAdapter@native_p1"
LICENSE_ACK = "FLUX.1-dev-NC-1.1.1"
AUTHORIZED_TPU = "kmh-tpuvm-v5p-16-kaiminghe-fa0dfd"
EXPECTED_ACCELERATOR = "v5p-16"
EXPECTED_TOPOLOGY = "2x2x2"
EXPECTED_GLOBAL_DEVICES = 8
EXPECTED_ADDRESSABLE_DEVICES = 4
EXPECTED_WORKERS = 2


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_png(path, pixels):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(np.asarray(pixels, np.uint8), "RGB").save(temp, format="PNG", optimize=False)
    os.replace(temp, path)


def descriptor(path, root):
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"not PNG: {path}")
        pixels = np.asarray(image.convert("RGB"), np.uint8)
        shape = [image.height, image.width, 3]
    return {"path": str(path.relative_to(root)), "sha256": sha256_file(path),
            "pixels_sha256": sha256_array(pixels), "shape": shape, "dtype": "uint8", "mode": "RGB"}


def task_rows(pack, task):
    return [i for i, row in enumerate(pack["records"]) if row["task"] == task]


def pack_slug(rel):
    return rel.replace("/", "__")


def episode_sha(demo_inputs, demo_targets, query_input, query_target):
    h = hashlib.sha256()
    for tile in (*demo_inputs, *demo_targets, query_input, query_target):
        h.update(memoryview(np.ascontiguousarray(tile)).cast("B"))
    return h.hexdigest()


def verify_license_ack(path: Path) -> None:
    if not path.is_file() or path.read_text().strip() != LICENSE_ACK:
        raise SystemExit(f"FLUX license acknowledgement missing; {path} must contain exactly {LICENSE_ACK!r}")


def parse_sha256s(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("* ")
        if len(digest) != 64 or name in rows or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"invalid asset manifest row: {line!r}")
        rows[name] = digest
    return rows


def verify_asset_tree(root: Path, expected_manifest_sha: str) -> dict:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file() or sha256_file(manifest) != expected_manifest_sha:
        raise SystemExit(f"asset tree manifest differs: {root}")
    expected = parse_sha256s(manifest)
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"}
    if actual != set(expected):
        raise SystemExit(f"asset file set differs: {root}; missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    for name, digest in expected.items():
        if sha256_file(root / name) != digest:
            raise SystemExit(f"asset hash differs: {root / name}")
    return {"manifest_sha256": expected_manifest_sha, "files": len(expected),
            "bytes": sum((root / name).stat().st_size for name in expected)}


def source_hashes(source_dir):
    files = {
        "runner": Path(__file__).resolve(), "defaults": Path(__file__).with_name("defaults.json"),
        "runtime_v2": LIVE_REPO / "tools/bench_v2_runtime_v2.py",
        "episode_builder": LIVE_REPO / "jflux2/analysis/capacity_ladder_v9.py",
        "episode_contract": LIVE_REPO / "jflux2/analysis/t16_offline_eval_v1.py",
        "episode_replay": LIVE_REPO / "tools/bench_episode_v1.py",
        "conventions": LIVE_REPO / "jflux2/dataloaders/multi_task_loader_v4.py",
        "official_inference": source_dir / "infer_single.py",
        "official_pipeline": source_dir / "src/pipeline_pe_clone.py",
        "official_transformer": source_dir / "transformer_flux.py",
        "official_attention": source_dir / "attention_processor.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def phase_rels(manifest: dict, phase: str) -> list[str]:
    tests = sorted(row["rel"] for row in manifest["packs"]
                   if str(row["rel"]).startswith("bench_v2_test/"))
    if phase == "val":
        return ["bench_v2_val"]
    if phase == "test":
        return ["bench_v2_val", *tests]
    raise ValueError(f"unknown plan phase: {phase}")


def validation_binding(phase: str, path: Path | None, expected_sha256: str) -> dict | None:
    """Authenticate TEST's VAL receipt before any TEST pack can be staged."""
    if phase == "val":
        if path is not None or expected_sha256:
            raise SystemExit("VAL planning must not consume a validation receipt")
        return None
    if path is None or not expected_sha256:
        raise SystemExit("TEST planning requires a hash-pinned validation receipt")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise SystemExit("TEST planning validation receipt SHA256 differs")
    receipt = json.loads(path.read_text())
    expected = {
        "schema": "relation_adapter_bench_v2_validation_receipt_v1",
        "status": "authoritative_corrected_v3_validation_complete",
        "episodes": 221,
        "metric_protocol": "bench_v2_PP2_CSA_seed1_AP_FIRST_v1",
        "release_sha256": "bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SystemExit("TEST planning validation receipt contract differs")
    for key in ("plan_identity", "qualification_receipt_sha256",
                "validation_inference_marker_sha256"):
        if not isinstance(receipt.get(key), str) or len(receipt[key]) != 64:
            raise SystemExit(f"TEST planning validation receipt lacks {key}")
    return receipt


def prepare_plan(args):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(LIVE_REPO))
    from tools.bench_v2_runtime_v2 import stage_pack
    from jflux2.analysis import ttt_ladder_v1 as _side_effect  # noqa: F401
    from jflux2.analysis import capacity_ladder_v9 as CL9
    if sha256_file(TRUSTED_MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("trusted Bench v2 manifest changed")
    commit = subprocess.check_output(["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise SystemExit("official source commit differs")
    receipt = validation_binding(args.phase, args.validation_receipt,
                                 args.validation_receipt_sha256)
    manifest = json.loads(TRUSTED_MANIFEST.read_text())
    rels = phase_rels(manifest, args.phase)
    expected_packs, expected_episodes = ((1, 221) if args.phase == "val" else (27, 17_732))
    if len(rels) != expected_packs:
        raise SystemExit("phase pack coverage differs from Bench v2")
    plan = {
        "schema": PLAN_SCHEMA, "phase": args.phase, "protocol": SCHEMA,
        "trusted_manifest": str(TRUSTED_MANIFEST),
        "trusted_manifest_file_sha256": MANIFEST_SHA256, "pack_uri_root": PACK_URI_ROOT,
        "source_commit": SOURCE_COMMIT, "flux_revision": FLUX_REVISION,
        "adapter_revision": ADAPTER_REVISION, "siglip_revision": SIGLIP_REVISION,
        "episode_seed": EPISODE_SEED, "frozen_p": FROZEN_P, "native_p": NATIVE_P,
        "adapter": "direct_same_source_native_p1", "condition": "real",
        "validation_plan_identity": None if receipt is None else receipt["plan_identity"],
        "validation_receipt_sha256": None if receipt is None else args.validation_receipt_sha256,
        "bound_qualification_receipt_sha256": None if receipt is None else receipt["qualification_receipt_sha256"],
        "prompt": PROMPT, "prompt_status": "benchmark-neutral adaptation; not official default",
        "dtype": "bfloat16", "steps": STEPS, "guidance_scale": GUIDANCE,
        "relation_scale_alpha": ALPHA, "model_seed": MODEL_SEED, "readout": READOUT,
        "backend_contract": "PyTorch/XLA v5p-16 two-worker SPMD-FSDP; no CUDA parity claim",
        "required_accelerator": EXPECTED_ACCELERATOR, "required_topology": EXPECTED_TOPOLOGY,
        "required_global_devices": EXPECTED_GLOBAL_DEVICES,
        "required_addressable_devices": EXPECTED_ADDRESSABLE_DEVICES,
        "required_workers": EXPECTED_WORKERS,
        "source_sha256": source_hashes(args.source_dir), "packs": [], "episodes": [],
    }
    for rel in rels:
        pack = stage_pack(rel, uri_root=PACK_URI_ROOT, manifest_path=TRUSTED_MANIFEST,
                          manifest_sha256=MANIFEST_SHA256, local_root=args.pack_cache)
        tasks = list(dict.fromkeys(row["task"] for row in pack["records"]))
        start, keys = len(plan["episodes"]), []
        for task in tasks:
            rows = task_rows(pack, task)
            images, masks = pack["images"][rows], pack["masks"][rows]
            records = [pack["records"][i] for i in rows]
            sources = [str(row.get("source")) for row in records]
            for q in range(len(rows)):
                built = CL9.build_episode(images, masks, sources, task, q, EPISODE_SEED, FROZEN_P)
                if int(built["p0"]) != FROZEN_P:
                    raise SystemExit(f"{rel} {task} q{q}: insufficient same-source demonstrations")
                demo_local = [int(i) for i in built["demo_indices"]]
                demo_global = [int(rows[i]) for i in demo_local]
                demo_inputs = [np.asarray(images[i], np.uint8) for i in demo_local]
                demo_targets = [np.rint(np.transpose(pair[1], (1, 2, 0)) * 255).astype(np.uint8) for pair in built["real"]]
                query_input, query_target = np.asarray(images[q], np.uint8), np.asarray(built["query_t"], np.uint8)
                key = f"{task}:{q}"
                keys.append(key)
                plan["episodes"].append({
                    "episode_id": f"{pack_slug(rel)}__{task}__q{q:05d}__P1__real",
                    "key": key, "pack_rel": rel, "pack_slug": pack_slug(rel), "task": task, "q": q,
                    "frozen_p": FROZEN_P, "native_p": NATIVE_P, "cid": int(built["cid"]),
                    "query_global": int(rows[q]), "query_record": str(records[q]["record_id"]),
                    "demo_global": demo_global, "demo_indices": demo_local,
                    "demo_records": [str(records[i]["record_id"]) for i in demo_local],
                    "native_demo_global": demo_global[0], "native_demo_index": demo_local[0],
                    "native_demo_record": str(records[demo_local[0]]["record_id"]),
                    "episode_pixels_sha256": episode_sha(demo_inputs, demo_targets, query_input, query_target),
                    "query_input_pixels_sha256": sha256_array(query_input),
                    "query_target_pixels_sha256": sha256_array(query_target),
                })
        count = len(plan["episodes"]) - start
        plan["packs"].append({
            "rel": rel, "slug": pack_slug(rel), "local_dir": pack["staged_copy"]["local_dir"],
            "data_identity": pack["data_identity"], "data_identity_sha256": pack["data_identity_sha256"],
            "tasks": tasks, "expected_episode_keys": keys, "n_episodes": count,
            "plan_range": [start, len(plan["episodes"])],
        })
        print(f"[plan] {rel}: {count}", flush=True)
        del pack
        gc.collect()
    if (len(plan["episodes"]) != expected_episodes
            or plan["packs"][0]["n_episodes"] != 221):
        raise SystemExit("Bench v2 phase plan counts differ")
    plan["plan_identity"] = identity(plan)
    atomic_json(args.out, plan)
    print(f"[plan] identity={plan['plan_identity']} phase={args.phase} "
          f"episodes={expected_episodes} packs={expected_packs}", flush=True)


class PackCache:
    def __init__(self, plan):
        self.info = {row["rel"]: row for row in plan["packs"]}
        self.rel = None
        self.images = self.masks = None

    def load(self, rel):
        if rel == self.rel:
            return self.images, self.masks
        with np.load(Path(self.info[rel]["local_dir"]) / "samples.npz", allow_pickle=False) as archive:
            self.images, self.masks = np.array(archive["images"], copy=True), np.array(archive["masks"], copy=True)
        self.images.setflags(write=False); self.masks.setflags(write=False); self.rel = rel
        return self.images, self.masks


def reconstruct(ep, packs):
    sys.path.insert(0, str(LIVE_REPO))
    from tools import bench_episode_v1 as BE
    images, masks = packs.load(ep["pack_rel"])
    demo_inputs = [np.asarray(images[i], np.uint8) for i in ep["demo_global"]]
    coded = BE.code_episode([masks[i] for i in ep["demo_global"]] + [masks[ep["query_global"]]], ep["task"], ep["cid"])
    query_input, query_target = np.asarray(images[ep["query_global"]], np.uint8), coded[-1]
    if episode_sha(demo_inputs, coded[:-1], query_input, query_target) != ep["episode_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: reconstructed frozen episode differs")
    return demo_inputs, coded[:-1], query_input, query_target


def resize_demo(image: Image.Image) -> Image.Image:
    """Byte-for-byte policy from official infer_single.resize_img(..., pad_to_regular=True)."""
    image = image.convert("RGB")
    w, h = image.size
    ratios = ((3, 4), (4, 3), (1, 1), (16, 9), (9, 16))
    rw, rh = min(ratios, key=lambda r: abs(r[0] / r[1] - w / h))
    if w / h >= rw / rh:
        target_w, target_h = w, int(w * rh / rw)
    else:
        target_h, target_w = h, int(h * rw / rh)
    padded = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    padded.paste(image, ((target_w - w) // 2, (target_h - h) // 2))
    scale = 512.0 / max(target_w, target_h)
    return padded.resize((round(target_w * scale), round(target_h * scale)), Image.Resampling.BILINEAR)


def recursive_shard_output(output, mesh):
    import torch
    import torch_xla.distributed.spmd as xs
    def visit(value):
        if isinstance(value, torch.Tensor) and value.ndim:
            spec = [None] * value.ndim
            axes = [axis for axis, size in enumerate(value.shape) if axis > 0 and size % mesh.size() == 0]
            axis = axes[0] if axes else next((axis for axis, size in enumerate(value.shape) if size % mesh.size() == 0), None)
            if axis is not None:
                spec[axis] = "fsdp"; xs.mark_sharding(value, mesh, tuple(spec))
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif hasattr(value, "to_tuple"):
            visit(value.to_tuple())
    visit(output)


class MLPProjModel:
    @staticmethod
    def build(torch, cross_attention_dim, num_tokens):
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Sequential(torch.nn.Linear(1152, 2304), torch.nn.GELU(),
                                                torch.nn.Linear(2304, cross_attention_dim * num_tokens))
                self.norm = torch.nn.LayerNorm(cross_attention_dim)
            def forward(self, embeds):
                return self.norm(self.proj(embeds).reshape(-1, num_tokens, cross_attention_dim))
        return Projection()


def load_runtime(args, device, mesh, task_prompts=None):
    import torch
    from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    from transformers import AutoProcessor, SiglipVisionModel
    sys.path.insert(0, str(args.source_dir.resolve()))
    from attention_processor import IPAFluxAttnProcessor2_0
    from src.pipeline_pe_clone import FluxPipeline
    from transformer_flux import FluxTransformer2DModel
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(args.flux_dir, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    pipe = FluxPipeline.from_pretrained(args.flux_dir, transformer=transformer, torch_dtype=dtype, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(PROMPT, PROMPT, device=torch.device("cpu"), max_sequence_length=512)
    task_embeddings = {}
    with torch.no_grad():
        for task, text in (task_prompts or {}).items():
            pe, pp, _ = pipe.encode_prompt(text, text, device=torch.device("cpu"), max_sequence_length=512)
            task_embeddings[task] = (pe.to(device, dtype=dtype), pp.to(device, dtype=dtype))
    pipe.load_lora_weights(args.adapter_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora(); pipe.unload_lora_weights()
    processors = {}
    for name in transformer.attn_processors:
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            processors[name] = IPAFluxAttnProcessor2_0(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim, num_tokens=128, scale=ALPHA).to(dtype=dtype)
        else:
            processors[name] = transformer.attn_processors[name]
    transformer.set_attn_processor(processors)
    state = torch.load(args.adapter_dir / "ip_adapter-100000.bin", map_location="cpu", weights_only=True, mmap=True)
    projection_state = state["image_proj"]
    if next(iter(projection_state)).startswith("module."):
        projection_state = {k.removeprefix("module."): v for k, v in projection_state.items()}
    projection = MLPProjModel.build(torch, transformer.config.joint_attention_dim, 128).to(dtype=dtype)
    projection.load_state_dict(projection_state, strict=True)
    adapter_state = state["ip_adapter"]
    if next(iter(adapter_state)).startswith("module."):
        adapter_state = {k.removeprefix("module."): v for k, v in adapter_state.items()}
    layers = torch.nn.ModuleList(transformer.attn_processors.values())
    missing, unexpected = layers.load_state_dict(adapter_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"adapter state mismatch missing={missing} unexpected={unexpected}")
    image_encoder = SiglipVisionModel.from_pretrained(args.siglip_dir, torch_dtype=dtype, local_files_only=True).eval()
    image_processor = AutoProcessor.from_pretrained(args.siglip_dir, local_files_only=True)
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    del state, projection_state, adapter_state, layers
    gc.collect()
    auto_wrap = functools.partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={FluxTransformerBlock, FluxSingleTransformerBlock})
    transformer = SpmdFullyShardedDataParallel(transformer.eval(), mesh=mesh, auto_wrap_policy=auto_wrap,
                                                shard_output=recursive_shard_output)
    pipe.transformer = transformer
    pipe.vae = pipe.vae.eval().to(device)
    return {"task_embeddings": task_embeddings, "pipe": pipe, "image_encoder": image_encoder.to(device), "image_processor": image_processor,
            "projection": projection.eval().to(device), "prompt_embeds": prompt_embeds.to(device, dtype=dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(device, dtype=dtype), "dtype": dtype}


def relation_embedding(runtime, before, after, device):
    import torch
    import torch_xla.core.xla_model as xm
    values = []
    for image in (before, after):
        pixels = runtime["image_processor"](images=resize_demo(image), return_tensors="pt").pixel_values.to(device=device, dtype=runtime["dtype"])
        with torch.no_grad():
            values.append(runtime["projection"](runtime["image_encoder"](pixels).pooler_output))
        xm.mark_step()
    return torch.cat(values, dim=0)


def predict(runtime, before, after, query, device):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    relation = relation_embedding(runtime, Image.fromarray(before, "RGB"), Image.fromarray(after, "RGB"), device)
    torch_xla.manual_seed(MODEL_SEED, device=device)
    with torch.no_grad():
        result = runtime["pipe"](
            prompt=None, prompt_embeds=runtime["prompt_embeds"], pooled_prompt_embeds=runtime["pooled_prompt_embeds"],
            image_emb=relation, condition_image=Image.fromarray(query, "RGB"), height=IMAGE_SIZE, width=IMAGE_SIZE,
            guidance_scale=GUIDANCE, num_inference_steps=STEPS, generator=None, output_type="pt").images
    xm.mark_step()
    return (result[0].float().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()


def validate_plan(path):
    plan = json.loads(Path(path).read_text())
    claimed = plan.pop("plan_identity")
    if identity(plan) != claimed:
        raise SystemExit("plan identity differs")
    plan["plan_identity"] = claimed
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("frozen_p") != FROZEN_P
            or plan.get("native_p") != NATIVE_P or plan.get("prompt") != PROMPT or plan.get("condition") != "real"):
        raise SystemExit("plan is outside the frozen RelationAdapter contract")
    phase = plan.get("phase")
    expected = {"val": (1, 221), "test": (27, 17_732)}.get(phase)
    if expected is None or len(plan["packs"]) != expected[0] or len(plan["episodes"]) != expected[1] or plan["packs"][0]["n_episodes"] != 221:
        raise SystemExit("plan counts differ")
    if phase == "test":
        for key in ("validation_plan_identity", "validation_receipt_sha256",
                    "bound_qualification_receipt_sha256"):
            if not isinstance(plan.get(key), str) or len(plan[key]) != 64:
                raise SystemExit(f"TEST plan lacks {key}")
    elif any(plan.get(key) is not None for key in
             ("validation_plan_identity", "validation_receipt_sha256",
              "bound_qualification_receipt_sha256")):
        raise SystemExit("VAL plan contains TEST-only receipt binding")
    for ep in plan["episodes"]:
        if len(ep["demo_global"]) != FROZEN_P or ep["native_demo_global"] != ep["demo_global"][0]:
            raise SystemExit(f"{ep['episode_id']}: direct-P1 binding differs")
    return plan


def validation_plan_identity(plan: dict) -> str:
    return plan.get("validation_plan_identity") or plan["plan_identity"]


def validate_qualification_receipt(path: Path, expected_sha256: str, plan: dict) -> dict:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise SystemExit("TPU qualification receipt is absent or its pinned SHA256 differs")
    receipt = json.loads(path.read_text())
    expected = {
        "schema": QUALIFICATION_SCHEMA,
        "accepted": True,
        "plan_identity": validation_plan_identity(plan),
        "repeat_pixel_identical": True,
        "tpu_native_no_cuda_parity_accepted": True,
        "qualified_devices": EXPECTED_GLOBAL_DEVICES,
        "qualified_addressable_devices": EXPECTED_ADDRESSABLE_DEVICES,
        "qualified_workers": EXPECTED_WORKERS,
        "topology": EXPECTED_TOPOLOGY,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SystemExit("TPU qualification receipt contract differs")
    return receipt


def validate_validation_receipt(path: Path, expected_sha256: str, plan: dict,
                                qualification_receipt_sha256: str) -> dict:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise SystemExit("validation scoring receipt is absent or its pinned SHA256 differs")
    receipt = json.loads(path.read_text())
    expected = {
        "schema": "relation_adapter_bench_v2_validation_receipt_v1",
        "status": "authoritative_corrected_v3_validation_complete",
        "plan_identity": validation_plan_identity(plan),
        "episodes": 221,
        "metric_protocol": "bench_v2_PP2_CSA_seed1_AP_FIRST_v1",
        "release_sha256": "bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792",
        "qualification_receipt_sha256": qualification_receipt_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SystemExit("validation scoring receipt contract differs")
    return receipt


def initial_view(pack, plan):
    view_plan_identity = (validation_plan_identity(plan)
                          if pack["rel"] == "bench_v2_val" and plan.get("phase") == "test"
                          else plan["plan_identity"])
    return {"schema": SCHEMA, "protocol": SCHEMA, "pack_rel": pack["rel"], "frozen_p": FROZEN_P,
            "native_p": NATIVE_P, "adapter": "direct same-source P=1 episode", "condition": "real",
            "plan_identity": view_plan_identity, "data_identity": pack["data_identity"],
            "data_identity_sha256": pack["data_identity_sha256"],
            "expected_episode_keys": pack["expected_episode_keys"], "source_sha256": plan["source_sha256"],
            "episodes": {}, "complete": False}


def load_views(results_root, plan):
    views, done = {}, set()
    for pack in plan["packs"]:
        root, path = results_root / "views" / pack["slug"], results_root / "views" / pack["slug"] / "results.json"
        view = json.loads(path.read_text()) if path.exists() else initial_view(pack, plan)
        allowed_plan_identity = plan["plan_identity"]
        if pack["rel"] == "bench_v2_val" and plan.get("phase") == "test":
            allowed_plan_identity = validation_plan_identity(plan)
        if view.get("plan_identity") != allowed_plan_identity or view.get("data_identity") != pack["data_identity"]:
            raise SystemExit(f"{path}: resume identity differs")
        for key, ep in list(view["episodes"].items()):
            artifacts = ep.get("artifacts", {})
            needed = [artifacts.get("query_input"), artifacts.get("query_target"), artifacts.get("prediction")]
            if any(not desc or not (root / desc["path"]).is_file() for desc in needed):
                del view["episodes"][key]; view["complete"] = False; continue
            for desc in needed:
                if sha256_file(root / desc["path"]) != desc["sha256"]:
                    raise SystemExit(f"{root / desc['path']}: resume artifact differs")
            done.add(f"{pack['rel']}|{key}")
        atomic_json(path, view); views[pack["rel"]] = view
    return views, done


def save_episode(results_root, pack, view, ep, query_input, query_target, prediction, seconds):
    root = results_root / "views" / pack["slug"]
    episode_dir = root / f"{ep['task']}_q{ep['q']}"
    input_path, target_path, pred_path = episode_dir / "query_input.png", episode_dir / "query_target.png", episode_dir / "prediction256.png"
    atomic_png(input_path, query_input); atomic_png(target_path, query_target); atomic_png(pred_path, prediction)
    view["episodes"][ep["key"]] = {
        "task": ep["task"], "q": ep["q"], "frozen_p": FROZEN_P, "native_p": NATIVE_P, "cid": ep["cid"],
        "query_record": ep["query_record"], "demo_records": ep["demo_records"],
        "native_demo_record": ep["native_demo_record"], "prediction_complete": True, "complete": True,
        "readouts": {READOUT: {"real": {
            "prompt": PROMPT, "prompt_status": "benchmark-neutral adaptation; not official default",
            "model_seed": MODEL_SEED, "steps": STEPS, "guidance_scale": GUIDANCE,
            "relation_scale_alpha": ALPHA, "dtype": "bfloat16", "native_prompt_count": NATIVE_P,
            "frozen_episode_p": FROZEN_P, "demo_selection": "sole direct-P1 demonstration", "ensemble": False,
            "backend": "PyTorch/XLA v5p-16 two-worker SPMD-FSDP port", "cuda_parity_claimed": False,
            "seconds": seconds, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }}},
        "artifacts": {"query_input": descriptor(input_path, root), "query_target": descriptor(target_path, root),
                      "prediction": descriptor(pred_path, root),
                      "readouts": {READOUT: {"real": descriptor(pred_path, root)}}},
    }


def infer(args):
    verify_license_ack(args.license_ack)
    plan = validate_plan(args.plan)
    expected_plan_phase = "test" if args.phase == "test" else "val"
    if plan["phase"] != expected_plan_phase:
        raise SystemExit(f"{args.phase} requires a {expected_plan_phase} plan")
    if sha256_file(TRUSTED_MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("trusted manifest changed")
    if subprocess.check_output(["git", "-C", str(args.source_dir), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT:
        raise SystemExit("official source commit changed")
    asset_evidence = {
        "flux": verify_asset_tree(args.flux_dir, ASSET_MANIFESTS["FLUX.1-dev"]),
        "adapter": verify_asset_tree(args.adapter_dir, ASSET_MANIFESTS["RelationAdapter"]),
        "siglip": verify_asset_tree(args.siglip_dir, ASSET_MANIFESTS["siglip-so400m-patch14-384"]),
    }
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU" or not list(Path("/dev/vfio").glob("*")):
        raise SystemExit("refusing non-TPU execution")
    if args.expected_tpu != AUTHORIZED_TPU or args.expected_zone != "us-east5-a":
        raise SystemExit("RelationAdapter v5p16_q1 is bound to its authorized east5 TPU")
    if args.expected_devices != EXPECTED_GLOBAL_DEVICES:
        raise SystemExit("RelationAdapter v5p16_q1 requires its exact 16-device topology")
    if args.phase in ("val", "test"):
        if not args.qualification_receipt or not args.qualification_receipt_sha256:
            raise SystemExit("VAL/TEST requires a hash-pinned TPU qualification receipt")
        validate_qualification_receipt(args.qualification_receipt, args.qualification_receipt_sha256, plan)
    validation_gate = None
    if args.phase == "test":
        if not args.validation_receipt or not args.validation_receipt_sha256:
            raise SystemExit("TEST requires a hash-pinned corrected-V3 validation receipt")
        if (args.validation_receipt_sha256 != plan["validation_receipt_sha256"]
                or args.qualification_receipt_sha256 != plan["bound_qualification_receipt_sha256"]):
            raise SystemExit("TEST receipts differ from plan-time receipt binding")
        validation_gate = validate_validation_receipt(
            args.validation_receipt, args.validation_receipt_sha256, plan,
            args.qualification_receipt_sha256)
        local_val_marker = args.results_root / "VAL_INFERENCE_COMPLETE.json"
        if (not local_val_marker.is_file()
                or sha256_file(local_val_marker) != validation_gate.get("validation_inference_marker_sha256")):
            raise SystemExit("TEST validation receipt is not bound to the local VAL terminal marker")
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    count = int(xr.global_runtime_device_count())
    addressable = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if (backend.upper() != "TPU" or count != args.expected_devices
            or addressable != EXPECTED_ADDRESSABLE_DEVICES):
        raise SystemExit(f"refusing backend={backend} global={count} addressable={addressable}")
    mesh = xs.Mesh(np.arange(count), (count,), ("fsdp",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {
        "backend": backend, "device": str(device), "runtime_device_count": count,
        "addressable_runtime_device_count": addressable, "hostname": socket.gethostname(),
        "expected_tpu": args.expected_tpu, "expected_zone": args.expected_zone,
        "asset_evidence": asset_evidence, "license_acknowledgement": LICENSE_ACK,
        "backend_label": "PyTorch/XLA v5p-16 two-worker SPMD-FSDP port; official CUDA numerical parity is not established",
        "cuda_parity_claimed": False, "phase": args.phase,
        "qualification_receipt_sha256": args.qualification_receipt_sha256,
        "validation_receipt_sha256": args.validation_receipt_sha256,
    }
    atomic_json(args.results_root / f"backend.{args.phase}.json", evidence)
    views, done = load_views(args.results_root, plan)
    val_pack = plan["packs"][0]
    val_keys = {f"{val_pack['rel']}|{key}" for key in val_pack["expected_episode_keys"]}
    if any(not key.startswith("bench_v2_val|") for key in done) and not val_keys <= done:
        raise SystemExit("resume contains TEST output before complete validation")
    if args.phase == "test" and (not val_keys <= done or not (args.results_root / "VAL_INFERENCE_COMPLETE.json").is_file()):
        raise SystemExit("TEST requires all 221 validation predictions and its terminal marker")
    runtime = load_runtime(args, device, mesh)
    packs, by_rel = PackCache(plan), {row["rel"]: row for row in plan["packs"]}
    if args.phase == "smoke":
        selected = plan["episodes"][:1]
    elif args.phase == "val":
        selected = [ep for ep in plan["episodes"] if ep["pack_rel"] == "bench_v2_val"]
    else:
        selected = [ep for ep in plan["episodes"] if ep["pack_rel"] != "bench_v2_val"]
    for ordinal, ep in enumerate(selected, 1):
        token = f"{ep['pack_rel']}|{ep['key']}"
        if token in done:
            continue
        demo_inputs, demo_targets, query_input, query_target = reconstruct(ep, packs)
        started = time.monotonic()
        prediction = predict(runtime, demo_inputs[0], demo_targets[0], query_input, device)
        if prediction.shape != (256, 256, 3) or prediction.dtype != np.uint8:
            raise SystemExit(f"{ep['episode_id']}: output contract differs")
        pack, view = by_rel[ep["pack_rel"]], views[ep["pack_rel"]]
        save_episode(args.results_root, pack, view, ep, query_input, query_target, prediction, time.monotonic() - started)
        view["complete"] = len(view["episodes"]) == pack["n_episodes"]
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
        done.add(token)
        print(f"[{ordinal}/{len(selected)}] {ep['episode_id']}", flush=True)
    if args.phase == "smoke":
        ep = selected[0]
        pred = views[ep["pack_rel"]]["episodes"][ep["key"]]["artifacts"]["prediction"]
        sampling = views[ep["pack_rel"]]["episodes"][ep["key"]]["readouts"][READOUT]["real"]
        try:
            raw_memory = xm.get_memory_info(device)
            memory = {str(key): int(value) for key, value in raw_memory.items()}
        except Exception as exc:  # diagnostic is mandatory for qualification
            memory = {"error": f"{type(exc).__name__}: {exc}"}
        atomic_json(args.results_root / "SMOKE_COMPLETE.json", {
            "schema": SMOKE_SCHEMA, "plan_identity": plan["plan_identity"],
            "episode_id": ep["episode_id"], "prediction_sha256": pred["sha256"],
            "prediction_pixels_sha256": pred["pixels_sha256"],
            "prediction_path": str(Path("views") / by_rel[ep["pack_rel"]]["slug"] / pred["path"]),
            "device_count": count, "addressable_device_count": addressable,
            "worker_count": EXPECTED_WORKERS, "topology": EXPECTED_TOPOLOGY,
            "accelerator": EXPECTED_ACCELERATOR,
            "backend": backend, "expected_tpu": args.expected_tpu,
            "memory_after_first_sample": memory,
            "first_compile_and_sampling_seconds": sampling["seconds"],
            "backend_label": "PyTorch/XLA v5p-16 two-worker SPMD-FSDP port; official CUDA numerical parity is not established",
            "cuda_parity_claimed": False,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return
    terminal_packs = plan["packs"][:1] if args.phase == "val" else plan["packs"][1:]
    for pack in terminal_packs:
        view = views[pack["rel"]]
        if set(view["episodes"]) != set(pack["expected_episode_keys"]):
            raise SystemExit(f"{pack['rel']}: terminal coverage differs")
        view["complete"] = True
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
    if args.phase == "val":
        atomic_json(args.results_root / "VAL_INFERENCE_COMPLETE.json", {
            "schema": "relation_adapter_bench_v2_validation_inference_complete_v1",
            "episodes": 221, "plan_identity": plan["plan_identity"],
            "qualification_receipt_sha256": args.qualification_receipt_sha256,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return
    atomic_json(args.results_root / "INFERENCE_COMPLETE.json", {
        "schema": "relation_adapter_bench_v2_native_p1_complete_v2", "packs": 27, "episodes": 17_732,
        "plan_identity": plan["plan_identity"], "qualification_receipt_sha256": args.qualification_receipt_sha256,
        "validation_receipt_sha256": args.validation_receipt_sha256,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("prepare-plan")
    plan.add_argument("--out", type=Path, required=True); plan.add_argument("--pack-cache", type=Path, required=True)
    plan.add_argument("--source-dir", type=Path, required=True)
    plan.add_argument("--phase", choices=("val", "test"), required=True)
    plan.add_argument("--validation-receipt", type=Path)
    plan.add_argument("--validation-receipt-sha256", default="")
    run = subs.add_parser("infer")
    for name in ("plan", "results_root", "source_dir", "flux_dir", "adapter_dir", "siglip_dir", "cache_dir", "license_ack"):
        run.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    run.add_argument("--expected-tpu", required=True); run.add_argument("--expected-zone", required=True)
    run.add_argument("--expected-devices", type=int, required=True)
    run.add_argument("--phase", choices=("smoke", "val", "test"), required=True)
    run.add_argument("--qualification-receipt", type=Path)
    run.add_argument("--qualification-receipt-sha256", default="")
    run.add_argument("--validation-receipt", type=Path)
    run.add_argument("--validation-receipt-sha256", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    return prepare_plan(args) if args.command == "prepare-plan" else infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
