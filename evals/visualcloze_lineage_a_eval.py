#!/usr/bin/env python3
"""Reproducible VisualCloze evaluation on the lineage-A validation pack.

The three subcommands deliberately use separate environments:

* ``prepare`` freezes the pack's native lineage-A targets without re-coding.
* ``infer`` runs in an isolated PyTorch/XLA environment on a v5e-8 and writes
  both the native 384px VisualCloze result and a 256px scoring copy.
* ``score`` applies the registered lineage-A task metrics to saved predictions.

The full run is 4 tasks x 32 queries x four conditions, with P=2. All arms use
the same deterministic episode draw and the same open-weight model revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path


LIVE_REPO = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
PACK_DIR = Path("/kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_packs/val_s32_px256_lineageA")
EXPECTED_PACK_ID = "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2"
MODEL_ID = "VisualCloze/VisualClozePipeline-384"
MODEL_REVISION = "59c469d2772d927ffe55f3543c4d3bd556fd46a4"
TASK_PROMPT = (
    "Each preceding row shows the same visual transformation from the left image "
    "to the right image. Apply that transformation to the left image in the final "
    "row and complete the missing right image."
)
CONTENT_PROMPT = ""
SOURCE_FILES = {
    "episode_builder": LIVE_REPO / "jflux2/analysis/t16_offline_eval_v1.py",
    "pack_loader": LIVE_REPO / "jflux2/analysis/ttt_ladder_v1.py",
    "scorer": LIVE_REPO / "jflux2/fasteval/metrics.py",
}


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o666)


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in SOURCE_FILES.items()}


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def prepare(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(LIVE_REPO))
    from jflux2.analysis import t16_offline_eval_v1 as T16  # noqa: PLC0415
    from jflux2.analysis.ttt_ladder_v1 import load_pack  # noqa: PLC0415
    pack = load_pack(str(args.pack))
    if str(pack.get("pack_id")) != EXPECTED_PACK_ID:
        raise SystemExit(
            f"pack identity mismatch: got {pack.get('pack_id')!r}, expected {EXPECTED_PACK_ID!r}"
        )
    if int(pack["images"].shape[0]) != 128 or pack["images"].dtype != np.uint8:
        raise SystemExit(f"unexpected pack arrays: {pack['images'].shape} {pack['images'].dtype}")
    if str(pack.get("supervision_version")) != "lvm-native" or str(pack.get("lineage")) != "A":
        raise SystemExit("refusing prepare: pack is not the pinned native lineage-A validation pack")

    out = args.out.resolve()
    episodes_dir = out / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o777)
    os.chmod(episodes_dir, 0o777)
    tasks = _csv(args.tasks)
    conditions = _csv(args.conditions)
    manifest: dict[str, object] = {
        "schema": "visualcloze_lineage_a_eval_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_dir": str(args.pack),
        "pack_id": str(pack["pack_id"]),
        "records": int(len(pack["records"])),
        "live_repo": str(LIVE_REPO),
        "live_repo_git_sha": _git_sha(LIVE_REPO),
        "source_files_sha256": _source_hashes(),
        "conventions": "lineage-A native targets; no episode re-coding",
        "demo_source_policy": "same",
        "seed_rule": "[seed, task_seed(task), q, P]",
        "seed": int(args.seed),
        "p": int(args.p),
        "task_prompt": TASK_PROMPT,
        "content_prompt": CONTENT_PROMPT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "episodes": [],
    }

    for task in tasks:
        global_indices = [i for i, r in enumerate(pack["records"]) if r["task"] == task]
        if not global_indices:
            raise SystemExit(f"task {task!r} is absent from pack")
        images = pack["images"][global_indices]
        masks = pack["masks"][global_indices]
        sources = [str(pack["records"][i].get("source")) for i in global_indices]
        same_source = {s: [j for j, x in enumerate(sources) if x == s] for s in set(sources)}

        for q in range(min(int(args.queries), len(global_indices))):
            rng = np.random.default_rng([args.seed, T16.task_seed(task), q, args.p])
            convention_id = 0
            demo_indices = T16.draw_demo_indices(
                len(images), q, args.p, rng, candidates=same_source[sources[q]]
            )
            target = np.asarray(masks[q], np.uint8)
            built = {}
            demo_targets = [np.asarray(masks[j], np.uint8) for j in demo_indices]
            if "real" in conditions:
                built["real"] = [
                    (T16.to_chw01(images[j]), T16.to_chw01(t))
                    for j, t in zip(demo_indices, demo_targets)
                ]
            if "scrambled" in conditions:
                if args.p < 2:
                    raise SystemExit("lineage-A scrambled control requires P>=2")
                shuffled = T16.derange(demo_indices, rng)
                by_demo = {int(j): t for j, t in zip(demo_indices, demo_targets)}
                built["scrambled"] = [
                    (T16.to_chw01(images[j]), T16.to_chw01(by_demo[int(k)]))
                    for j, k in zip(demo_indices, shuffled)
                ]
            if "grey" in conditions:
                grey = np.full_like(images[0], 128)
                built["grey"] = [(T16.to_chw01(grey), T16.to_chw01(grey))] * args.p
            if "oracle" in conditions:
                built["oracle"] = [
                    (T16.to_chw01(images[q]), T16.to_chw01(target))
                ] * args.p
            record = pack["records"][global_indices[q]]
            for condition in conditions:
                pairs = built[condition]
                demo_inputs = np.stack(
                    [np.rint(np.transpose(a, (1, 2, 0)) * 255).astype(np.uint8) for a, _ in pairs]
                )
                demo_targets = np.stack(
                    [np.rint(np.transpose(b, (1, 2, 0)) * 255).astype(np.uint8) for _, b in pairs]
                )
                episode_id = f"{task}__q{q:02d}__P{args.p}__{condition}"
                episode_file = episodes_dir / f"{episode_id}.npz"
                np.savez_compressed(
                    episode_file,
                    demo_inputs=demo_inputs,
                    demo_targets=demo_targets,
                    query_input=np.asarray(images[q], np.uint8),
                    query_target=np.asarray(target, np.uint8),
                )
                os.chmod(episode_file, 0o666)
                if not args.no_previews:
                    preview = Image.new("RGB", (512, 256 * (args.p + 1)), (0, 0, 0))
                    for row in range(args.p):
                        preview.paste(Image.fromarray(demo_inputs[row]), (0, row * 256))
                        preview.paste(Image.fromarray(demo_targets[row]), (256, row * 256))
                    preview.paste(Image.fromarray(images[q]), (0, args.p * 256))
                    preview_path = episodes_dir / f"{episode_id}__context.png"
                    preview.save(preview_path)
                    os.chmod(preview_path, 0o666)
                manifest["episodes"].append(
                    {
                        "episode_id": episode_id,
                        "file": str(episode_file.relative_to(out)),
                        "sha256": _sha256(episode_file),
                        "task": task,
                        "q": q,
                        "p": int(args.p),
                        "condition": condition,
                        "source": sources[q],
                        "record_id": str(record.get("record_id")),
                        "demo_indices": [int(x) for x in demo_indices],
                        "demo_record_ids": [
                            str(pack["records"][global_indices[int(x)]].get("record_id"))
                            for x in demo_indices
                        ],
                        "convention_id": convention_id,
                        "palette_task": False,
                        "native_lineage_a_target": True,
                    }
                )
        print(f"prepared {task}: {len(global_indices)} queries x {len(conditions)} conditions", flush=True)

    _json_dump(out / "manifest.json", manifest)
    print(f"prepared {len(manifest['episodes'])} episodes in {out}", flush=True)
    return 0


def _pil_grid(npz_path: Path):
    import numpy as np
    from PIL import Image

    data = np.load(npz_path)
    rows = [
        [Image.fromarray(a), Image.fromarray(b)]
        for a, b in zip(data["demo_inputs"], data["demo_targets"])
    ]
    rows.append([Image.fromarray(data["query_input"]), None])
    return rows


def infer(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from PIL import Image
    from torch_xla.experimental.custom_kernel import FlashAttention

    from diffusers import VisualClozeGenerationPipeline

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pack_id") != EXPECTED_PACK_ID:
        raise SystemExit("refusing inference: manifest pack_id is not the pinned v5 pack")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise SystemExit("refusing inference: manifest model revision changed")

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    compiler_cache = cache_dir / "xla_compile_cache"
    compiler_cache.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(compiler_cache), readonly=False)
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_devices != 8:
        raise SystemExit(f"this gate requires the named v5e-8; XLA reports {num_devices} devices")
    mesh = xs.Mesh(np.arange(num_devices), (num_devices // 4, 4), ("data", "model"))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    print(f"XLA device={device}; mesh={mesh.mesh_shape}; loading {MODEL_ID}@{MODEL_REVISION}", flush=True)

    episodes = list(manifest["episodes"])
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("manifest contains no episodes")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.flash_block_size < 128 or args.flash_block_size % 128:
        raise SystemExit("--flash-block-size must be a positive multiple of 128")
    if len(episodes) % args.batch_size:
        raise SystemExit(
            f"episode count {len(episodes)} is not divisible by batch size {args.batch_size}; "
            "use a divisible --limit for shape-stable TPU inference"
        )
    first_batch = episodes[: args.batch_size]
    first_grids = [_pil_grid(run_dir / ep["file"]) for ep in first_batch]
    first_task_prompts = [TASK_PROMPT] * args.batch_size
    first_content_prompts = [CONTENT_PROMPT] * args.batch_size
    text_pipe = VisualClozeGenerationPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        transformer=None,
        vae=None,
        torch_dtype=torch.bfloat16,
        cache_dir=str(cache_dir / "hf"),
        local_files_only=args.local_files_only,
        resolution=384,
    ).to("cpu")
    processed = text_pipe.image_processor.preprocess(
        first_task_prompts,
        first_content_prompts,
        first_grids,
        vae_scale_factor=text_pipe.vae_scale_factor,
    )
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = text_pipe.encode_prompt(
            layout_prompt=processed["layout_prompt"],
            task_prompt=processed["task_prompt"],
            content_prompt=processed["content_prompt"],
            device=torch.device("cpu"),
            max_sequence_length=512,
        )
    del processed, text_pipe

    pipe = VisualClozeGenerationPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        torch_dtype=torch.bfloat16,
        cache_dir=str(cache_dir / "hf"),
        local_files_only=args.local_files_only,
        resolution=384,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    for _, param in pipe.transformer.named_parameters():
        if param.dim() >= 2:
            spec = [None] * param.dim()
            largest = max(range(param.dim()), key=lambda d: param.shape[d])
            spec[largest] = "model"
            xs.mark_sharding(param, mesh, tuple(spec))

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

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    text_ids = text_ids.to(device)
    xs.mark_sharding(prompt_embeds, mesh, ("data", None, None))
    xs.mark_sharding(pooled_prompt_embeds, mesh, ("data", None))

    def fixed_encode_prompt(_self, *unused_args, **unused_kwargs):
        return prompt_embeds, pooled_prompt_embeds, text_ids

    pipe.encode_prompt = types.MethodType(fixed_encode_prompt, pipe)
    predictions = run_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    os.chmod(predictions, 0o777)
    partial_path = run_dir / "inference.partial.json"
    final_path = run_dir / "inference.json"
    resume_path = partial_path if partial_path.exists() else final_path
    if args.resume and resume_path.exists():
        infer_log = json.loads(resume_path.read_text())
        expected = {
            "model_revision": MODEL_REVISION,
            "num_inference_steps": int(args.steps),
            "guidance_scale": float(args.guidance),
            "seed": int(args.model_seed),
            "batch_size": int(args.batch_size),
            "flash_block_size": int(args.flash_block_size),
        }
        mismatched = {
            k: (infer_log.get(k, 1536 if k == "flash_block_size" else None), v)
            for k, v in expected.items()
            if infer_log.get(k, 1536 if k == "flash_block_size" else None) != v
        }
        if mismatched:
            raise SystemExit(f"resume configuration mismatch: {mismatched}")
        infer_log.pop("finished_utc", None)
        print(f"resuming with {len(infer_log['results'])}/{len(episodes)} episodes complete", flush=True)
    else:
        infer_log = {
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": os.uname().nodename,
            "torch": torch.__version__,
            "torch_xla_devices": int(num_devices),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "resolution": 384,
            "num_inference_steps": int(args.steps),
            "guidance_scale": float(args.guidance),
            "seed": int(args.model_seed),
            "batch_size": int(args.batch_size),
            "flash_block_size": int(args.flash_block_size),
            "results": [],
        }
    _json_dump(run_dir / "inference.partial.json", infer_log)

    completed = {row["episode_id"] for row in infer_log["results"]}
    batches = [episodes[i : i + args.batch_size] for i in range(0, len(episodes), args.batch_size)]
    for batch_index, batch in enumerate(batches):
        batch_ids = [ep["episode_id"] for ep in batch]
        already = [episode_id in completed for episode_id in batch_ids]
        if all(already):
            continue
        if any(already):
            raise SystemExit(f"partial batch in resume log at batch {batch_index}: {batch_ids}")
        grids = [_pil_grid(run_dir / ep["file"]) for ep in batch]
        started = time.perf_counter()
        outputs = pipe(
            task_prompt=[TASK_PROMPT] * args.batch_size,
            content_prompt=[CONTENT_PROMPT] * args.batch_size,
            image=grids,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            generator=[torch.Generator("cpu").manual_seed(args.model_seed) for _ in batch],
        ).images
        elapsed = time.perf_counter() - started
        for ep, cropped in zip(batch, outputs):
            episode_id = ep["episode_id"]
            output = cropped[0]
            raw_path = predictions / f"{episode_id}__raw384.png"
            score_path = predictions / f"{episode_id}__pred256.png"
            output.save(raw_path)
            output.resize((256, 256), Image.Resampling.LANCZOS).save(score_path)
            os.chmod(raw_path, 0o666)
            os.chmod(score_path, 0o666)
            row = {
                "episode_id": episode_id,
                "raw_prediction": str(raw_path.relative_to(run_dir)),
                "prediction": str(score_path.relative_to(run_dir)),
                "raw_sha256": _sha256(raw_path),
                "prediction_sha256": _sha256(score_path),
                "batch_seconds": elapsed,
                "batch_size": int(args.batch_size),
            }
            infer_log["results"].append(row)
            completed.add(episode_id)
        _json_dump(run_dir / "inference.partial.json", infer_log)
        print(
            f"batch {batch_index + 1}/{len(batches)}: {','.join(batch_ids)}: {elapsed:.1f}s "
            f"({len(infer_log['results'])}/{len(episodes)})",
            flush=True,
        )

    infer_log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_dump(run_dir / "inference.json", infer_log)
    (run_dir / "inference.partial.json").unlink(missing_ok=True)
    return 0


def score(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(LIVE_REPO))
    from jflux2.fasteval import metrics  # noqa: PLC0415

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    inference = json.loads((run_dir / "inference.json").read_text())
    current_hashes = _source_hashes()
    expected_hashes = manifest.get("source_files_sha256")
    if expected_hashes and expected_hashes.get("scorer") != current_hashes["scorer"]:
        raise SystemExit("refusing score: lineage-A scorer changed since episode preparation")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be positive")
    by_id = {x["episode_id"]: x for x in manifest["episodes"]}
    score_stem = "scores"
    partial_path = run_dir / f"{score_stem}.partial.json"
    if args.resume and partial_path.exists():
        progress = json.loads(partial_path.read_text())
        if progress.get("pack_id") != manifest["pack_id"]:
            raise SystemExit("refusing score resume: partial score pack_id changed")
        if progress.get("score_source_files_sha256") != current_hashes:
            raise SystemExit("refusing score resume: scorer source changed")
        results = list(progress["results"])
        print(f"resuming score with {len(results)}/{len(inference['results'])} records complete", flush=True)
    else:
        results = []
    completed = {row["episode_id"] for row in results}
    for generated in inference["results"]:
        if generated["episode_id"] in completed:
            continue
        ep = by_id[generated["episode_id"]]
        with np.load(run_dir / ep["file"]) as packed:
            target = np.asarray(packed["query_target"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        pred = np.asarray(Image.open(run_dir / generated["prediction"]).convert("RGB"), np.uint8)
        scored = metrics.score_pair(ep["task"], pred, target, strict=True)
        copied = metrics.score_pair(ep["task"], query_input, target, strict=True)
        perfect = metrics.score_pair(ep["task"], target, target, strict=True)
        results.append(
            {
                **ep,
                "prediction": generated["prediction"],
                "batch_seconds": generated.get("batch_seconds", generated.get("seconds")),
                "primary_metric": scored["primary_metric"],
                "primary_lower_is_better": bool(scored["primary_lower_is_better"]),
                "primary_value": float(scored["primary_value"]),
                "copy_input_floor": float(copied["primary_value"]),
                "perfect_target_ceiling": float(perfect["primary_value"]),
                "l1_01": float(scored["l1_01"]),
            }
        )
        completed.add(ep["episode_id"])
        if len(results) % args.checkpoint_every == 0 or len(results) == len(inference["results"]):
            _json_dump(
                partial_path,
                {
                    "schema": "visualcloze_lineage_a_score_progress_v1",
                    "pack_id": manifest["pack_id"],
                    "score_source_files_sha256": current_hashes,
                    "total_results": len(inference["results"]),
                    "results": results,
                },
            )
            print(f"scored {len(results)}/{len(inference['results'])} records", flush=True)
    cells = {}
    for task in sorted({row["task"] for row in results}):
        for condition in ("real", "scrambled", "grey", "oracle"):
            rows = [row for row in results if row["task"] == task and row["condition"] == condition]
            if not rows:
                continue
            values = np.asarray([row["primary_value"] for row in rows], np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                rng = np.random.default_rng(0)
                boots = finite[rng.integers(0, finite.size, size=(args.bootstrap, finite.size))].mean(axis=1)
                mean = float(finite.mean())
                ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                mean = float("nan")
                ci = [float("nan"), float("nan")]
            key = f"{task}@{condition}"
            cells[key] = {
                "task": task,
                "condition": condition,
                "primary_metric": rows[0]["primary_metric"],
                "primary_lower_is_better": rows[0]["primary_lower_is_better"],
                "mean": mean,
                "ci95": ci,
                "n": int(finite.size),
                "n_nan": int(values.size - finite.size),
                "copy_input_mean": float(np.nanmean([row["copy_input_floor"] for row in rows])),
                "perfect_target_mean": float(np.nanmean([row["perfect_target_ceiling"] for row in rows])),
            }
    report = {
        "schema": "visualcloze_lineage_a_score_v1",
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_id": manifest["pack_id"],
        "live_repo_git_sha": _git_sha(LIVE_REPO),
        "episode_source_files_sha256": expected_hashes,
        "score_source_files_sha256": current_hashes,
        "scorer": "jflux2.fasteval.metrics.score_pair(strict=True)",
        "bootstrap_samples": int(args.bootstrap),
        "cells": cells,
        "results": results,
    }
    _json_dump(run_dir / f"{score_stem}.json", report)
    partial_path.unlink(missing_ok=True)
    if args.verbose_records:
        for row in results:
            direction = "lower" if row["primary_lower_is_better"] else "higher"
            print(
                f"{row['episode_id']}: {row['primary_metric']}={row['primary_value']:.6f} "
                f"({direction} is better; copy={row['copy_input_floor']:.6f}; "
                f"perfect={row['perfect_target_ceiling']:.6f})",
                flush=True,
            )
    for key, cell in cells.items():
        direction = "lower" if cell["primary_lower_is_better"] else "higher"
        print(
            f"{key}: {cell['primary_metric']}={cell['mean']:.6f} "
            f"CI95=[{cell['ci95'][0]:.6f},{cell['ci95'][1]:.6f}] n={cell['n']} "
            f"({direction} is better; copy={cell['copy_input_mean']:.6f})",
            flush=True,
        )
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--pack", type=Path, default=PACK_DIR)
    pp.add_argument("--out", type=Path, required=True)
    pp.add_argument("--tasks", default="depth,normal,segmentation,panoptic")
    pp.add_argument("--conditions", default="real,scrambled,grey,oracle")
    pp.add_argument("--queries", type=int, default=32)
    pp.add_argument("--p", type=int, default=2)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--no-previews", action="store_true")
    pp.set_defaults(func=prepare)

    ip = sub.add_parser("infer")
    ip.add_argument("--run-dir", type=Path, required=True)
    ip.add_argument("--cache-dir", type=Path, default=Path("/dev/shm/visualcloze_eval"))
    ip.add_argument("--steps", type=int, default=30)
    ip.add_argument("--guidance", type=float, default=30.0)
    ip.add_argument("--model-seed", type=int, default=0)
    ip.add_argument("--batch-size", type=int, default=1)
    ip.add_argument("--flash-block-size", type=int, default=1536)
    ip.add_argument("--limit", type=int, default=0)
    ip.add_argument("--local-files-only", action="store_true")
    ip.add_argument("--resume", action="store_true")
    ip.set_defaults(func=infer)

    sp = sub.add_parser("score")
    sp.add_argument("--run-dir", type=Path, required=True)
    sp.add_argument("--bootstrap", type=int, default=20000)
    sp.add_argument("--checkpoint-every", type=int, default=64)
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--verbose-records", action="store_true")
    sp.set_defaults(func=score)
    return ap


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(parsed.func(parsed))
