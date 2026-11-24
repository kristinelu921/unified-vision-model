#!/usr/bin/env python3
"""Consensus-gated multi-device native inference for LaVin-DiT P=8."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

import lavin_dit_bench_v2 as base


QUALIFICATION_SCHEMA = "lavin_dit_bench_v2_native_p8_parallel_gate_v1"
PARALLEL_EXECUTION = "multi_device_native_reference_consensus_v1"
REFERENCE_SIGNATURE_SOURCE = "p8_v5p_retry15_eight_rank_reference_consensus_v1"


def rank_cpu_affinity(index: int, world_size: int) -> list[int]:
    """Give every XLA rank a disjoint share of the host CPUs.

    A cold XLA compile can otherwise let each rank fan out across the whole
    host.  Eight such compilers oversubscribe the machine badly enough to
    stall its control plane.  Affinity changes only host-side scheduling; the
    exact output gate still validates every rank before the long run begins.
    """

    available = sorted(os.sched_getaffinity(0))
    if not 0 <= index < world_size or len(available) < world_size:
        raise SystemExit(
            f"cannot partition host CPUs rank={index} world={world_size} "
            f"available={len(available)}"
        )
    width, remainder = divmod(len(available), world_size)
    start = index * width + min(index, remainder)
    stop = start + width + (1 if index < remainder else 0)
    assigned = available[start:stop]
    if not assigned:
        raise SystemExit(f"empty CPU affinity for rank={index}/{world_size}")
    return assigned


def worker(index: int, packed: dict) -> None:
    local_rank = int(os.environ["PJRT_LOCAL_PROCESS_RANK"])
    local_world = int(os.environ["PJRT_LOCAL_PROCESS_COUNT"])
    if local_world != 4 or not 0 <= local_rank < local_world:
        raise SystemExit(f"unexpected v5p host topology {local_rank}/{local_world}")
    cpu_affinity = rank_cpu_affinity(local_rank, local_world)
    os.sched_setaffinity(0, cpu_affinity)
    print(
        f"CPU_AFFINITY rank={index} local_rank={local_rank} local_world={local_world} count={len(cpu_affinity)} "
        f"first={cpu_affinity[0]} last={cpu_affinity[-1]}",
        flush=True,
    )
    args = argparse.Namespace(
        phase=packed["phase"],
        plan=Path(packed["plan"]),
        results_root=Path(packed["worker_root"]) / f"rank-{index:02d}",
        official_repo=Path(packed["official_repo"]),
        weights=Path(packed["weights"]),
        expected_tpu=packed["expected_tpu"],
        expected_zone=packed["expected_zone"],
        expected_devices=int(packed["expected_devices"]),
        validation_gate=(Path(packed["validation_gate"])
                         if packed.get("validation_gate") else None),
        validation_gate_sha256=packed.get("validation_gate_sha256"),
        sampler="reference",
        cfg_impl=packed["cfg_impl"],
        worker_rank=index,
        worker_world_size=int(packed["expected_devices"]),
        episode_key=packed.get("episode_key"),
        compile_cache=Path(packed["compile_cache"]) / f"rank-{index:02d}",
        compile_cache_readonly=bool(packed["compile_cache_readonly"]),
        cpu_affinity=cpu_affinity,
    )
    if packed.get("inline_qualification_key"):
        args.inline_qualification_key = packed["inline_qualification_key"]
        args.inline_qualification_root = (
            Path(packed["qualification_root"]) / f"rank-{index:02d}"
        )
        args.inline_qualification_compile_lock = (
            Path(packed["compile_cache"]) / f"rank-{index:02d}" / "qualification.lock"
        )
        args.inline_qualification_callback = lambda: inline_qualification(index, packed)
    base.infer(args)
    # On a multi-host slice, do not let worker zero's parent begin the merge
    # merely because its four local ranks happened to finish first.  This
    # rendezvous covers every global PJRT rank and therefore makes all rank
    # directories complete before the coordinator reads the shared NFS tree.
    import torch_xla.core.xla_model as xm
    xm.rendezvous(f"lavin_complete_{packed['plan_identity'][:20]}")


def host_index() -> int:
    """Return this TPU VM worker's stable slice index, failing closed."""
    for name in ("TPU_WORKER_ID", "CLOUD_TPU_TASK_ID"):
        value = os.environ.get(name)
        if value is not None:
            if not value.isdigit():
                raise SystemExit(f"invalid {name}: {value!r}")
            return int(value)
    match = re.search(r"-w-(\d+)$", socket.gethostname())
    if match:
        return int(match.group(1))
    # Single-host TPU VMs do not always expose a worker-id environment
    # variable.  Treat only the unambiguous single-host case as worker zero.
    return 0


def artifact_signature(episode: dict) -> dict:
    artifacts = episode["artifacts"]
    return {
        name: {
            "sha256": artifacts[name]["sha256"],
            "pixels_sha256": artifacts[name]["pixels_sha256"],
            "shape": artifacts[name]["shape"],
        }
        for name in ("raw_prediction", "prediction")
    }


def valid_signature(signature: object) -> bool:
    if not isinstance(signature, dict) or set(signature) != {"raw_prediction", "prediction"}:
        return False
    shapes = {"raw_prediction": [512, 512, 3], "prediction": [256, 256, 3]}
    for name, shape in shapes.items():
        descriptor = signature.get(name)
        if not isinstance(descriptor, dict) or descriptor.get("shape") != shape:
            return False
        for key in ("sha256", "pixels_sha256"):
            if not isinstance(descriptor.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", descriptor[key]):
                return False
    return True


def only_episode(results_root: Path, pack_slug: str) -> dict:
    payload = json.loads((results_root / "views" / pack_slug / "results.json").read_text())
    if payload.get("complete") is not True or len(payload.get("episodes", {})) != 1:
        raise SystemExit(f"qualification result is incomplete: {results_root}")
    return next(iter(payload["episodes"].values()))


def write_qualification(args, plan: dict) -> None:
    if args.qualification_receipt is None or args.episode_key is None:
        raise SystemExit("qualification requires its fixed episode and receipt path")
    matches = [ep for ep in plan["episodes"] if ep["key"] == args.episode_key]
    if len(matches) != 1:
        raise SystemExit("qualification episode does not identify exactly one frozen plan episode")
    pack = next(row for row in plan["packs"] if row["rel"] == matches[0]["pack_rel"])
    ranks = []
    reference_signature = None
    for rank in range(args.expected_devices):
        root = args.worker_root / f"rank-{rank:02d}"
        episode = only_episode(root, pack["slug"])
        signature = artifact_signature(episode)
        if not valid_signature(signature):
            raise SystemExit(f"INVALID_SIGNATURE rank={rank} actual={signature}")
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise SystemExit(
                f"CHECKSUM_MISMATCH rank={rank} reference={reference_signature} actual={signature}"
            )
        ranks.append({
            "rank": rank,
            "signature": signature,
            "seconds": episode["readouts"][base.READOUT]["real"]["seconds"],
        })
    receipt = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "PASS",
        "execution": PARALLEL_EXECUTION,
        "phase": args.phase,
        "episode_key": args.episode_key,
        "episode_id": matches[0]["episode_id"],
        "episode_pixels_sha256": matches[0]["episode_pixels_sha256"],
        "plan_identity": plan["plan_identity"],
        "model_seed": base.MODEL_SEED,
        "steps": base.STEPS,
        "cfg_scale": base.CFG,
        "cfg_impl": args.cfg_impl,
        "rng_contract": base.RNG_CONTRACT,
        "sampler": "torchdiffeq_fixed_grid_euler",
        "reference_signature": reference_signature,
        "reference_signature_source": REFERENCE_SIGNATURE_SOURCE,
        "parallel_ranks": ranks,
        "world_size": args.expected_devices,
        "expected_tpu": args.expected_tpu,
        "expected_zone": args.expected_zone,
        "runner_sha256": base.sha256_file(Path(base.__file__)),
        "parallel_runner_sha256": base.sha256_file(Path(__file__)),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    base.atomic_json(args.qualification_receipt, receipt)
    print(
        f"PARALLEL_CHECKSUM_GATE_PASS episode={args.episode_key} "
        f"raw={reference_signature['raw_prediction']['sha256']} "
        f"prediction={reference_signature['prediction']['sha256']}",
        flush=True,
    )


def inline_qualification(index: int, packed: dict) -> None:
    """Fail closed on all ranks, then keep the compiled workers alive for the run."""
    import torch_xla.core.xla_model as xm

    receipt = Path(packed["qualification_receipt"])
    plan = base.validate_plan(Path(packed["plan"]), packed["phase"])
    tag = plan["plan_identity"][:20]
    xm.rendezvous(f"lavin_outputs_{tag}")
    if index == 0:
        try:
            matches = [ep for ep in plan["episodes"] if ep["key"] == packed["inline_qualification_key"]]
            if len(matches) != 1:
                raise RuntimeError("inline qualification episode is not unique")
            pack = next(row for row in plan["packs"] if row["rel"] == matches[0]["pack_rel"])
            ranks = []
            # Bind every rank to the independently hash-verified eight-rank v5p reference.
            reference_signature = packed["reference_signature"]
            for rank in range(int(packed["expected_devices"])):
                root = Path(packed["qualification_root"]) / f"rank-{rank:02d}"
                episode = only_episode(root, pack["slug"])
                signature = artifact_signature(episode)
                if not valid_signature(signature):
                    raise RuntimeError(f"INVALID_SIGNATURE rank={rank} actual={signature}")
                if reference_signature is None:
                    reference_signature = signature
                elif signature != reference_signature:
                    raise RuntimeError(
                        f"CHECKSUM_MISMATCH rank={rank} reference={reference_signature} actual={signature}"
                    )
                ranks.append({
                    "rank": rank,
                    "signature": signature,
                    "seconds": episode["readouts"][base.READOUT]["real"]["seconds"],
                })
            payload = {
                "schema": QUALIFICATION_SCHEMA,
                "status": "PASS",
                "execution": PARALLEL_EXECUTION,
                "phase": packed["phase"],
                "episode_key": packed["inline_qualification_key"],
                "episode_id": matches[0]["episode_id"],
                "episode_pixels_sha256": matches[0]["episode_pixels_sha256"],
                "plan_identity": plan["plan_identity"],
                "model_seed": base.MODEL_SEED,
                "steps": base.STEPS,
                "cfg_scale": base.CFG,
                "cfg_impl": packed["cfg_impl"],
                "rng_contract": base.RNG_CONTRACT,
                "sampler": "torchdiffeq_fixed_grid_euler",
                "reference_signature": reference_signature,
                "reference_signature_source": REFERENCE_SIGNATURE_SOURCE,
                "parallel_ranks": ranks,
                "world_size": int(packed["expected_devices"]),
                "expected_tpu": packed["expected_tpu"],
                "expected_zone": packed["expected_zone"],
                "runner_sha256": base.sha256_file(Path(base.__file__)),
                "parallel_runner_sha256": base.sha256_file(Path(__file__)),
                "continued_in_process": True,
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        except BaseException as error:
            payload = {
                "schema": QUALIFICATION_SCHEMA,
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        base.atomic_json(receipt, payload)
    xm.rendezvous(f"lavin_gate_{tag}")
    gate = json.loads(receipt.read_text())
    if gate.get("status") != "PASS":
        raise SystemExit(f"inline checksum gate failed: {gate}")


def verify_artifacts(root: Path, episode: dict) -> None:
    for name in ("query_input", "query_target", "raw_prediction", "prediction"):
        descriptor = episode["artifacts"][name]
        path = root / descriptor["path"]
        if not path.is_file() or base.sha256_file(path) != descriptor["sha256"]:
            raise SystemExit(f"parallel artifact failed merge verification: {path}")


def merge_parallel(args, plan: dict) -> None:
    execution = {
        "mode": PARALLEL_EXECUTION,
        "sampler": "reference",
        "cfg_impl": args.cfg_impl,
        "worker_world_size": args.expected_devices,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    by_rel = {row["rel"]: row for row in plan["packs"]}
    merged = {
        row["rel"]: base.initial_view(row, plan, row["expected_episode_keys"], execution)
        for row in plan["packs"]
    }
    for rank in range(args.expected_devices):
        rank_root = args.worker_root / f"rank-{rank:02d}"
        for pack in plan["packs"]:
            rank_view_path = rank_root / "views" / pack["slug"] / "results.json"
            if not rank_view_path.is_file():
                continue
            rank_view = json.loads(rank_view_path.read_text())
            rank_view_root = rank_view_path.parent
            canonical_root = args.results_root / "views" / pack["slug"]
            for key, episode in rank_view["episodes"].items():
                if key in merged[pack["rel"]]["episodes"]:
                    raise SystemExit(f"duplicate parallel episode during merge: {pack['rel']} {key}")
                episode_dir = Path(episode["artifacts"]["prediction"]["path"]).parent
                shutil.copytree(rank_view_root / episode_dir, canonical_root / episode_dir,
                                dirs_exist_ok=False)
                verify_artifacts(canonical_root, episode)
                merged[pack["rel"]]["episodes"][key] = episode
    for rel, view in merged.items():
        pack = by_rel[rel]
        if set(view["episodes"]) != set(pack["expected_episode_keys"]):
            missing = sorted(set(pack["expected_episode_keys"]) - set(view["episodes"]))
            raise SystemExit(f"parallel merge lacks {len(missing)} episodes for {rel}: {missing[:8]}")
        view["complete"] = True
        base.atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
    base.atomic_json(args.results_root / "INFERENCE_COMPLETE.json", {
        "schema": "lavin_dit_bench_v2_native_p8_parallel_complete_v1",
        "phase": args.phase,
        "packs": len(plan["packs"]),
        "episodes": len(plan["episodes"]),
        "plan_identity": plan["plan_identity"],
        "execution": execution,
        "optimization_gate_sha256": base.sha256_file(
            args.optimization_gate or args.qualification_receipt
        ),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def validate_optimization_gate(args, plan: dict) -> dict:
    if args.optimization_gate is None or not args.optimization_gate.is_file():
        raise SystemExit("long parallel inference requires the checksum parity gate")
    gate = json.loads(args.optimization_gate.read_text())
    expected = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "PASS",
        "execution": PARALLEL_EXECUTION,
        "plan_identity": plan["plan_identity"],
        "world_size": args.expected_devices,
        "expected_tpu": args.expected_tpu,
        "expected_zone": args.expected_zone,
        "episode_key": "saliency_detection:0",
        "model_seed": base.MODEL_SEED,
        "steps": base.STEPS,
        "cfg_scale": base.CFG,
        "cfg_impl": args.cfg_impl,
        "rng_contract": base.RNG_CONTRACT,
        "sampler": "torchdiffeq_fixed_grid_euler",
        "runner_sha256": base.sha256_file(Path(base.__file__)),
        "parallel_runner_sha256": base.sha256_file(Path(__file__)),
    }
    ranks = gate.get("parallel_ranks")
    signature = gate.get("reference_signature")
    ranks_match = (
        isinstance(ranks, list) and len(ranks) == args.expected_devices
        and {row.get("rank") for row in ranks if isinstance(row, dict)} == set(range(args.expected_devices))
        and all(row.get("signature") == signature for row in ranks if isinstance(row, dict))
    )
    if (any(gate.get(key) != value for key, value in expected.items())
            or not valid_signature(signature)
            or gate.get("reference_signature_source") != REFERENCE_SIGNATURE_SOURCE
            or not ranks_match):
        raise SystemExit("checksum parity gate identity differs; refusing long inference")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "run", "gated-run"), required=True)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-tpu", required=True)
    parser.add_argument("--expected-zone", required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--cfg-impl", choices=("reference", "batched"), default="reference")
    parser.add_argument("--compile-cache", type=Path, required=True)
    parser.add_argument("--episode-key")
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--optimization-gate", type=Path)
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--validation-gate-sha256")
    parser.add_argument("--resume-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_devices not in (4, 8, 16):
        raise SystemExit("this parallel runner requires exactly four, eight, or sixteen TPU devices")
    plan = base.validate_plan(args.plan, args.phase)
    resume = json.loads(args.resume_receipt.read_text())
    if (resume.get("status") != "PASS"
            or resume.get("plan_identity") != plan["plan_identity"]
            or resume.get("world_size") != args.expected_devices
            or not valid_signature(resume.get("reference_signature"))):
        raise SystemExit("verified same-hardware reference receipt required")
    if args.mode == "run":
        if args.results_root is None:
            raise SystemExit("run mode requires the canonical results root")
        validate_optimization_gate(args, plan)
    if args.mode == "gated-run" and (
        args.results_root is None
        or args.qualification_receipt is None
        or args.episode_key != "saliency_detection:0"
    ):
        raise SystemExit(
            "gated-run requires results, its receipt, and the frozen qualification episode"
        )
    packed = {
        "phase": args.phase,
        "plan": str(args.plan),
        "worker_root": str(args.worker_root),
        "official_repo": str(args.official_repo),
        "weights": str(args.weights),
        "expected_tpu": args.expected_tpu,
        "expected_zone": args.expected_zone,
        "expected_devices": args.expected_devices,
        "compile_cache": str(args.compile_cache),
        "compile_cache_readonly": args.mode == "run",
        "cfg_impl": args.cfg_impl,
        "episode_key": args.episode_key if args.mode == "qualify" else None,
        "inline_qualification_key": args.episode_key if args.mode == "gated-run" else None,
        "qualification_root": str(args.worker_root.parent / "inline_qualification"),
        "qualification_receipt": str(args.qualification_receipt) if args.qualification_receipt else None,
        "validation_gate": str(args.validation_gate) if args.validation_gate else None,
        "validation_gate_sha256": args.validation_gate_sha256,
        "plan_identity": plan["plan_identity"],
        "reference_signature": resume["reference_signature"],
    }
    import torch_xla
    torch_xla.launch(worker, args=(packed,), start_method="spawn")
    coordinator = host_index() == 0
    completion = args.results_root / "INFERENCE_COMPLETE.json" if args.results_root else None
    if not coordinator:
        if args.mode == "qualify":
            return 0
        # The global device rendezvous above guarantees that all ranks are
        # finished.  Only worker zero performs the canonical merge; other
        # parents wait for its atomic completion receipt and then exit.
        deadline = time.monotonic() + 3600
        while not completion.is_file():
            if time.monotonic() >= deadline:
                raise SystemExit("timed out waiting for coordinator merge receipt")
            time.sleep(2)
        return 0
    if args.mode == "qualify":
        write_qualification(args, plan)
    else:
        if args.mode == "gated-run":
            gate = json.loads(args.qualification_receipt.read_text())
            if gate.get("status") != "PASS" or not gate.get("continued_in_process"):
                raise SystemExit("inline qualification did not pass")
        merge_parallel(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
