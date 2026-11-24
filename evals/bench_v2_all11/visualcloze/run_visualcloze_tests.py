"""Run the qualified VisualCloze P=2 recipe on all frozen Bench v2 test episodes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket


ROOT = Path(__file__).resolve().parent
WORK = Path("/tmp/kristine_bench_all11_visualcloze_20261111")
PACK_ROOT = Path("/dev/shm/kristine_bench_all11_visualcloze_test_packs")
TEST_LOCAL_ROOT = Path("/dev/shm/kristine_bench_all11_visualcloze_test_run")
OUTPUT = TEST_LOCAL_ROOT / "outputs"
TPU = "kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a"
HOST = "t1v-n-4b587646-w-0"
ORIGINAL_PLAN = ROOT / "visualcloze_p2_test_plan.json"
PLAN_COMPLETE = ROOT / "visualcloze_p2_test_plan_complete.json"
RUNNER = ROOT / "recovered/visualcloze_recovery/source/baseline/p2/visualcloze_p2_test_runner.py"
DERIVATION = ROOT / "visualcloze_p2_test_runner_derivation.json"
QUALIFICATION = ROOT / "visualcloze_diagnostic9_qualification_decision.json"
ORIGINAL_PLAN_SHA256 = "6f41ba19c62f6d8f54bc7a93b8c946114bab2e8d12b680fa30469e0e455b9484"
ORIGINAL_PLAN_IDENTITY = "a39fa491834152f410f71a0034a5c14daa6067b03fee51d22175b26c5937c133"
ORIGINAL_RUNNER_SHA256 = "5b13ab08885d48b20cf91c457f6ff8f960e788238bfb24a05106a95613162912"
DERIVED_RUNNER_SHA256 = "3f326df33186851ad8e3f82c381a474744e550b7f062fdfb49fcca428d0836d8"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    assert socket.gethostname() == HOST
    assert os.environ["EXPECTED_TPU_NAME"] == TPU
    assert os.environ["EXPECTED_TPU_ZONE"] == "us-west4-a"
    assert sha(ORIGINAL_PLAN) == ORIGINAL_PLAN_SHA256
    assert sha(RUNNER) == DERIVED_RUNNER_SHA256
    complete = json.loads(PLAN_COMPLETE.read_text())
    assert complete == {
        "builder_sha256": "f6898192c729d3eb6e4a4217ae6ea7d69f6c2696ee91b5c8035baa570d24fe2f",
        "episodes": 17511,
        "packs": 26,
        "padded_episodes": 17512,
        "plan_identity": ORIGINAL_PLAN_IDENTITY,
        "plan_sha256": ORIGINAL_PLAN_SHA256,
        "scope": "Full selected P2 bindings and exact validation replay; model qualification pending. Pack paths describe preparation host and require verified restaging before accelerator launch.",
        "status": "PASS",
        "validation_binding_parity": 221,
    }
    derivation = json.loads(DERIVATION.read_text())
    assert derivation["status"] == "PASS"
    assert derivation["source_sha256"] == ORIGINAL_RUNNER_SHA256
    assert derivation["output_sha256"] == DERIVED_RUNNER_SHA256
    assert derivation["unchanged_bytes_sha256_excluding_scope_block"] == derivation["derived_bytes_sha256_excluding_scope_block"]
    qualification = json.loads(QUALIFICATION.read_text())
    assert qualification["status"] == "QUALIFIED"
    assert qualification["model"] == "visualcloze" and qualification["recipe"] == "visualcloze-p2"

    spec = importlib.util.spec_from_file_location("visualcloze_p2_test_runner", RUNNER)
    selected = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(selected)
    plan = json.loads(ORIGINAL_PLAN.read_text())
    original_identity = plan.pop("plan_identity")
    assert original_identity == ORIGINAL_PLAN_IDENTITY
    assert hashlib.sha256(selected.canonical_bytes(plan)).hexdigest() == original_identity
    assert plan["source_sha256"]["runner"] == ORIGINAL_RUNNER_SHA256
    assert len(plan["packs"]) == 26
    assert len(plan["episodes"]) == 17512
    assert sum(not episode["padding"] for episode in plan["episodes"]) == 17511
    assert all(int(episode["p"]) == 2 for episode in plan["episodes"])
    assert sum(bool(episode["padding"]) for episode in plan["episodes"]) == 1

    verified_files = 0
    for pack in plan["packs"]:
        assert pack["rel"].startswith("bench_v2_test/")
        relative = pack["rel"].removeprefix("bench_v2_test/")
        assert relative != pack["rel"]
        local_dir = PACK_ROOT / relative
        for name, expected in sorted(pack["data_identity"]["file_sha256"].items()):
            path = local_dir / name
            assert path.is_file(), path
            assert sha(path) == expected, path
            verified_files += 1
        pack["local_dir"] = str(local_dir)
    assert verified_files == 78
    plan["source_sha256"]["runner"] = DERIVED_RUNNER_SHA256
    plan["plan_identity"] = hashlib.sha256(selected.canonical_bytes(plan)).hexdigest()
    node_plan = WORK / "visualcloze_p2_test_plan_node_local.json"
    if node_plan.exists():
        assert json.loads(node_plan.read_text()) == plan
    else:
        atomic_json(node_plan, plan)
    atomic_json(
        WORK / "visualcloze_p2_test_plan_rebinding.json",
        {
            "status": "PASS",
            "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "original_plan_identity": original_identity,
            "node_local_plan_identity": plan["plan_identity"],
            "original_runner_sha256": ORIGINAL_RUNNER_SHA256,
            "test_runner_sha256": DERIVED_RUNNER_SHA256,
            "verified_pack_files": verified_files,
            "real_episodes": 17511,
            "padded_episodes": 17512,
            "pack_count": 26,
            "attempt": args.attempt,
        },
    )

    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    original_decode = AutoencoderKL.decode
    raw_outputs = 0
    raw_hash_chain = bytes(32)

    def decode(self, *decode_args, **decode_kwargs):
        nonlocal raw_outputs, raw_hash_chain
        target_norm = self.decoder.up_blocks[3].resnets[2].norm2
        original_norm_forward = target_norm.forward

        def fp32_group_norm(input):
            weight = target_norm.weight.float() if target_norm.weight is not None else None
            bias = target_norm.bias.float() if target_norm.bias is not None else None
            return torch.nn.functional.group_norm(
                input.float(), target_norm.num_groups, weight, bias, target_norm.eps
            ).to(input.dtype)

        target_norm.forward = fp32_group_norm
        try:
            result = original_decode(self, *decode_args, **decode_kwargs)
        finally:
            target_norm.forward = original_norm_forward
        value = result[0] if isinstance(result, tuple) else result.sample
        array = np.ascontiguousarray(value.detach().float().cpu().numpy())
        if not np.isfinite(array).all():
            raise RuntimeError("FP32 norm2 patched raw VAE output contains nonfinite values")
        raw_outputs += int(array.shape[0])
        raw_digest = hashlib.sha256()
        raw_digest.update(raw_hash_chain)
        raw_digest.update(memoryview(array).cast("B"))
        raw_hash_chain = raw_digest.digest()
        if raw_outputs % 20 == 0:
            atomic_json(
                WORK / "visualcloze_p2_test_patch_observer.json",
                {
                    "attempt": args.attempt,
                    "finite_raw_outputs_this_attempt": raw_outputs,
                    "raw_output_hash_chain": raw_hash_chain.hex(),
                    "target_patch": "decoder.up_blocks.3.resnets.2.norm2 FP32 group_norm arithmetic on every decode, cast back to input dtype",
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
        return result

    AutoencoderKL.decode = decode
    parsed = selected.parser().parse_args(
        [
            "infer",
            "--plan", str(node_plan),
            "--results-root", str(OUTPUT),
            "--model-dir", str(WORK / "model"),
            "--model-sha256s", str(WORK / "model/MODEL_SHA256SUMS"),
            "--cache-dir", str(TEST_LOCAL_ROOT / "cache"),
            "--expected-tpu", TPU,
            "--expected-zone", "us-west4-a",
            "--expected-devices", "8",
            "--batch-size", "2",
            "--flash-block-size", "1024",
        ]
    )
    state = "FAILED"
    error = None
    try:
        result = selected.infer(parsed)
        assert result == 0
        final = json.loads((OUTPUT / "status.json").read_text())
        assert final["state"] == "inference_complete"
        assert final["completed"] == final["total"] == 17511
        assert final["plan_identity"] == plan["plan_identity"]
        state = "TEST_INFERENCE_COMPLETE"
        return 0
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        atomic_json(
            WORK / "visualcloze_p2_test_run_receipt.json",
            {
                "status": state,
                "error": error,
                "model": "visualcloze",
                "recipe": "visualcloze-p2",
                "attempt": args.attempt,
                "tpu": TPU,
                "host": HOST,
                "original_plan_identity": original_identity,
                "node_local_plan_identity": plan["plan_identity"],
                "test_runner_sha256": DERIVED_RUNNER_SHA256,
                "finite_raw_outputs_this_attempt": raw_outputs,
                "raw_output_hash_chain": raw_hash_chain.hex(),
                "target_patch": "decoder.up_blocks.3.resnets.2.norm2 FP32 group_norm arithmetic on every decode, cast back to input dtype",
                "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
