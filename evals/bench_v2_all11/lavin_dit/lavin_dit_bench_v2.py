#!/usr/bin/env python3
"""Prepare and run the frozen LaVin-DiT native-P=8 Bench v2 evaluation."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image


LIVE_REPO = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
TRUSTED_MANIFEST = LIVE_REPO / "docs/bench_v2_audit/manifest_v1.json"
MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"
PACK_URI_ROOT = os.environ.get(
    "LAVIN_PACK_URI_ROOT", "gs://kmh-gcp-us-central2/data/xtiange/mt_eval_packs"
)
SCHEMA = "lavin_dit_bench_v2_native_p8_inference_v1"
PLAN_SCHEMA = "lavin_dit_bench_v2_native_p8_plan_v1"
VALIDATION_GATE_SCHEMA = "lavin_dit_bench_v2_native_p8_validation_gate_v1"
CORRECTED_PROTOCOL = "bench_v2_PP2_CSA_seed1_AP_FIRST_v1"
CORRECTED_RELEASE = "bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792"
READOUT = "LaVin-DiT@native_p8"
P = 8
EPISODE_SEED = 42
MODEL_SEED = 0
RNG_CONTRACT = "cpu_generator_seed0_per_draw_v1"
RAW_SIZE = 512
SCORE_SIZE = 256
STEPS = 20
CFG = 1.5
OFFICIAL_COMMIT = "f3728580115a4efbcd38d2279f1dbd868297b05e"
WEIGHTS_REVISION = "908f27447c65a95ba480ead9ef30f59965817593"
DIT_SHA256 = "bba27654d882641f6dde683c0ff07b698d07a4d9c4d9bd6bec6050f204199cee"
VAE_SHA256 = "4a219a05e273906b3979bbd2a149dab9c13257ee606bf8f4c7be3609da1ac902"
SOURCE_DIT_SHA256 = "9e6074490dfde9e1ad2c814503b31889952c9b60a17667eef791e7ea47a4b2f3"
SOURCE_VAE_SHA256 = "d8305e48b3fe83f55a90d3fe00791ae87841a03b4b6872cb8924dd65b6a8dbff"
EULER_TIMES_20 = (
    0.0000, 0.0056, 0.0111, 0.0167, 0.0222, 0.0278, 0.0333, 0.0389,
    0.0444, 0.0500, 0.0524, 0.0730, 0.1148, 0.1777, 0.2618, 0.3671,
    0.4936, 0.6412, 0.8100, 1.0000,
)
STVAE_TRAINING_ONLY_PREFIXES = (
    "loss2d.perceptual_loss.", "loss2d.discriminator.",
    "loss3d.perceptual_loss.", "loss3d.discriminator.", "lpips.net.",
)


class AuthenticatedGcsDownloadStore:
    """Minimal gs:// reader that streams directly without gsutil or /dev/shm."""

    scheme = "gs://"

    def download(self, uri: str, destination: Path) -> None:
        if not uri.startswith(self.scheme):
            raise ValueError(f"unsupported URI for authenticated GCS download: {uri}")
        bucket_and_object = uri[len(self.scheme):]
        bucket, separator, object_name = bucket_and_object.partition("/")
        if not separator or not bucket or not object_name:
            raise ValueError(f"malformed GCS object URI: {uri}")
        url = (
            f"https://storage.googleapis.com/{bucket}/"
            f"{urllib.parse.quote(object_name, safe='/')}"
        )
        destination = Path(destination)
        last_error = None
        for attempt in range(5):
            try:
                token = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"], text=True
                ).strip()
                request = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {token}"}
                )
                with urllib.request.urlopen(request, timeout=120) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=8 << 20)
                return
            except Exception as error:  # network retries still end fail-closed at the pinned SHA-256 check
                last_error = error
                destination.unlink(missing_ok=True)
                if attempt < 4:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"authenticated GCS download failed for {uri}: {last_error}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    Image.fromarray(np.asarray(pixels, np.uint8), "RGB").save(tmp, format="PNG", optimize=False)
    os.replace(tmp, path)


def png_descriptor(path: Path, root: Path) -> dict:
    raw = path.read_bytes()
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"not a PNG: {path}")
        pixels = np.asarray(image.convert("RGB"), np.uint8)
        shape = [image.height, image.width, 3]
    return {
        "path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest(),
        "pixels_sha256": sha256_array(pixels), "shape": shape, "dtype": "uint8", "mode": "RGB",
    }


def task_rows(pack: dict, task: str) -> list[int]:
    return [i for i, row in enumerate(pack["records"]) if row["task"] == task]


def pack_slug(rel: str) -> str:
    return rel.replace("/", "__")


def episode_sha(demo_inputs, demo_targets, query_input, query_target) -> str:
    h = hashlib.sha256()
    for tile in (*demo_inputs, *demo_targets, query_input, query_target):
        h.update(memoryview(np.ascontiguousarray(tile)).cast("B"))
    return h.hexdigest()


def validate_conversion_manifest(weights_dir: Path) -> dict:
    path = weights_dir / "conversion_manifest.json"
    if not path.is_file():
        raise SystemExit("BF16 conversion manifest is missing")
    manifest = json.loads(path.read_text())
    expected_source = {
        "lavin_dit.safetensors": {"bytes": 13_595_959_128, "sha256": SOURCE_DIT_SHA256},
        "stvae.pt": {"bytes": 1_068_865_830, "sha256": SOURCE_VAE_SHA256},
    }
    expected_converted = {
        "lavin_dit.bf16.safetensors": {"bytes": 6_798_007_216, "sha256": DIT_SHA256},
        "stvae.bf16.pt": {"bytes": 534_531_387, "sha256": VAE_SHA256},
    }
    if (manifest.get("schema") != "lavin_dit_bench_v2_bf16_conversion_v1"
            or manifest.get("source_revision") != WEIGHTS_REVISION
            or manifest.get("source") != expected_source
            or manifest.get("converted") != expected_converted):
        raise SystemExit("BF16 conversion manifest does not bind the pinned official sources and outputs")
    for name, spec in expected_converted.items():
        file = weights_dir / name
        if not file.is_file() or file.stat().st_size != spec["bytes"] or sha256_file(file) != spec["sha256"]:
            raise SystemExit(f"converted checkpoint differs from its conversion manifest: {name}")
    return manifest


def source_hashes(official_repo: Path) -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "parallel_runner": Path(__file__).with_name("lavin_dit_parallel.py"),
        "defaults": Path(__file__).with_name("defaults.json"),
        "tpu_compat": Path(__file__).with_name("tpu_compat.py"),
        "corrected_materializer": Path(__file__).with_name("materialize_corrected_v3_units.py"),
        "phase_receipt": Path(__file__).with_name("create_phase_receipt.py"),
        "runtime_v2": LIVE_REPO / "tools/bench_v2_runtime_v2.py",
        "episode_contract": LIVE_REPO / "jflux2/analysis/t16_offline_eval_v1.py",
        "episode_replay": LIVE_REPO / "tools/bench_episode_v1.py",
        "conventions": LIVE_REPO / "jflux2/dataloaders/multi_task_loader_v4.py",
        "producer_validator": LIVE_REPO / "tools/score_bench_outputs_v1.py",
        "corrected_backend": LIVE_REPO / "tools/bench_eval_v3_backend_v1.py",
        "official_inference": official_repo / "inference.py",
        "official_dit": official_repo / "lib/model/dit.py",
        "official_blocks": official_repo / "lib/model/blocks.py",
        "official_vae": official_repo / "lib/model/vae3d.py",
        "official_integrator": official_repo / "lib/transport/integrators.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


def phase_rels(manifest: dict, phase: str) -> list[str]:
    if phase == "validation":
        rels = ["bench_v2_val"]
        expected = (1, 221)
    elif phase == "test":
        rels = sorted(row["rel"] for row in manifest["packs"] if row["rel"].startswith("bench_v2_test/"))
        expected = (26, 17_511)
    else:
        raise ValueError(f"unsupported phase: {phase}")
    by_rel = {row["rel"]: int(row["n"]) for row in manifest["packs"]}
    if len(rels) != expected[0] or sum(by_rel[rel] for rel in rels) != expected[1]:
        raise SystemExit("Bench v2 phase cardinality differs")
    return rels


def build_episode(imgs, msks, srcs, task: str, q: int, seed: int, p_max: int) -> dict:
    """Exact lightweight form of capacity_ladder_v9.build_episode at pool start 0.

    Importing the full capacity-ladder module also imports the FLUX training
    stack, which is unrelated to this model's frozen episode plan.  This keeps
    the authoritative T16/CV4 draw and coding contract while avoiding those
    accelerator-only dependencies during CPU plan construction.
    """
    from jflux2.analysis import t16_offline_eval_v1 as T16
    from jflux2.dataloaders import multi_task_loader_v4 as CV4

    T16.CV = CV4
    same = [k for k, source in enumerate(srcs) if source == srcs[int(q)]]
    candidates = [k for k in same if k != int(q)]
    p0 = min(int(p_max), len(candidates))
    rng = np.random.default_rng([int(seed), T16.task_seed(task), int(q), p0])
    cid = 0
    try:
        is_palette = CV4.family_of(task) == "palette"
    except (KeyError, ValueError):
        is_palette = False
    if is_palette:
        pool = CV4.convention_pools(task)[1]
        cid = int(pool[int(rng.integers(0, len(pool)))])
    conditions, demo_indices, query_target = T16.build_conditions(
        imgs, msks, task, int(q), p0, rng, cid=cid,
        which=("real",), candidates=same,
    )
    return {
        "cid": cid,
        "p0": p0,
        "candidates": [int(k) for k in candidates],
        "same": same,
        "demo_indices": [int(j) for j in demo_indices],
        "real": conditions["real"],
        "query_t": np.asarray(query_target, np.uint8),
    }


def prepare_plan(args) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(LIVE_REPO))
    from tools.bench_v2_runtime_v2 import stage_pack

    if sha256_file(TRUSTED_MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("trusted Bench v2 manifest changed")
    if not (args.official_repo / ".git").exists():
        raise SystemExit("official repository checkout is missing")
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.official_repo}", "-C", str(args.official_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise SystemExit(f"official source commit differs: {commit}")
    manifest = json.loads(TRUSTED_MANIFEST.read_text())
    rels = phase_rels(manifest, args.phase)
    plan = {
        "schema": PLAN_SCHEMA, "protocol": SCHEMA, "phase": args.phase,
        "trusted_manifest": str(TRUSTED_MANIFEST), "trusted_manifest_file_sha256": MANIFEST_SHA256,
        "pack_uri_root": PACK_URI_ROOT, "official_source_commit": OFFICIAL_COMMIT,
        "weights_revision": WEIGHTS_REVISION, "condition": "real", "episode_seed": EPISODE_SEED,
        "p": P, "model_seed": MODEL_SEED, "height": RAW_SIZE, "width": RAW_SIZE,
        "steps": STEPS, "cfg_scale": CFG, "dtype": "bfloat16", "readout": READOUT,
        "backend_label": "PyTorch/XLA TPU-native; CUDA numerical parity is not claimed",
        "source_sha256": source_hashes(args.official_repo), "packs": [], "episodes": [],
    }
    pack_store = AuthenticatedGcsDownloadStore()
    for rel in rels:
        pack = stage_pack(rel, uri_root=PACK_URI_ROOT, manifest_path=TRUSTED_MANIFEST,
                          manifest_sha256=MANIFEST_SHA256, local_root=args.pack_cache,
                          store=pack_store)
        tasks = list(dict.fromkeys(row["task"] for row in pack["records"]))
        start = len(plan["episodes"])
        keys = []
        for task in tasks:
            rows = task_rows(pack, task)
            images, masks = pack["images"][rows], pack["masks"][rows]
            records = [pack["records"][i] for i in rows]
            sources = [str(row.get("source")) for row in records]
            for q in range(len(rows)):
                built = build_episode(images, masks, sources, task, q, EPISODE_SEED, P)
                if int(built["p0"]) != P:
                    raise SystemExit(f"{rel} {task} q{q}: insufficient same-source demonstrations")
                demo_local = [int(i) for i in built["demo_indices"]]
                demo_global = [int(rows[i]) for i in demo_local]
                demo_inputs = [np.asarray(images[i], np.uint8) for i in demo_local]
                demo_targets = [np.rint(np.transpose(pair[1], (1, 2, 0)) * 255).astype(np.uint8) for pair in built["real"]]
                query_input = np.asarray(images[q], np.uint8)
                query_target = np.asarray(built["query_t"], np.uint8)
                key = f"{task}:{q}"
                keys.append(key)
                plan["episodes"].append({
                    "episode_id": f"{pack_slug(rel)}__{task}__q{q:05d}__P8__real",
                    "key": key, "pack_rel": rel, "pack_slug": pack_slug(rel), "task": task,
                    "q": q, "p": P, "cid": int(built["cid"]), "query_global": int(rows[q]),
                    "query_record": str(records[q]["record_id"]), "demo_global": demo_global,
                    "demo_indices": demo_local,
                    "demo_records": [str(records[i]["record_id"]) for i in demo_local],
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
    expected = 221 if args.phase == "validation" else 17_511
    if len(plan["episodes"]) != expected:
        raise SystemExit("Bench v2 phase query count differs")
    plan["plan_identity"] = identity(plan)
    atomic_json(args.out, plan)
    print(f"[plan] phase={args.phase} identity={plan['plan_identity']} episodes={expected} packs={len(rels)}", flush=True)
    return 0


class PackCache:
    def __init__(self, plan):
        self.info = {row["rel"]: row for row in plan["packs"]}
        self.rel = None
        self.images = self.masks = None

    def load(self, rel):
        if rel == self.rel:
            return self.images, self.masks
        with np.load(Path(self.info[rel]["local_dir"]) / "samples.npz", allow_pickle=False) as archive:
            self.images = np.array(archive["images"], copy=True)
            self.masks = np.array(archive["masks"], copy=True)
        self.images.setflags(write=False)
        self.masks.setflags(write=False)
        self.rel = rel
        return self.images, self.masks


def reconstruct(ep, packs):
    sys.path.insert(0, str(LIVE_REPO))
    from tools import bench_episode_v1 as BE
    images, masks = packs.load(ep["pack_rel"])
    demo_inputs = [np.asarray(images[i], np.uint8) for i in ep["demo_global"]]
    coded = BE.code_episode([masks[i] for i in ep["demo_global"]] + [masks[ep["query_global"]]], ep["task"], ep["cid"])
    demo_targets, query_target = coded[:-1], coded[-1]
    query_input = np.asarray(images[ep["query_global"]], np.uint8)
    if episode_sha(demo_inputs, demo_targets, query_input, query_target) != ep["episode_pixels_sha256"]:
        raise SystemExit(f"{ep['episode_id']}: reconstructed episode differs")
    return demo_inputs, demo_targets, query_input, query_target


def image_tensor(torch, image, size=RAW_SIZE):
    from torchvision.transforms import v2
    value = torch.from_numpy(np.asarray(image, np.uint8).copy()).permute(2, 0, 1)
    value = v2.functional.resize(value, (size, size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True)
    return (value.float() / 255.0 - 0.5) * 2.0


def condition_tensor(torch, demo_inputs, demo_targets, query_input):
    frames = []
    for before, after in zip(demo_inputs, demo_targets):
        frames.extend((image_tensor(torch, before), image_tensor(torch, after)))
    frames.append(image_tensor(torch, query_input))
    result = torch.stack(frames, dim=1).unsqueeze(0)
    if tuple(result.shape) != (1, 3, 2 * P + 1, RAW_SIZE, RAW_SIZE):
        raise RuntimeError(f"condition shape differs: {tuple(result.shape)}")
    return result


@contextlib.contextmanager
def suspend_nn_inits(torch):
    names = ("kaiming_uniform_", "uniform_", "normal_", "xavier_uniform_")
    saved = {name: getattr(torch.nn.init, name) for name in names}
    for name in names:
        setattr(torch.nn.init, name, lambda *args, **kwargs: None)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


@contextlib.contextmanager
def exclusive_file_lock(lock_path: Path):
    """Serialize one host-side critical section across spawned XLA ranks."""

    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Engine:
    def __init__(self, source_dir, weights_dir, device, torch_xla, cfg_impl="reference",
                 init_lock_path=None):
        import torch
        from safetensors.torch import load_file
        from tpu_compat import install_import_shims
        install_import_shims()
        sys.path.insert(0, str(source_dir))
        from lib.model.dit import LaVinDiT_3B_2
        from lib.model.vae3d import AutoencoderKLSTVAE
        if cfg_impl not in ("reference", "batched"):
            raise ValueError(f"unsupported CFG implementation: {cfg_impl}")
        self.device, self.torch_xla, self.cfg_impl = device, torch_xla, cfg_impl
        dit_path, vae_path = weights_dir / "lavin_dit.bf16.safetensors", weights_dir / "stvae.bf16.pt"
        lock_path = Path(init_lock_path) if init_lock_path is not None else weights_dir / ".engine_load.lock"
        # Eight simultaneous safetensors loads briefly require ~180 GiB of
        # anonymous memory even though steady-state inference is much smaller.
        # Deserialize one rank at a time, release each temporary state mapping,
        # then move the already-populated modules to their independent devices
        # outside the lock.  This changes only initialization timing; the
        # frozen checksum gate proves the numerical path remains exact.
        with exclusive_file_lock(lock_path):
            self.vae = AutoencoderKLSTVAE(
                tile_sample_min_size=512, tile_sample_min_size_t=129
            ).to(torch.bfloat16)
            payload = torch.load(vae_path, map_location="cpu", weights_only=False)
            missing, unexpected = self.vae.load_state_dict(payload["state_dict"], strict=False)
            del payload
            rejected = [k for k in unexpected if not k.startswith(STVAE_TRAINING_ONLY_PREFIXES)]
            if missing or rejected:
                raise SystemExit(f"STVAE state mismatch missing={missing} unexpected={rejected}")
            saved = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with suspend_nn_inits(torch):
                    self.dit = LaVinDiT_3B_2()
            finally:
                torch.set_default_dtype(saved)
            dit_payload = load_file(str(dit_path), device="cpu")
            missing, unexpected = self.dit.load_state_dict(dit_payload, strict=False)
            del dit_payload
            if missing or unexpected:
                raise SystemExit(f"DiT state mismatch missing={missing} unexpected={unexpected}")
            gc.collect()
        self.vae.requires_grad_(False).eval().to(device)
        self.dit.requires_grad_(False).eval().to(device)
        gc.collect()

    def predict(self, condition):
        import torch
        self.torch_xla.manual_seed(MODEL_SEED, device=self.device)
        condition = condition.to(self.device, dtype=torch.bfloat16).permute(2, 1, 0, 3, 4)
        posterior = self.vae.encode(condition)
        # The official distribution samples on CPU and then copies to the
        # model device. Use an explicit CPU generator so the fixed model seed
        # is identical in single- and multi-process XLA topologies.
        latent_generator = torch.Generator(device="cpu").manual_seed(MODEL_SEED)
        latent_noise = torch.randn(posterior.mean.shape, generator=latent_generator).to(
            self.device, dtype=posterior.mean.dtype
        )
        latent = posterior.mean + posterior.std * latent_noise
        latent = ((latent - 0.032470703125) * 1.08936170213).permute(2, 1, 0, 3, 4)
        b, c, _, h, w = latent.shape
        null = torch.zeros((b, c, 1, h, w), device=self.device, dtype=latent.dtype)
        conds = (latent, torch.cat((null, latent[:, :, -1:]), dim=2))
        self.torch_xla.manual_seed(MODEL_SEED, device=self.device)
        state_generator = torch.Generator(device="cpu").manual_seed(MODEL_SEED)
        state = torch.randn(
            (1, self.dit.in_channels, 1, 64, 64), generator=state_generator
        ).to(self.device, dtype=torch.bfloat16).repeat(2, 1, 1, 1, 1)
        times = torch.tensor(EULER_TIMES_20, device=self.device, dtype=torch.float32)
        def drift(t, value):
            step = torch.ones((value.size(0),), device=self.device) * t
            with torch.autocast("xla", dtype=torch.bfloat16):
                if self.cfg_impl == "reference":
                    return self.dit.forward_with_cfg(value, step, conds=conds, cfg_scale=CFG)
                return self._batched_cfg(value, step, conds)
        from torchdiffeq import odeint
        sample = odeint(drift, state, times, method="euler", atol=[1e-6], rtol=[1e-3])[-1][:1]
        with torch.autocast("xla", dtype=torch.bfloat16):
            decoded = self.vae.decode(sample.to(torch.bfloat16) / 1.08936170213 + 0.032470703125)
        image = ((decoded / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).cpu().squeeze()
        return image.permute(1, 2, 0).numpy()

    def _batched_cfg(self, value, step, conds):
        """Evaluate conditional and null CFG branches in one model call.

        This is algebraically identical to the official two-call method.  It is
        admitted only through the pixel-exact qualification gate because XLA is
        free to choose different batch-size-dependent kernels.
        """
        import torch
        from einops import rearrange

        value = rearrange(
            value,
            "B C F (H pH) (W pW) -> B F H W (pH pW C)",
            pH=self.dit.patch_size[0],
            pW=self.dit.patch_size[1],
        )
        cond, cond_null = conds
        both_cond = torch.cat((cond, cond_null), dim=0)
        both_cond = rearrange(
            both_cond,
            "B C F (H pH) (W pW) -> B F H W (pH pW C)",
            pH=self.dit.patch_size[0],
            pW=self.dit.patch_size[1],
        )
        both = self.dit.forward(value, step, both_cond)
        conditional, unconditional = both.chunk(2, dim=0)
        guided = unconditional + CFG * (conditional - unconditional)
        return torch.cat((guided, guided), dim=0)

def validate_plan(path: Path, phase: str) -> dict:
    plan = json.loads(path.read_text())
    claimed = plan.pop("plan_identity")
    if identity(plan) != claimed:
        raise SystemExit("plan identity differs")
    plan["plan_identity"] = claimed
    if (plan.get("schema") != PLAN_SCHEMA or plan.get("phase") != phase
            or plan.get("p") != P or plan.get("condition") != "real"):
        raise SystemExit("plan is outside the frozen LaVin-DiT P=8 contract")
    expected = (1, 221) if phase == "validation" else (26, 17_511)
    if len(plan["packs"]) != expected[0] or len(plan["episodes"]) != expected[1]:
        raise SystemExit("plan pack/query counts differ")
    if phase == "validation" and (plan["packs"][0]["rel"] != "bench_v2_val"
                                  or plan["packs"][0]["n_episodes"] != 221):
        raise SystemExit("validation plan differs")
    if phase == "test" and any(row["rel"] == "bench_v2_val" for row in plan["packs"]):
        raise SystemExit("test plan contains validation")
    return plan


def validate_validation_gate(path: Path | None, expected_sha256: str | None) -> dict:
    if (path is None or expected_sha256 is None or len(expected_sha256) != 64
            or not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256):
        raise SystemExit("TEST requires the hash-pinned corrected-V3 validation receipt")
    gate = json.loads(path.read_text())
    expected = {
        "schema": VALIDATION_GATE_SCHEMA,
        "status": "PASS",
        "phase": "validation",
        "manifest_sha256": MANIFEST_SHA256,
        "validation_episodes": 221,
        "p": P,
        "readout": READOUT,
        "metric_protocol": CORRECTED_PROTOCOL,
        "release_sha256": CORRECTED_RELEASE,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise SystemExit("validation receipt does not admit this LaVin-DiT TEST")
    plan_identity = gate.get("validation_plan_identity")
    if not isinstance(plan_identity, str) or len(plan_identity) != 64:
        raise SystemExit("validation receipt lacks its phase-plan identity")
    if gate.get("plan_identity") != plan_identity:
        raise SystemExit("validation receipt plan binding differs")
    units_sha = gate.get("units_manifest_sha256")
    packs = gate.get("packs")
    if (not isinstance(units_sha, str) or len(units_sha) != 64 or not isinstance(packs, list)
            or len(packs) != 1 or packs[0].get("pack_rel") != "bench_v2_val"
            or packs[0].get("episodes") != 221):
        raise SystemExit("validation receipt corrected-score coverage differs")
    return gate


def initial_view(pack, plan, expected_episode_keys=None, execution=None):
    return {
        "schema": SCHEMA, "protocol": SCHEMA, "pack_rel": pack["rel"], "p": P,
        "phase": plan["phase"], "condition": "real", "plan_identity": plan["plan_identity"],
        "data_identity": pack["data_identity"], "data_identity_sha256": pack["data_identity_sha256"],
        "expected_episode_keys": (pack["expected_episode_keys"] if expected_episode_keys is None
                                  else expected_episode_keys),
        "source_sha256": plan["source_sha256"], "execution": execution,
        "episodes": {}, "complete": False,
    }


def load_views(results_root, plan, expected_by_rel=None, execution=None):
    views, done = {}, set()
    for pack in plan["packs"]:
        expected_keys = (pack["expected_episode_keys"] if expected_by_rel is None
                         else expected_by_rel.get(pack["rel"], []))
        if not expected_keys and expected_by_rel is not None:
            continue
        root = results_root / "views" / pack["slug"]
        path = root / "results.json"
        view = (json.loads(path.read_text()) if path.exists()
                else initial_view(pack, plan, expected_keys, execution))
        if view.get("plan_identity") != plan["plan_identity"] or view.get("data_identity") != pack["data_identity"]:
            raise SystemExit(f"{path}: resume identity mismatch")
        if (view.get("expected_episode_keys") != expected_keys
                or view.get("execution") != execution):
            raise SystemExit(f"{path}: execution selection mismatch")
        for key, ep in list(view["episodes"].items()):
            descs = ep.get("artifacts", {})
            required = [descs.get("query_input"), descs.get("query_target"), descs.get("raw_prediction"), descs.get("prediction")]
            if any(not d or not (root / d["path"]).is_file() for d in required):
                del view["episodes"][key]
                view["complete"] = False
                continue
            for desc in required:
                if sha256_file(root / desc["path"]) != desc["sha256"]:
                    raise SystemExit(f"{root / desc['path']}: resume artifact hash mismatch")
            done.add(f"{pack['rel']}|{key}")
        atomic_json(path, view)
        views[pack["rel"]] = view
    return views, done


def save_episode(results_root, pack, view, ep, query_input, query_target, raw, seconds, cfg_impl):
    root = results_root / "views" / pack["slug"]
    episode_dir = root / f"{ep['task']}_q{ep['q']}"
    input_path, target_path = episode_dir / "query_input.png", episode_dir / "query_target.png"
    raw_path, pred_path = episode_dir / "raw512.png", episode_dir / "prediction256.png"
    atomic_png(input_path, query_input)
    atomic_png(target_path, query_target)
    atomic_png(raw_path, raw)
    scored = np.asarray(Image.fromarray(raw, "RGB").resize((SCORE_SIZE, SCORE_SIZE), Image.Resampling.LANCZOS), np.uint8)
    atomic_png(pred_path, scored)
    view["episodes"][ep["key"]] = {
        "task": ep["task"], "q": ep["q"], "p": P, "cid": ep["cid"],
        "query_record": ep["query_record"], "demo_records": ep["demo_records"],
        "prediction_complete": True, "complete": True,
        "readouts": {READOUT: {"real": {
            "model_seed": MODEL_SEED, "steps": STEPS, "cfg_scale": CFG, "dtype": "bfloat16",
            "rng_contract": RNG_CONTRACT,
            "cfg_impl": cfg_impl,
            "native_prompt_count": P, "ensemble": False,
            "backend": "PyTorch/XLA TPU-native", "cuda_parity_claimed": False,
            "seconds": seconds, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }}},
        "artifacts": {
            "query_input": png_descriptor(input_path, root), "query_target": png_descriptor(target_path, root),
            "raw_prediction": png_descriptor(raw_path, root), "prediction": png_descriptor(pred_path, root),
            "readouts": {READOUT: {"real": png_descriptor(pred_path, root)}},
        },
    }


def infer(args) -> int:
    plan = validate_plan(args.plan, args.phase)
    validation_gate = None
    if args.phase == "test":
        validation_gate = validate_validation_gate(args.validation_gate, args.validation_gate_sha256)
    if sha256_file(TRUSTED_MANIFEST) != MANIFEST_SHA256:
        raise SystemExit("trusted manifest changed before inference")
    import subprocess
    commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={args.official_repo}", "-C", str(args.official_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != OFFICIAL_COMMIT:
        raise SystemExit("official source commit changed")
    validate_conversion_manifest(args.weights)
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU" or not list(Path("/dev/vfio").glob("*")):
        raise SystemExit("refusing non-TPU execution")
    if "kristine" not in args.expected_tpu.lower():
        authorization = json.loads(Path("/kmh-nfs-ssd-us-mount/code/kristine/lvm/eval_runs/2026-10-24/lavin_dit_bench_v2_native_p8_v4_kristine/recovery_retry18/AUTHORIZATION.json").read_text())
        if (authorization.get("approved") is not True
                or authorization.get("tpu") != args.expected_tpu
                or authorization.get("user_message") != "AUTHORIZE PYTHON PROCESS ON " + args.expected_tpu
                or args.expected_tpu != "kmh-tpuvm-v5p-32-kaiminghe-ef3404"):
            raise SystemExit("exact-resource Python authorization is missing")
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    if args.compile_cache is not None:
        args.compile_cache.mkdir(parents=True, exist_ok=True)
        xr.initialize_cache(str(args.compile_cache), readonly=args.compile_cache_readonly)
    backend = str(xr.device_type())
    if backend.upper() != "TPU" or int(xr.global_runtime_device_count()) != args.expected_devices:
        raise SystemExit(f"refusing backend={backend} devices={xr.global_runtime_device_count()}")
    device = torch_xla.device()
    process_count, process_index = int(xr.process_count()), int(xr.process_index())
    if (args.worker_world_size != process_count or args.worker_rank != process_index
            or not 0 <= args.worker_rank < args.worker_world_size):
        raise SystemExit(
            f"worker topology differs expected={args.worker_rank}/{args.worker_world_size} "
            f"actual={process_index}/{process_count}"
        )
    if args.episode_key is not None:
        selected = [ep for ep in plan["episodes"] if ep["key"] == args.episode_key]
        if len(selected) != 1:
            raise SystemExit(f"qualification episode key differs: {args.episode_key}")
    else:
        selected = [ep for position, ep in enumerate(plan["episodes"])
                    if position % args.worker_world_size == args.worker_rank]
    expected_by_rel = {}
    for ep in selected:
        expected_by_rel.setdefault(ep["pack_rel"], []).append(ep["key"])
    execution = {
        "sampler": args.sampler, "cfg_impl": args.cfg_impl, "worker_rank": args.worker_rank,
        "worker_world_size": args.worker_world_size, "episode_key": args.episode_key,
    }
    evidence = {
        "backend": backend, "device": str(device), "device_count": int(xr.global_runtime_device_count()),
        "hostname": socket.gethostname(), "expected_tpu": args.expected_tpu, "expected_zone": args.expected_zone,
        "backend_label": plan["backend_label"], "cuda_parity_claimed": False,
        "process_count": process_count, "process_index": process_index,
        "sampler": args.sampler, "cfg_impl": args.cfg_impl,
        "rng_contract": RNG_CONTRACT,
        "cpu_affinity": list(getattr(args, "cpu_affinity", [])),
        "execution_source_sha256": source_hashes(args.official_repo),
        "frozen_plan_source_sha256_is_original_lineage": True,
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.results_root / "backend.json", evidence)
    views, done = load_views(args.results_root, plan, expected_by_rel, execution)
    engine = Engine(
        args.official_repo,
        args.weights,
        device,
        torch_xla,
        cfg_impl=args.cfg_impl,
        init_lock_path=args.compile_cache.parent / "engine_load.lock",
    )
    packs = PackCache(plan)
    by_rel = {row["rel"]: row for row in plan["packs"]}

    def predict_episode(ep, results_root, view):
        demo_inputs, demo_targets, query_input, query_target = reconstruct(ep, packs)
        started = time.monotonic()
        # PyTorch/XLA v4 still mutates internal version counters while tracing;
        # inference_mode tensors reject that mutation. no_grad preserves the
        # proven numerical path without retaining autograd state.
        with torch.no_grad():
            raw = engine.predict(condition_tensor(torch, demo_inputs, demo_targets, query_input))
        if raw.shape != (RAW_SIZE, RAW_SIZE, 3) or raw.dtype != np.uint8:
            raise SystemExit(f"{ep['episode_id']}: raw output contract differs: {raw.shape} {raw.dtype}")
        pack = by_rel[ep["pack_rel"]]
        save_episode(
            results_root, pack, view, ep, query_input, query_target, raw,
            time.monotonic() - started, args.cfg_impl,
        )
        atomic_json(results_root / "views" / pack["slug"] / "results.json", view)

    qualification_key = getattr(args, "inline_qualification_key", None)
    if qualification_key is not None:
        qualification_root = Path(args.inline_qualification_root)
        qualification_matches = [ep for ep in plan["episodes"] if ep["key"] == qualification_key]
        if len(qualification_matches) != 1:
            raise SystemExit(f"inline qualification episode differs: {qualification_key}")
        qualification_ep = qualification_matches[0]
        qualification_execution = {
            "sampler": args.sampler,
            "cfg_impl": args.cfg_impl,
            "worker_rank": args.worker_rank,
            "worker_world_size": args.worker_world_size,
            "episode_key": qualification_key,
            "inline_gate": True,
        }
        qualification_views, qualification_done = load_views(
            qualification_root,
            plan,
            {qualification_ep["pack_rel"]: [qualification_key]},
            qualification_execution,
        )
        qualification_token = f"{qualification_ep['pack_rel']}|{qualification_key}"
        if qualification_token not in qualification_done:
            qualification_view = qualification_views[qualification_ep["pack_rel"]]
            # Compiling the full reference graph on all ranks at once
            # overloads this v5e host and TPU runtime even after CPU affinity
            # partitioning. Serialize only each rank's first compile/infer;
            # once all exact hashes pass, their already-warm processes
            # continue with the validation partition in parallel.
            compile_lock = Path(args.inline_qualification_compile_lock)
            print(f"QUALIFICATION_COMPILE_WAIT rank={args.worker_rank}", flush=True)
            with exclusive_file_lock(compile_lock):
                print(f"QUALIFICATION_COMPILE_START rank={args.worker_rank}", flush=True)
                predict_episode(qualification_ep, qualification_root, qualification_view)
                print(f"QUALIFICATION_COMPILE_DONE rank={args.worker_rank}", flush=True)
            qualification_view["complete"] = True
            atomic_json(
                qualification_root / "views" / qualification_ep["pack_slug"] / "results.json",
                qualification_view,
            )
        args.inline_qualification_callback()
        # Rank zero owns the frozen qualification episode in the normal
        # modulo partition. Reuse its already-verified artifacts so the gate
        # costs one warm-up episode rather than two.
        if qualification_token not in done and qualification_token in {
            f"{ep['pack_rel']}|{ep['key']}" for ep in selected
        }:
            pack = by_rel[qualification_ep["pack_rel"]]
            qualification_view_root = (
                qualification_root / "views" / qualification_ep["pack_slug"]
            )
            qualification_view = json.loads(
                (qualification_view_root / "results.json").read_text()
            )
            qualification_result = qualification_view["episodes"][qualification_key]
            episode_dir = Path(
                qualification_result["artifacts"]["prediction"]["path"]
            ).parent
            normal_view_root = args.results_root / "views" / qualification_ep["pack_slug"]
            shutil.copytree(
                qualification_view_root / episode_dir,
                normal_view_root / episode_dir,
                dirs_exist_ok=False,
            )
            views[qualification_ep["pack_rel"]]["episodes"][qualification_key] = (
                qualification_result
            )
            atomic_json(normal_view_root / "results.json", views[qualification_ep["pack_rel"]])
            done.add(qualification_token)

        # Rank zero performs the artifact reuse above while the other ranks
        # have no equivalent filesystem work. Do not let those ranks resume
        # TPU execution early: after the first production attempt passed the
        # checksum gate, a child exited while ranks were crossing this exact
        # boundary. Keep all PJRT ranks aligned before validation continues.
        post_gate_tag = plan["plan_identity"][:20]
        print(f"POST_GATE_WAIT rank={args.worker_rank}", flush=True)
        xm.rendezvous(f"lavin_post_gate_ready_{post_gate_tag}")
        print(f"POST_GATE_READY rank={args.worker_rank}", flush=True)

    for ordinal, ep in enumerate(selected, 1):
        token = f"{ep['pack_rel']}|{ep['key']}"
        if token in done:
            continue
        pack, view = by_rel[ep["pack_rel"]], views[ep["pack_rel"]]
        predict_episode(ep, args.results_root, view)
        view["complete"] = len(view["episodes"]) == pack["n_episodes"]
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
        done.add(token)
        print(f"[{ordinal}/{len(selected)}] {ep['episode_id']}", flush=True)
    for pack in plan["packs"]:
        expected_keys = expected_by_rel.get(pack["rel"], [])
        if not expected_keys:
            continue
        view = views[pack["rel"]]
        if set(view["episodes"]) != set(expected_keys):
            raise SystemExit(f"{pack['rel']}: terminal coverage differs")
        view["complete"] = True
        atomic_json(args.results_root / "views" / pack["slug"] / "results.json", view)
    atomic_json(args.results_root / "INFERENCE_COMPLETE.json", {
        "schema": "lavin_dit_bench_v2_native_p8_complete_v1", "phase": args.phase,
        "packs": len(expected_by_rel), "episodes": len(selected),
        "plan_identity": plan["plan_identity"],
        "execution": execution,
        "validation_gate_sha256": args.validation_gate_sha256 if validation_gate is not None else None,
        "validation_plan_identity": validation_gate["validation_plan_identity"] if validation_gate is not None else None,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("prepare-plan")
    plan.add_argument("--phase", choices=("validation", "test"), required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--pack-cache", type=Path, required=True)
    plan.add_argument("--official-repo", type=Path, required=True)
    run = subs.add_parser("infer")
    run.add_argument("--phase", choices=("validation", "test"), required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--results-root", type=Path, required=True)
    run.add_argument("--official-repo", type=Path, required=True)
    run.add_argument("--weights", type=Path, required=True)
    run.add_argument("--expected-tpu", required=True)
    run.add_argument("--expected-zone", required=True)
    run.add_argument("--expected-devices", type=int, required=True)
    run.add_argument("--sampler", choices=("reference",), default="reference")
    run.add_argument("--cfg-impl", choices=("reference", "batched"), default="reference")
    run.add_argument("--worker-rank", type=int, default=0)
    run.add_argument("--worker-world-size", type=int, default=1)
    run.add_argument("--episode-key")
    run.add_argument("--compile-cache", type=Path)
    run.add_argument("--compile-cache-readonly", action="store_true")
    run.add_argument("--validation-gate", type=Path)
    run.add_argument("--validation-gate-sha256")
    return parser.parse_args()


def main():
    args = parse_args()
    return prepare_plan(args) if args.command == "prepare-plan" else infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
