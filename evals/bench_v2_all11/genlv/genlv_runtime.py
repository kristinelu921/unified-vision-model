"""Three-input GenLV execution; no query targets, scores, or text interface.

This module wraps the pinned released architecture. It does not modify its
weights, operations, sampling recipe, or output quantization.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "09e110d8c0781ba40848b49bf40c759daaa5eb1a6d4a9c959f5a75877d467ccc"
MODEL_SHA256 = "335170a288a224b181b90b579d4321af749a4da7e3f1d8afc181f10cbbc29bac"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Inputs:
    demo_input: np.ndarray
    demo_target: np.ndarray
    query_input: np.ndarray

    def __post_init__(self):
        for value in (self.demo_input, self.demo_target, self.query_input):
            if value.dtype != np.uint8 or value.shape != (256, 256, 3):
                raise ValueError("Only exact uint8 256x256 RGB authorized inputs are accepted")


def preprocess(rows):
    """Exactly the frozen producer's RGB->BGR /255 CHW conversion."""
    return tuple(np.ascontiguousarray(np.stack([
        getattr(row, name)[..., ::-1].astype(np.float32) / np.float32(255.0)
        for row in rows]).transpose(0, 3, 1, 2))
        for name in ("demo_input", "demo_target", "query_input"))


def postprocess(output):
    import torch
    value = torch.clip(output.detach().cpu() * 255.0, 0.0, 255.0).to(torch.int32).to(torch.uint8)
    return value.permute(0, 2, 3, 1).numpy()[..., ::-1].copy()


def load_cpu_model(checkpoint, *, mmap=True):
    import torch
    checkpoint = Path(checkpoint)
    if checkpoint.stat().st_size != 2457130637 or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint identity mismatch")
    model_path = ROOT / "source/official/extended_version/models/xrestormer_prompt_crossattn_wores.py"
    if sha256_file(model_path) != MODEL_SHA256:
        raise ValueError("Official model source identity mismatch")
    sys.path.insert(0, str(ROOT / "source/baseline"))
    from genlv_compat import install
    install()
    sys.path.insert(0, str(ROOT / "source/official/extended_version"))
    from models.xrestormer_prompt_crossattn_wores import xrestormer_prompt_crossattn_huge
    torch.manual_seed(0)
    model = xrestormer_prompt_crossattn_huge()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=mmap)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval().requires_grad_(False)
    return model


def predict_raw(model, arrays, device="cpu"):
    import torch
    tensors = [torch.from_numpy(v).to(device) for v in arrays]
    with torch.no_grad():
        # The released model's fourth slot is an unused query-input placeholder.
        # Its visual_tokens argument remains None; no target or loss is supplied.
        _, output, _ = model(imgs=[*tensors, tensors[2]], input_is_list=True)
    return output


class PromptCache:
    """Bounded cache of immutable encoder features for an exact whole batch.

    Whole-batch keys preserve the reference encoder's batch shape. There is no
    row-wise regrouping, adapted state, query target, or final prediction cache.
    This conservative form may have no hits on the actual validation schedule.
    """
    def __init__(self, model, identity, max_entries=8):
        required = {"checkpoint", "source", "adapter", "environment", "configuration",
                    "precision", "backend", "preprocessing", "model_seed", "episode_seed"}
        if set(identity) != required or max_entries < 1:
            raise ValueError("Incomplete cache dependency identity")
        self.model, self.encoder = model, model.Prompt_KV_Encoder
        self.identity = canonical_sha(identity)
        self.max_entries = max_entries
        self.values = OrderedDict()
        self.hits = self.misses = 0
        self.versions = tuple(p._version for p in self.encoder.parameters())
        self.original_forward = self.encoder.forward

    def install(self):
        self.encoder.forward = self.forward
        return self

    def close(self):
        self.encoder.forward = self.original_forward
        self.values.clear()

    def forward(self, demo_input, demo_target):
        if self.model.training or tuple(p._version for p in self.encoder.parameters()) != self.versions:
            self.values.clear()
            raise ValueError("Prompt encoder must remain frozen and in eval mode")
        # CPU probe only. TPU cache requires separate device-resident identity and
        # compile/hit-path qualification before it may be selected.
        if demo_input.device.type != "cpu" or demo_target.device.type != "cpu":
            raise ValueError("Prompt cache is CPU-qualified only")
        h = hashlib.sha256(self.identity.encode())
        for value in (demo_input, demo_target):
            h.update(str((tuple(value.shape), str(value.dtype), str(value.device))).encode())
            h.update(value.detach().contiguous().numpy().tobytes())
        key = h.hexdigest()
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        value = tuple(v.detach().clone() for v in self.original_forward(demo_input, demo_target))
        self.values[key] = value
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)
        return value


def prefetched_batches(batches):
    """At most one CPU preprocessing batch ahead; fixed order and shape."""
    iterator = iter(batches)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = next(iterator, None)
        if first is None:
            return
        future = pool.submit(preprocess, first)
        for following in iterator:
            current = future.result()
            future = pool.submit(preprocess, following)
            yield current
        yield future.result()
