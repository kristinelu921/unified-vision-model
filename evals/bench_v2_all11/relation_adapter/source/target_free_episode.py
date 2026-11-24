"""Reconstruct only authorized model inputs. No query target parameter exists.

The frozen Bench v2 P=1 plan has CID 0 for every episode. At CID 0, demo-first
palette relabeling assigns demonstration labels before any query-only labels;
therefore coding demonstrations alone yields the same demonstration pixels.
This claim is qualified by the synthetic property tests, never validation views.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from bench_episode_v1 import code_episode


def pixels_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


@dataclass(frozen=True)
class InferenceEpisode:
    demo_inputs: tuple[np.ndarray, ...]
    demo_outputs: tuple[np.ndarray, ...]
    query_input: np.ndarray


def build_inputs(demo_inputs, raw_demo_outputs, query_input, *, task, cid=0):
    if cid != 0:
        raise ValueError('only frozen CID 0 is qualified for target-free reconstruction')
    if len(demo_inputs) != 1 or len(raw_demo_outputs) != 1:
        raise ValueError('the frozen RelationAdapter condition requires P=1')
    arrays = [*demo_inputs, *raw_demo_outputs, query_input]
    if any(a.dtype != np.uint8 or a.shape != (256, 256, 3) for a in arrays):
        raise ValueError('only frozen uint8 256x256 RGB inputs are accepted')
    coded = code_episode(raw_demo_outputs, task, cid=0)
    def freeze(a):
        b = np.array(a, copy=True)
        b.flags.writeable = False
        return b
    return InferenceEpisode(tuple(map(freeze, demo_inputs)), tuple(map(freeze, coded)), freeze(query_input))
