"""Target-free inference schema, content identities and bounded task execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

MODEL_REVISION = '60ea7b6cc48d08487f95d06ba6ce59e15b619efd'
SOURCE_REVISION = '8ecd2d126d9f9e17b80d9655bf8ed2df2e93f660'
PACK_ID = '8e139f91b083e0ae73b2ed81d8e235460ac2f37d'
MANIFEST_SHA256 = '89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d'
NOISE_NAMESPACE = 'prompt_diffusion_bench_v2_v1'
DEADLINE = '2026-10-30T22:00:00+00:00'

def launch_resource_is_authorized(launch, now=None):
    return (launch.get('name') == 'kmh-tpuvm-v5e-4-us-west4-spot-jaustin-rploh0'
        and launch.get('zone') == 'us-west4-a'
        and launch.get('run_id') == 'promptdiffusion_range_recovery_1958'
        and launch.get('exact_resource_authorized') is True
        and launch.get('authorization_phrase') == 'AUTHORIZE PYTHON PROCESS ON kmh-tpuvm-v5e-4-us-west4-spot-jaustin-rploh0')

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def array_sha256(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()

def noise_seed(pack_id, query_record):
    material = '|'.join((NOISE_NAMESPACE, MANIFEST_SHA256, pack_id, query_record))
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'little') & 0x7fff_ffff_ffff_ffff

def normalize_rgb(value, convention='zero_one'):
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError('RGB input must be uint8 NHWC')
    value = value.astype(np.float32).transpose(0, 3, 1, 2) / np.float32(255.)
    if convention == 'minus_one_one':
        value = np.float32(2.) * value - np.float32(1.)
    elif convention != 'zero_one':
        raise ValueError('unknown normalization')
    return value

@dataclass(frozen=True)
class Recipe:
    candidate: str = 'S1'
    resolution: int = 256
    steps: int = 50
    guidance: float = 7.5
    eta: float = 0.
    conditioning_scale: float = 1.
    precision: str = 'bfloat16'
    batch_size: int = 4
    normalization: str = 'zero_one'
    cache_conditioning: bool = False
    dynamic_coefficients: bool = False

    def validate(self):
        if (self.resolution, self.steps, self.guidance, self.eta, self.conditioning_scale) != (256, 50, 7.5, 0., 1.):
            raise ValueError('unregistered scientific recipe')
        if self.candidate not in ('B0','S1','S2','S3','A1','T1'):
            raise ValueError('unregistered candidate')
        if self.batch_size != (8 if self.candidate == 'S3' else 4):
            raise ValueError('unregistered batch size')
        expected = 'minus_one_one' if self.candidate == 'A1' else 'zero_one'
        if self.normalization != expected:
            raise ValueError('normalization and candidate label differ')
        if self.cache_conditioning != (self.candidate in ('S2','S3')):
            raise ValueError('conditioning cache and candidate label differ')
        if self.dynamic_coefficients != (self.candidate in ('S2','S3')):
            raise ValueError('coefficient mode and candidate label differ')
        if self.precision != 'bfloat16':
            raise ValueError('fresh TPU run must use registered precision')
        return asdict(self)

# Deliberately no full plan identity, pack-file hash or target-derived field:
# whole-pack/plan integrity is checked by preparation and the independent scorer.
EPISODE_FIELDS = frozenset({'ordinal','key','task','query_record','seed',
                           'demo_input_sha256','demo_output_sha256','query_input_sha256'})
PAYLOAD_FIELDS = frozenset({'schema','pack_id','source_revision','model_revision',
                           'episode_seed','demonstrations','cid','episodes','inputs_sha256',
                           'input_identity'})

def validate_payload(payload):
    if set(payload) != PAYLOAD_FIELDS:
        raise ValueError('inference payload has unapproved or missing fields')
    if payload['schema'] != 'promptdiffusion_permitted_inputs_v1':
        raise ValueError('payload schema differs')
    if payload['pack_id'] != PACK_ID or payload['model_revision'] != MODEL_REVISION or payload['source_revision'] != SOURCE_REVISION:
        raise ValueError('payload benchmark/model identity differs')
    if (payload['episode_seed'],payload['demonstrations'],payload['cid']) != (42,1,0):
        raise ValueError('episode recipe differs')
    raw = dict(payload); claimed = raw.pop('input_identity')
    if identity(raw) != claimed:
        raise ValueError('permitted input identity differs')
    episodes = payload['episodes']
    if len(episodes) != 221:
        raise ValueError('full validation coverage required')
    counts = {}
    for i, ep in enumerate(episodes):
        if set(ep) != EPISODE_FIELDS or ep['ordinal'] != i:
            raise ValueError('episode contains unapproved fields or order differs')
        if ep['seed'] != noise_seed(payload['pack_id'], ep['query_record']):
            raise ValueError('noise seed differs')
        counts[ep['task']] = counts.get(ep['task'],0) + 1
    if len(counts) != 13 or set(counts.values()) != {17} or len({ep['key'] for ep in episodes}) != 221:
        raise ValueError('task/query coverage differs')

def cache_key(*, stage, inputs, weights, source, environment, recipe, prompts,
              tokenization, seeds):
    """Every caller must provide all identity classes, including unused seeds=[]."""
    dependencies = dict(stage=stage, inputs=inputs, weights=weights, source=source,
        environment=environment, recipe=recipe, prompts=prompts,
        tokenization=tokenization, seeds=seeds)
    if any(value is None for value in dependencies.values()):
        raise ValueError('missing immutable-cache dependency')
    return identity(dependencies)

def remaining_seconds(deadline=DEADLINE):
    return (datetime.fromisoformat(deadline)-datetime.now(timezone.utc)).total_seconds()

def require_time(minimum_seconds=0):
    if remaining_seconds() <= minimum_seconds:
        raise TimeoutError('task deadline reached or insufficient completion margin')

def fresh_output(path):
    path=Path(path)
    path.mkdir(parents=True,exist_ok=False)
    return path


def runtime_identity():
    """Numerical software/flags shared by parity and production processes."""
    import importlib.metadata
    import os
    import sys
    flags = ['XLA_FLAGS','LIBTPU_INIT_ARGS','XLA_USE_BF16','XLA_DOWNCAST_BF16',
             'PJRT_DEVICE','TPU_PROCESS_BOUNDS','TPU_CHIPS_PER_PROCESS_BOUNDS','XLA_USE_SPMD']
    return {'packages': sorted((d.metadata.get('Name',''),d.version)
                              for d in importlib.metadata.distributions()),
            'python':sys.version, 'flags':{k:os.environ.get(k,'') for k in flags}}
