"""Frozen numerical identities shared by qualification, inference and scoring."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
from instruction_policy import policy

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE_MANIFEST_SHA = 'ff1287161174b80d32e378cc98fedfcc42bea13f524b758eec4f891032223a79'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_identity():
    names = ['runtime_loader.py', 'finite_output.py', 'run_target_free.py', 'attention_processor_cached.py', 'conditioning_scope.py',
             'cache_identity.py', 'target_free_episode.py', 'prepare_inference_inputs.py',
             'qualify_on_synthetic.py', 'execution_contract.py', 'run_telemetry.py', 'bench_episode_v1.py', 'instruction_policy.py',
             'prepare_scoring_view.py', 'aggregate_validation.py', 'baseline_archive/materialize_corrected_v3_units.py']
    sources = {name: sha(ROOT / 'source' / name) for name in names}
    sources['task_instructions.json'] = sha(ROOT / 'configs/task_instructions.json')
    sources['baseline_runtime.py'] = sha(ROOT / 'source/baseline_archive/relation_adapter_bench_v2.py')
    sources['hardware_profile.json'] = sha(ROOT / 'configs/hardware_profile.json')
    return identity(sources)


def hardware_profile():
    profile = json.loads((ROOT / 'configs/hardware_profile.json').read_text())
    expected = {'v5p-16': ('2x2x2', 2, 8, 4, 7200),
                'v5p-64': ('2x4x4', 8, 32, 4, 1800)}
    actual = tuple(profile.get(k) for k in ['topology', 'workers', 'global_devices',
                                          'addressable_devices', 'max_accelerator_seconds_per_trial'])
    if (profile.get('accelerator') not in expected or actual != expected[profile['accelerator']]
            or profile.get('spot') is not True or profile.get('max_chip_hours_per_trial') != 16
            or profile['global_devices'] * profile['max_accelerator_seconds_per_trial'] != 16 * 3600):
        raise ValueError('undeclared hardware or compute budget')
    return profile


def require_authorized_resource(name):
    auth = json.loads((ROOT / 'configs/resource_authorization.json').read_text())
    if (name != 'kmh-tpuvm-v5p-64-us-east5-spot-jaustin-lqai6p' or auth.get('tpu_name') != name
            or auth.get('authority') != 'Direct user message in this conversation'
            or auth.get('received_current_session') is not True
            or auth.get('run_id') != 'relationadapter_runtime_repair_0010'
            or auth.get('user_message') != 'AUTHORIZE PYTHON PROCESS ON ' + name):
        raise ValueError('Exact current-conversation resource authorization required')


def check_official_source(source_dir):
    manifest = ROOT / 'official/source_manifest.json'
    if sha(manifest) != OFFICIAL_SOURCE_MANIFEST_SHA:
        raise ValueError('official source manifest identity changed')
    for item in json.loads(manifest.read_text()):
        if sha(Path(source_dir) / item['path']) != item['sha256']:
            raise ValueError('pinned official source file changed')


def require_node_local_paths(args):
    prefix = Path('/dev/shm/relationadapter_runtime_repair_0010/relationadapter').resolve()
    for name in ['flux_dir', 'adapter_dir', 'siglip_dir', 'source_dir', 'cache_dir', 'out', 'inputs']:
        value = getattr(args, name, None)
        if value is not None and not Path(value).resolve().is_relative_to(prefix):
            raise ValueError('TPU work must use this run\'s node-local staging: ' + name)


def production_environment():
    # Check every direct pin plus overrides from the archived constraints.
    expected = {}
    for path in [ROOT / 'source/baseline_archive/requirements.lock', ROOT / 'source/baseline_archive/requirements.constraints']:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            name, version = line.split('==', 1)
            name = re.sub(r'\[.*\]', '', name).lower().replace('_', '-')
            expected[name] = version
    installed = {name: importlib.metadata.version(name) for name in expected}
    if installed != expected:
        mismatch = sorted(name for name in expected if installed.get(name) != expected[name])
        raise ValueError('production dependency version mismatch: ' + ', '.join(mismatch))
    # Full inventory captures transitive packages and libtpu, even where the old
    # environment record did not pin them. Qualification and validation must match.
    packages = sorted((d.metadata['Name'].lower().replace('_', '-'), d.version)
                      for d in importlib.metadata.distributions() if d.metadata.get('Name'))
    settings = {key: os.environ.get(key) for key in ['PJRT_DEVICE', 'XLA_FLAGS', 'XLA_USE_BF16',
        'XLA_DOWNCAST_BF16', 'XLA_USE_SPMD', 'XLA_DISABLE_FUNCTIONALIZATION', 'XLA_IR_DEBUG', 'XLA_HLO_DEBUG']}
    return {'direct_versions': installed, 'packages': packages, 'python': platform.python_version(),
            'python_implementation': platform.python_implementation(), 'platform': platform.platform(),
            'numerical_environment': settings,
            'locks': {name: sha(ROOT / 'source/baseline_archive' / name)
                      for name in ['requirements.lock', 'requirements.constraints']}}


def runtime_contract(environment, condition='generic_text'):
    return {'source_identity': source_identity(), 'official_manifest_sha256': OFFICIAL_SOURCE_MANIFEST_SHA,
            'environment': environment, 'hardware': hardware_profile(),
            'weight_manifests': {
                'FLUX.1-dev': '5d974c9898c86e35f0e786b7551a0d2914b37603beb8a940c3b49b96a9a0b892',
                'RelationAdapter': '3d00a444eab4964011a19212c9a5b53cd6bad7413fbd663c4ae2ec60e2e28621',
                'siglip-so400m-patch14-384': 'bd890cec873e059c1989854386b388c1c12db4e66bd11a021af8a4147b9535ff'},
            'config': {'instruction_policy': policy(condition), 'prompt_2': 'same',
                'max_sequence_length': 512, 'steps': 24, 'guidance': 3.5, 'alpha': 1.0,
                'model_seed': 1000, 'samples': 1, 'P': 1, 'condition': 'real',
                'query_shape': [256, 256, 3], 'demo_pad_resize': 'official 512-long-side then SigLIP384',
                'precision': 'bfloat16', 'topology': hardware_profile()['accelerator'] + '/SPMD-FSDP'},
            'compilation_cache_contract': 'This namespace contains executable graphs, not numerical outputs. '
                'XLA graph keys bind operations/constants/shapes/dtypes; input pixels remain dynamic arguments. '
                'Numerical conditioning reuse additionally hashes every permitted input per episode.'}


def validate_qualification(receipt, *, candidate, tpu_name, runtime_identity):
    if (receipt.get('schema') != 'relationadapter_same_backend_qualification_v2'
            or receipt.get('status') != 'PASS' or receipt.get('synthetic_only') is not True
            or receipt.get('candidate') != candidate or receipt.get('source_identity') != source_identity()
            or receipt.get('tpu_name') != tpu_name or receipt.get('all_outputs_exact') is not True
            or receipt.get('runtime_identity') != runtime_identity or receipt.get('fixtures') != 4
            or receipt.get('raw_reference_repeat_exact') is not True
            or receipt.get('finite_output_guard', {}).get('enabled') is not True
            or receipt.get('finite_output_guard', {}).get('checked_decodes', 0) < 8
            or receipt.get('block_wrapping', {}).get('wrapped_blocks') != 57):
        raise ValueError('same-source same-environment same-TPU synthetic qualification is missing')
    if candidate == 'C1' and (receipt.get('C1_speed_gate_passed') is not True or receipt.get('raw_C1_reference_exact') is not True):
        raise ValueError('C1 full-model speed gate has not passed')
