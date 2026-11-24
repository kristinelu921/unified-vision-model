"""Prepared TPU worker for fixed C0/C1 recipes; no target or score interface.

Not launch-qualified until the same-backend synthetic receipt is supplied.
Both v5p-16 workers must run with the same staged, hash-verified assets and inputs.
Only worker zero writes predictions. Cold staging is timed by the coordinator.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from cache_identity import ConditioningIdentity
from finite_output import finite_cpu, install_decode_guard, float_digest
from runtime_loader import load_runtime
from instruction_policy import policy, prompt_map_for, select_embeddings
from conditioning_scope import prepared_relation
from target_free_episode import pixels_sha256
from execution_contract import (source_identity as contract_source_identity, production_environment,
                                runtime_contract, check_official_source, validate_qualification, require_node_local_paths,
                                require_authorized_resource, hardware_profile)
from run_telemetry import xla_snapshot, snapshot_delta, seal_prediction

ROOT = Path(__file__).resolve().parents[1]
# The guarded launcher supplies the current allocation deadline.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_deadline():
    deadline = float(os.environ['EVALUATION_DEADLINE'])
    if not 0 < deadline < float('inf') or time.time() >= deadline:
        raise TimeoutError('RelationAdapter task deadline reached')


def candidate_source_identity():
    return contract_source_identity()


def load_baseline():
    p = ROOT / 'source/baseline_archive/relation_adapter_bench_v2.py'
    if sha(p) != 'c6a7b39db6a0aaf1e96f7846c0a20efe37fe889010d9ebde8c5f4b0dcf816be2':
        raise ValueError('archived baseline source changed')
    spec = importlib.util.spec_from_file_location('baseline_model_components', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', choices=['C0', 'C1'], required=True)
    p.add_argument('--condition', choices=['generic_text','task_text_alpha1'], default='task_text_alpha1')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--flux-dir', type=Path, required=True)
    p.add_argument('--adapter-dir', type=Path, required=True)
    p.add_argument('--siglip-dir', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--qualification-sha256', required=True)
    p.add_argument('--tpu-name', required=True)
    p.add_argument('--license-ack', type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    check_deadline()
    require_authorized_resource(args.tpu_name)
    if os.environ.get('PJRT_DEVICE') != 'TPU': raise SystemExit('TPU required')
    require_node_local_paths(args)
    if sha(args.inputs) != args.inputs_sha256: raise SystemExit('input manifest hash mismatch')
    manifest = json.loads(args.inputs.read_text())
    if manifest.get('schema') != 'relationadapter_allowed_inputs_v2' or manifest.get('condition') != 'shared_images_with_public_task':
        raise SystemExit('unsupported condition')
    if len(manifest['episodes']) != 221 or [x['ordinal'] for x in manifest['episodes']] != list(range(221)):
        raise SystemExit('complete frozen validation coverage required')
    if sha(args.qualification) != args.qualification_sha256: raise SystemExit('qualification hash mismatch')
    qualification = json.loads(args.qualification.read_text())
    source_identity = candidate_source_identity()
    environment = production_environment()
    execution = runtime_contract(environment, args.condition)
    runtime_identity = hash_value(execution)
    validate_qualification(qualification, candidate=args.candidate, tpu_name=args.tpu_name,
                           runtime_identity=runtime_identity)
    baseline = load_baseline()
    baseline.verify_license_ack(args.license_ack)
    for n, folder in [('FLUX.1-dev', args.flux_dir), ('RelationAdapter', args.adapter_dir), ('siglip-so400m-patch14-384', args.siglip_dir)]:
        baseline.verify_asset_tree(folder, baseline.ASSET_MANIFESTS[n])
    check_official_source(args.source_dir)
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    import torch_xla.debug.metrics as metrics
    xr.initialize_cache(str(args.cache_dir / runtime_identity), readonly=False)
    xr.use_spmd()
    count, local = int(xr.global_runtime_device_count()), int(xr.addressable_runtime_device_count())
    hardware = hardware_profile()
    if xr.device_type() != 'TPU' or (count, local) != (hardware['global_devices'], hardware['addressable_devices']):
        raise SystemExit('qualified declared topology required')
    if args.candidate == 'C1':
        import attention_processor_cached
        sys.modules['attention_processor'] = attention_processor_cached
    mesh = xs.Mesh(np.arange(count), (count,), ('fsdp',))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    writer = xr.process_index() == 0
    worker_receipts = args.out.parent / ('telemetry_' + args.out.name)
    worker_receipts.mkdir(parents=True, exist_ok=True)
    # Fresh means no prediction reuse, including after a failed attempt.
    if writer: args.out.mkdir(parents=True, exist_ok=False)
    telemetry = lambda: xla_snapshot(metrics, lambda: xm.get_memory_info(device))
    before_load = telemetry()
    load_start = time.monotonic()
    prompt_map = prompt_map_for(args.condition)
    if prompt_map and any(row.get('task') not in prompt_map for row in manifest['episodes']):
        raise ValueError('missing or unknown public task label')
    runtime = load_runtime(baseline, args, device, mesh, task_prompts=prompt_map)
    report_nonfinite = lambda detail: baseline.atomic_json(worker_receipts / f"nonfinite_worker_{xr.process_index()}.json", detail)
    finite_guard = install_decode_guard(runtime["pipe"].vae, report_nonfinite)
    xm.mark_step(wait=True)
    loading_seconds = time.monotonic() - load_start
    after_load = telemetry()
    processors = [m for m in runtime['pipe'].transformer.modules() if hasattr(m, 'bind_relation_cache')]
    if args.candidate == 'C1' and len(processors) != 57: raise SystemExit('adapter layer count differs')
    environment_sha = hash_value(environment)
    official_sha = sha(ROOT / 'official/source_manifest.json')
    weights_sha = hash_value(baseline.ASSET_MANIFESTS)
    config_sha = hash_value({'steps': 24, 'guidance': 3.5, 'alpha': 1.0, 'P': 1, 'seed': 1000, 'size': 256})
    prompt_sha = hash_value({'task_prompts': prompt_map, 'condition': args.condition, 'prompt': baseline.PROMPT, 'prompt_2': baseline.PROMPT, 'max_sequence_length': 512,
                             'tokenizers': baseline.FLUX_REVISION})
    receipt = {'instruction_policy':policy(args.condition), 'condition':args.condition, 'task_prompts':prompt_map, 'prompt_sha256':prompt_sha, 'candidate': args.candidate, 'fresh_predictions': True, 'episodes': 0, 'load_seconds': loading_seconds,
               'source_identity': source_identity, 'input_manifest_sha256': args.inputs_sha256,
               'runtime_identity': runtime_identity, 'execution_contract': execution,
               'finite_output_guard': finite_guard, 'block_wrapping': runtime['block_wrapping'],
               'qualification_sha256': args.qualification_sha256, 'tpu_name': args.tpu_name,
               'timings': [], 'outputs': [], 'backend': 'TPU-' + hardware['accelerator'], 'cuda_parity': False,
               'hardware': hardware,
               'telemetry_before_load': before_load, 'telemetry_after_load': after_load,
               'loading_metrics': snapshot_delta(before_load, after_load)}
    try:
        for row in manifest['episodes']:
            check_deadline()
            if time.monotonic() - start >= hardware['max_accelerator_seconds_per_trial']:
                raise TimeoutError('per-run hardware budget reached')
            file = args.inputs.parent / row['path']
            if Path(row['path']).name != row['path'] or sha(file) != row['sha256']: raise ValueError('input artifact differs')
            io_start = time.monotonic()
            with np.load(file, allow_pickle=False) as data:
                if set(data.files) != {'demo_input', 'demo_output', 'query_input'}: raise ValueError('input schema differs')
                before, after, query = (np.array(data[k]) for k in ['demo_input', 'demo_output', 'query_input'])
            for array, key in [(before, 'demo_input_sha256'), (after, 'demo_output_sha256'), (query, 'query_input_sha256')]:
                if array.dtype != np.uint8 or array.shape != (256, 256, 3) or pixels_sha256(array) != row[key]:
                    raise ValueError('input numerical identity differs')
            inference_start = time.monotonic()
            relation = baseline.relation_embedding(runtime, Image.fromarray(before), Image.fromarray(after), device)
            identity = ConditioningIdentity(weights_sha, official_sha, source_identity, environment_sha, config_sha,
                                             sha(ROOT / 'source/baseline_archive/relation_adapter_bench_v2.py'), prompt_sha,
                                             'TPU-' + hardware['accelerator'] + '/torch-xla2.8/bfloat16', (row['demo_input_sha256'],),
                                             (row['demo_output_sha256'],), row['query_input_sha256'], 1000)
            from contextlib import nullcontext
            scope = prepared_relation(processors, relation, identity, heads=24, head_dim=128) if args.candidate == 'C1' else nullcontext(relation)
            active_prompt, active_pooled = select_embeddings(runtime, args.condition, row['task'])
            with scope as conditioning:
                torch_xla.manual_seed(1000, device=device)
                with torch.no_grad():
                    result = runtime['pipe'](prompt=None, prompt_embeds=active_prompt,
                        pooled_prompt_embeds=active_pooled, image_emb=conditioning,
                        condition_image=Image.fromarray(query), height=256, width=256,
                        guidance_scale=3.5, num_inference_steps=24, generator=None, output_type='pt').images
                xm.mark_step()
                raw = finite_cpu(result[0], stage='postprocessed_image', report=report_nonfinite)
                pixels = (raw.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            inference_end = time.monotonic()
            if writer:
                output_path = args.out / f"prediction_{row['ordinal']:03d}.png"
                baseline.atomic_png(output_path, pixels)
                receipt['outputs'].append(seal_prediction(output_path, pixels_sha256(pixels), row['ordinal']))
            receipt['timings'].append({'io_seconds': inference_start - io_start,
                                       'inference_seconds': inference_end - inference_start,
                                       'write_seconds': time.monotonic() - inference_end})
            receipt['episodes'] += 1
            # Aggregate counters only. Current memory samples remain explicitly
            # distinct from an unsupported device peak measurement.
            receipt['telemetry_latest'] = telemetry()
            receipt['wall_seconds_worker'] = time.monotonic() - start
            if writer: baseline.atomic_json(args.out / 'progress_metadata.json', receipt)
        receipt['status'] = 'INFERENCE_COMPLETE_SCORING_REQUIRED'
    finally:
        receipt['wall_seconds_worker'] = time.monotonic() - start
        receipt['inference_metrics'] = snapshot_delta(after_load, receipt.get('telemetry_latest', after_load))
        receipt['worker_observed_slice_chip_hours'] = receipt['wall_seconds_worker'] * count / 3600
        receipt['chip_hours_scope'] = 'whole slice during worker phase; do not sum workers or call this cold end-to-end'
        if writer: baseline.atomic_json(args.out / 'timing_receipt.json', receipt)
        baseline.atomic_json(worker_receipts / f'worker_{xr.process_index()}.json', {
            'worker_index': xr.process_index(), 'global_devices': count, 'addressable_devices': local,
            'runtime_identity': runtime_identity, 'load_seconds': loading_seconds,
            'wall_seconds_worker': receipt['wall_seconds_worker'],
            'after_load': after_load, 'final': receipt.get('telemetry_latest', after_load)})
    if writer: print(json.dumps({'status': receipt['status'], 'episodes': receipt['episodes']}))


if __name__ == '__main__': main()
