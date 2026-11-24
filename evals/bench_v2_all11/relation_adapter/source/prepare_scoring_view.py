"""Scorer-side bridge; query targets are first read after inference completes.

The original plan's episodes are preserved, while the new execution gets a new
plan/run identity that records the candidate, source, and fresh inference receipt.
This script is not imported by the inference worker.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from run_target_free import ROOT, load_baseline, sha, candidate_source_identity
from prepare_inference_inputs import PLAN_SHA, META_SHA, DATA_SHA
from execution_contract import validate_qualification
from target_free_episode import pixels_sha256
from instruction_policy import verify_receipt_policy


def verify_handoff(predictions, qualification_path):
    """Authenticate every fresh prediction before any target is loaded."""
    timing_path = predictions / 'timing_receipt.json'
    timing = json.loads(timing_path.read_text())
    if (timing.get('status') != 'INFERENCE_COMPLETE_SCORING_REQUIRED' or timing.get('episodes') != 221
            or timing.get('fresh_predictions') is not True or timing.get('source_identity') != candidate_source_identity()):
        raise ValueError('complete fresh candidate inference required')
    verify_receipt_policy(timing)
    expected = {f'prediction_{i:03d}.png' for i in range(221)}
    if {x.name for x in predictions.glob('prediction_*.png')} != expected:
        raise ValueError('prediction coverage differs')
    artifacts = timing.get('outputs', [])
    if (len(artifacts) != 221 or [x['ordinal'] for x in artifacts] != list(range(221))
            or {x['path'] for x in artifacts} != expected or len(timing.get('timings', [])) != 221):
        raise ValueError('prediction artifact receipt coverage differs')
    if sha(qualification_path) != timing.get('qualification_sha256'):
        raise ValueError('qualification hash differs from inference')
    qual = json.loads(qualification_path.read_text())
    validate_qualification(qual, candidate=timing['candidate'], tpu_name=timing['tpu_name'],
                           runtime_identity=timing['runtime_identity'])
    for record in artifacts:
        path = predictions / record['path']
        if path.is_symlink() or sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('prediction artifact bytes differ')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (256, 256) or image.format != 'PNG':
                raise ValueError('prediction output contract differs')
            if pixels_sha256(np.asarray(image)) != record['pixels_sha256']:
                raise ValueError('prediction pixel identity differs')
    return timing, timing_path


def main():
    p = argparse.ArgumentParser()
    for name in ['frozen-plan', 'pack', 'predictions', 'qualification', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    for path, expected in [(args.frozen_plan, PLAN_SHA), (args.pack / 'samples.json', META_SHA), (args.pack / 'samples.npz', DATA_SHA)]:
        if sha(path) != expected: raise SystemExit('frozen benchmark identity differs')
    timing, timing_path = verify_handoff(args.predictions, args.qualification)
    baseline = load_baseline()
    plan = json.loads(args.frozen_plan.read_text())
    old_identity = plan.pop('plan_identity')
    plan['autoresearch'] = {'baseline_plan_identity': old_identity, 'candidate': timing['candidate'],
                            'source_identity': timing['source_identity'], 'timing_receipt_sha256': sha(timing_path),
                            'qualification_sha256': sha(args.qualification), 'fresh_predictions': True,
                            'instruction_policy': timing['instruction_policy'], 'condition': timing['condition'], 'task_prompts': timing['task_prompts'],
                            'label': 'target-isolated ' + timing['condition'] + (' with fixed relation K/V cache' if timing['candidate'] == 'C1' else '')}
    plan['source_sha256']['runner'] = sha(ROOT / 'source/run_target_free.py')
    plan['prompt_role'] = 'Inherited baseline template; active instructions are bound in autoresearch.instruction_policy'
    plan['packs'][0]['local_dir'] = str(args.pack.resolve())
    plan['plan_identity'] = baseline.identity(plan)
    args.out.mkdir(parents=True, exist_ok=False)
    baseline.atomic_json(args.out / 'plan.json', plan)
    packs = baseline.PackCache(plan)
    pack = plan['packs'][0]
    view = baseline.initial_view(pack, plan)
    for ordinal, ep in enumerate(plan['episodes']):
        # This target access is isolated from model inference and caches.
        _, _, query_input, query_target = baseline.reconstruct(ep, packs)
        with Image.open(args.predictions / f'prediction_{ordinal:03d}.png') as image:
            if image.mode != 'RGB' or image.size != (256, 256) or image.format != 'PNG':
                raise ValueError('prediction output contract differs')
            prediction = np.array(image, dtype=np.uint8)
        baseline.save_episode(args.out, pack, view, ep, query_input, query_target, prediction,
                              timing['timings'][ordinal]['inference_seconds'])
        readout = view['episodes'][ep['key']]['readouts'][baseline.READOUT]['real']
        readout['autoresearch_candidate'] = timing['candidate']
        readout['condition'] = timing['condition']
        readout['prompt'] = timing['task_prompts'].get(ep['task'], baseline.PROMPT)
        readout['prompt_2'] = readout['prompt']
        readout['prompt_status'] = 'Frozen public task instruction; benchmark adaptation' if timing['task_prompts'] else 'Benchmark-neutral generic instruction'
        readout['backend'] = timing['backend'] + '/SPMD-FSDP'
        readout['instruction_policy'] = timing['instruction_policy']
    view['complete'] = True
    baseline.atomic_json(args.out / 'views/bench_v2_val/results.json', view)
    baseline.atomic_json(args.out / 'VAL_INFERENCE_COMPLETE.json', {
        'schema': 'relation_adapter_bench_v2_validation_inference_complete_v1', 'episodes': 221,
        'plan_identity': plan['plan_identity'], 'qualification_receipt_sha256': sha(args.qualification),
        'candidate': timing['candidate'], 'fresh_predictions_receipt_sha256': sha(timing_path)})
    print(json.dumps({'episodes': 221, 'candidate': timing['candidate'], 'new_plan_identity': plan['plan_identity']}))


if __name__ == '__main__': main()
