"""Fixed dataset-side exporter. Inference receives only allowed episode inputs.

This program verifies the frozen validation pack without rendering or printing
any example. It never loads a query target by its query index. The separate
unchanged scorer retains access to the original pack and frozen plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from target_free_episode import build_inputs, pixels_sha256

PLAN_SHA = '7f2dcd27982d561ed674ef95235879877a7840e11a5c047db1d7eb28efba3701'
META_SHA = '7a130a55c9928b3509d95d03f3a3c9a63ffb5331c03335b2bee211374bb59322'
DATA_SHA = 'd36685bbd8b95e720f03980b910712cd99d51f4933a4d1e9cf2e118d2235548e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    start = time.monotonic()
    for path, expected in [(args.plan, PLAN_SHA), (args.pack / 'samples.json', META_SHA), (args.pack / 'samples.npz', DATA_SHA)]:
        if sha(path) != expected: raise SystemExit('frozen input asset identity mismatch')
    plan = json.loads(args.plan.read_text())
    if plan['phase'] != 'val' or len(plan['episodes']) != 221: raise SystemExit('validation-only fixed coverage required')
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'relationadapter_allowed_inputs_v2', 'condition': 'shared_images_with_public_task', 'episodes': []}
    # Dataset-side materialization only. Never pass the pack/plan to inference.
    with np.load(args.pack / 'samples.npz', allow_pickle=False) as archive:
        images, demo_source_outputs = archive['images'], archive['masks']
        if images.shape != (221, 256, 256, 3) or demo_source_outputs.shape != images.shape:
            raise SystemExit('frozen pack shape mismatch')
        for ordinal, ep in enumerate(plan['episodes']):
            if ep['query_global'] in ep['demo_global']: raise SystemExit('query used as its own demonstration')
            inputs = build_inputs([images[i] for i in ep['demo_global']],
                                  [demo_source_outputs[i] for i in ep['demo_global']],
                                  images[ep['query_global']], task=ep['task'], cid=ep['cid'])
            path = args.out / f'episode_{ordinal:03d}.npz'
            np.savez(path, demo_input=inputs.demo_inputs[0], demo_output=inputs.demo_outputs[0], query_input=inputs.query_input)
            manifest['episodes'].append({'ordinal': ordinal, 'task': ep['task'], 'path': path.name, 'sha256': sha(path),
                                         'demo_input_sha256': pixels_sha256(inputs.demo_inputs[0]),
                                         'demo_output_sha256': pixels_sha256(inputs.demo_outputs[0]),
                                         'query_input_sha256': pixels_sha256(inputs.query_input)})
    (args.out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'episodes': 221, 'elapsed_seconds': time.monotonic() - start,
                      'input_manifest_sha256': sha(args.out / 'inputs.json')}))


if __name__ == '__main__': main()
