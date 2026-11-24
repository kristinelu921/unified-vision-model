#!/usr/bin/env python3
"""Trusted protocol preparation; exports only each query's permitted inputs.

This process verifies the frozen pack/plan. It never computes a query target,
per-example statistic, caption, score or output-dependent selection. The masks
array is accessed only at the demonstration indices prescribed by the plan.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from contracts import (PACK_ID, MODEL_REVISION, SOURCE_REVISION, MANIFEST_SHA256,
                       identity, file_sha256, array_sha256, noise_seed, validate_payload)

ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA256=None  # Verified against the baseline evidence rather than a mutable location.

def allowed_episode(images,masks,ep,coder):
    if ep['cid']!=0 or ep['p']!=1 or ep['p0']!=1 or len(ep['demo_global'])!=1:
        raise ValueError('only the frozen single-demo CID-zero contract is supported')
    d=int(ep['demo_global'][0]);q=int(ep['query_global'])
    if d==q:raise ValueError('query cannot be its own demonstration')
    # CID-zero first-appearance coding is a prefix operation. Later query labels
    # cannot change earlier demo assignments; synthetic interventions test this.
    demo_output=coder([masks[d]],ep['task'],0)[0]
    return np.asarray(images[d],np.uint8),demo_output,np.asarray(images[q],np.uint8)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    evidence=json.loads((ROOT/'manifests/baseline_verified_files.json').read_text())
    known=ROOT.parents[2]/'eval_runs/2026-10-26/promptdiffusion_bench_v2_p1_task_text_v1_rerun2/provenance/plan_val.json'
    if file_sha256(args.plan)!=evidence[str(known)]['sha256']:
        raise ValueError('plan bytes differ from the verified baseline')
    plan=json.loads(args.plan.read_text())
    if plan['phase']!='val' or len(plan['episodes'])!=221 or len(plan['packs'])!=1:
        raise ValueError('full frozen validation plan required')
    pack=plan['packs'][0]
    if pack['data_identity']['pack_id']!=PACK_ID or pack['data_identity']['supervision_version']!=6:
        raise ValueError('pack identity differs')
    for name in ['samples.json','samples.npz']:
        if file_sha256(args.pack/name)!=pack['data_identity']['file_sha256'][name]:
            raise ValueError('pack bytes differ: '+name)
    code=ROOT/'source_reference/bench_episode_v1.py'
    if file_sha256(code)!=plan['episode_replay_sha256']:
        raise ValueError('episode coder differs')
    spec=importlib.util.spec_from_file_location('frozen_episode_coder',code)
    be=importlib.util.module_from_spec(spec);spec.loader.exec_module(be)
    with np.load(args.pack/'samples.npz',allow_pickle=False) as archive:
        images=archive['images'];masks=archive['masks']
    if images.shape!=(221,256,256,3) or masks.shape!=images.shape or images.dtype!=np.uint8 or masks.dtype!=np.uint8:
        raise ValueError('pack array contract differs')
    rows=[];episodes=[]
    for i,ep in enumerate(plan['episodes']):
        di,do,q=allowed_episode(images,masks,ep,be.code_episode)
        if array_sha256(q)!=ep['query_input_pixels_sha256']:
            raise ValueError('query input identity differs')
        rows.append((di,do,q))
        episodes.append({'ordinal':i,'key':ep['key'],'task':ep['task'],
            'query_record':ep['query_record'],'seed':noise_seed(PACK_ID,ep['query_record']),
            'demo_input_sha256':array_sha256(di),'demo_output_sha256':array_sha256(do),
            'query_input_sha256':array_sha256(q)})
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez(args.output/'inputs.npz',demo_input=np.stack([x[0] for x in rows]),
             demo_output=np.stack([x[1] for x in rows]),query_input=np.stack([x[2] for x in rows]))
    payload={'schema':'promptdiffusion_permitted_inputs_v1','pack_id':PACK_ID,
        'source_revision':SOURCE_REVISION,'model_revision':MODEL_REVISION,
        'episode_seed':42,'demonstrations':1,'cid':0,'episodes':episodes,
        'inputs_sha256':file_sha256(args.output/'inputs.npz')}
    payload['input_identity']=identity(payload);validate_payload(payload)
    (args.output/'inputs.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':'PASS','episodes':221,'permitted_inputs_sha256':payload['inputs_sha256'],
                      'input_identity':payload['input_identity'],'query_targets_exported':0}))

if __name__=='__main__':main()
