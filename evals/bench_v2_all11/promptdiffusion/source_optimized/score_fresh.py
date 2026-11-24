#!/usr/bin/env python3
"""Independent scorer: joins fresh predictions with frozen targets and receipts.

This program is separate from inference and may access query targets. It uses
the immutable baseline materializer and corrected-v3 scoring wrapper unchanged.
Only coverage, receipt identities and task aggregates are printed.
"""
import argparse
import concurrent.futures
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True

from contracts import file_sha256,array_sha256,identity,require_time,remaining_seconds,Recipe
ROOT=Path(__file__).resolve().parents[1]


def aggregate_task_rows(scored_episodes,tasks,readout):
    """Frozen display aggregation, returning task summaries only."""
    task_rows=[]
    for task in tasks:
        episodes=[ep for ep in scored_episodes.values() if ep['task']==task]
        if len(episodes)!=17:raise ValueError('task coverage differs')
        scores=[ep['readouts'][readout]['real'] for ep in episodes]
        metric=scores[0]['native_name']
        if any(s['native_name']!=metric or not math.isfinite(s['native']) for s in scores):
            raise ValueError('invalid native scores')
        copies=[ep['controls']['copy'] for ep in episodes]
        if any(s['native_name']!=metric or not math.isfinite(s['native']) for s in copies):
            raise ValueError('invalid copy-control scores')
        task_rows.append({'task':task,'n':17,'metric':metric,
            'direction':'lower' if scores[0]['native_lower'] else 'higher',
            'score':math.fsum(s['native'] for s in scores)/17,
            'copy_score':math.fsum(s['native'] for s in copies)/17})
    return task_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--pack-root',type=Path,required=True)
    p.add_argument('--generations',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parallel-jobs',type=int,default=16)
    args=p.parse_args();require_time(300)
    if not 1<=args.parallel_jobs<=16:raise ValueError('scorer concurrency outside fixed bound')
    started=time.monotonic()
    sys.path.insert(0,str(ROOT/'source_reference/bundle'))
    import prompt_diffusion_bench_v2 as reference
    import prepare_corrected_v3_units as materializer
    plan=reference.load_plan(args.plan,'val')
    pack=plan['packs'][0]
    for name in ['samples.json','samples.npz']:
        if file_sha256(args.pack_root/pack['rel']/name)!=pack['data_identity']['file_sha256'][name]:
            raise ValueError('scorer pack transfer identity differs')
    records={};sources=[];candidate=None;input_identity=None
    frozen_source={p.name:file_sha256(p) for p in (ROOT/'source_optimized').glob('*.py')}
    pins=json.loads((ROOT/'source_optimized/frozen_assets.json').read_text())
    expected_input=json.loads((ROOT/'inputs/inputs.json').read_text())['input_identity']
    for root in args.generations:
        generation=json.loads((root/'generation.json').read_text())
        if generation['status']!='COMPLETE' or generation['fresh_predictions'] is not True:
            raise ValueError('incomplete or nonfresh generation')
        if generation['completed']!=generation['expected_shard'] or len(generation['records'])!=generation['completed']:
            raise ValueError('shard count differs')
        if candidate is None:candidate=generation['candidate'];input_identity=generation['input_identity']
        if candidate!=generation['candidate'] or input_identity!=generation['input_identity']:
            raise ValueError('cannot mix candidates or permitted input identities')
        recipe=Recipe(candidate=candidate,normalization='minus_one_one' if candidate=='A1' else 'zero_one',batch_size=8 if candidate=='S3' else 4,
            cache_conditioning=candidate in ('S2','S3'),dynamic_coefficients=candidate in ('S2','S3'))
        recipe.validate()
        if (generation['source']!=frozen_source or generation['weights']!=pins['ASSET_SHA256']
            or generation['recipe']!=asdict(recipe) or input_identity!=expected_input):
            raise ValueError('generation source, weights, recipe or frozen inputs differ')
        for r in generation['records']:
            i=r['ordinal']
            if i in records or not 0<=i<221:raise ValueError('duplicate or invalid prediction ordinal')
            path=root/r['file']
            if not path.resolve().is_relative_to(root.resolve()) or file_sha256(path)!=r['sha256']:
                raise ValueError('prediction file integrity failure')
            records[i]=(path,r)
        sources.append({'generation_sha256':file_sha256(root/'generation.json'),
                        'source':generation['source'],'recipe':generation['recipe'],
                        'timings':generation['timings'],'total_seconds':generation['total_seconds']})
    if set(records)!=set(range(221)):raise ValueError('all 221 fresh predictions required')
    args.output.mkdir(parents=True,exist_ok=False)
    # Only the cache's local location changes; the frozen scientific plan remains
    # byte-identified and unmodified. Target construction occurs only here.
    cache_plan=copy.deepcopy(plan)
    cache_plan['packs'][0]['local_dir']=str(args.pack_root/pack['rel'])
    cache=reference.PackCache(cache_plan)
    view=reference.initial_view(pack,plan)
    view['fresh_generation_provenance']=sources
    for i,ep in enumerate(plan['episodes']):
        require_time(5)
        prediction=np.load(records[i][0],allow_pickle=False)
        if prediction.shape!=(256,256,3) or prediction.dtype!=np.uint8 or array_sha256(prediction)!=records[i][1]['pixels_sha256']:
            raise ValueError('prediction encoding integrity failure')
        _,_,query,target=reference.rebuild_episode(ep,cache)
        sampling={'model':reference.MODEL_ID,'model_revision':reference.MODEL_REVISION,
            'source_commit':reference.SOURCE_COMMIT,'native_prompt_count':1,
            'prompt':reference.TASK_PROMPTS[ep['task']],'negative_prompt':'',
            'noise_seed':reference.noise_seed(ep),'inference_steps':50,
            'guidance_scale':7.5,'eta':0.,'ensemble':False,
            'safety_policy':'safety_checker_disabled','score_pending':True,
            'autoresearch_candidate':candidate,'fresh_generation_provenance':sources}
        reference.save_episode(ep,prediction,query,target,pack,view,args.output,sampling)
    view['complete']=True
    reference.atomic_json(args.output/'views'/pack['slug']/'results.json',view)
    os.environ['BENCH_PACK_ROOT_OVERRIDE']=str(args.pack_root)
    groups=[(f'edge_q{ep["q"]:02d}',[ep]) for ep in plan['episodes'] if ep['task']=='edge_detection']
    groups += [(task,[ep for ep in plan['episodes'] if ep['task']==task]) for task in reference.TASK_PROMPTS if task!='edge_detection']
    def score(group):
        name,episodes=group
        unit_root=args.output/'units'/name
        receipt=materializer.build_unit(pack,episodes,plan,args.output/'views',unit_root)
        unit=Path(receipt['unit'])
        with (unit_root/'score.log').open('w') as log:
            subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(max(1,int(remaining_seconds()-30)))+'s',
                'bash',str(ROOT/'source_reference/score_corrected_v3.sh'),str(unit),str(args.pack_root)],
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest_path=unit/'scored_eval_v2/score_manifest.json';result_path=unit/'scored_eval_v2/results.json'
        manifest=json.loads(manifest_path.read_text());result=json.loads(result_path.read_text())
        if not manifest['authoritative'] or not manifest['requested_contract_satisfied'] or not result['score_complete']:
            raise ValueError('frozen metric contract not satisfied')
        return {'group':name,'episodes':len(episodes),'results_path':str(result_path),
                'scored_results_sha256':file_sha256(result_path),'score_manifest_sha256':file_sha256(manifest_path),
                'metric_identity_sha256':manifest['metric_identity_sha256']},result
    receipt_rows=[];scored_episodes={};score_start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
        for receipt,result in pool.map(score,groups):
            receipt_rows.append(receipt)
            for key,ep in result['episodes'].items():
                if key in scored_episodes:raise ValueError('duplicate score')
                scored_episodes[key]=ep
    if len(scored_episodes)!=221:raise ValueError('incomplete scoring')
    baseline=json.loads((ROOT/'audit/baseline_identity.json').read_text())
    if {r['metric_identity_sha256'] for r in receipt_rows}!={baseline['metric_identity_sha256']}:
        raise ValueError('frozen metric identity changed')
    task_rows=aggregate_task_rows(scored_episodes,reference.TASK_PROMPTS,reference.READOUT)
    aggregate={'status':'COMPLETE','candidate':candidate,'fresh_predictions':True,'episodes':221,'tasks':task_rows,
        'scored_pairs_including_copy':442,'units':receipt_rows,'pack_id':baseline['pack_id'],
        'supervision_version':6,'plan_identity':plan['plan_identity'],'release_sha256':materializer.CORRECTED_RELEASE_SHA256,
        'metric_identity_sha256':baseline['metric_identity_sha256'],'generation_sources':sources,
        'scoring_seconds':time.monotonic()-score_start,'materialization_and_scoring_seconds':time.monotonic()-started}
    reference.atomic_json(args.output/'aggregate.json',aggregate)
    print(json.dumps({'status':'COMPLETE','candidate':candidate,'episodes':221,'tasks':13,'score_units':29,
                      'aggregate_sha256':file_sha256(args.output/'aggregate.json')}))

if __name__=='__main__':main()
