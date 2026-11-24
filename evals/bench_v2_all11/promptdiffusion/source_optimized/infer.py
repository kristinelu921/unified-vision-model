#!/usr/bin/env python3
"""Fresh target-free TPU inference, requiring current launch and parity receipts."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from contracts import (Recipe,validate_payload,file_sha256,array_sha256,identity,cache_key,
                       remaining_seconds,require_time,fresh_output,normalize_rgb,runtime_identity,
                       launch_resource_is_authorized)

ROOT=Path(__file__).resolve().parents[1]

def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(temp,path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--official-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--launch-receipt',type=Path,required=True)
    parser.add_argument('--parity-receipt',type=Path,required=True)
    parser.add_argument('--speed-selection',type=Path)
    parser.add_argument('--candidate',choices=['S1','S2','S3','A1'],default='S1')
    parser.add_argument('--worker-index',type=int,default=0)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--devices',type=int,default=4)
    args=parser.parse_args();require_time(900)
    started=time.monotonic();start_utc=datetime.now(timezone.utc).isoformat()
    launch=json.loads(args.launch_receipt.read_text())
    checked=datetime.fromisoformat(launch['checked_utc'])
    if (launch['status']!='AVAILABLE' or not launch_resource_is_authorized(launch)
        or launch['name']!=os.environ.get('EXPECTED_TPU_NAME')
        or launch['zone']!=os.environ.get('EXPECTED_TPU_ZONE')
        or launch['run_id']!=ROOT.name or (datetime.now(timezone.utc)-checked).total_seconds()>120
        or not launch.get('exclusive_assignment_verified')
        or not launch.get('all_workers_available')):
        raise ValueError('fresh exclusive authorized every-worker launch receipt required')
    allocation_seconds=int(launch.get('allocated_max_seconds',0))
    if not 0<allocation_seconds<=int(32*3600/launch['physical_chip_count']):
        raise ValueError('bounded chip-hour allocation required')
    recipe=Recipe(candidate=args.candidate,normalization='minus_one_one' if args.candidate=='A1' else 'zero_one',batch_size=8 if args.candidate=='S3' else 4,
        cache_conditioning=args.candidate in ('S2','S3'),dynamic_coefficients=args.candidate in ('S2','S3'))
    recipe.validate()
    if args.devices!=recipe.batch_size or not 0<=args.worker_index<args.workers:
        raise ValueError('unsupported device/batch/shard assignment')
    pins=json.loads((Path(__file__).parent/'frozen_assets.json').read_text())
    for name,expected in pins['ASSET_SHA256'].items():
        if file_sha256(args.model/name)!=expected:raise ValueError('model bytes differ: '+name)
    for name,expected in pins['SOURCE_SHA256'].items():
        if file_sha256(args.official_source/name)!=expected:raise ValueError('official source bytes differ: '+name)
    source={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    parity=json.loads(args.parity_receipt.read_text())
    if (parity.get('status')!='PASS' or parity.get('candidate')!=args.candidate
        or parity.get('backend')!='TPU' or parity.get('fixtures')!=16
        or parity.get('exact_uint8_predictions')!=16 or parity.get('source')!=source
        or parity.get('weights')!=pins['ASSET_SHA256'] or parity.get('recipe')!=asdict(recipe)):
        raise ValueError('real-weight TPU synthetic parity gate not passed for this source/recipe')
    if args.candidate in ('S2','S3'):
        if args.speed_selection is None:raise ValueError('optimized validation requires frozen synthetic speed selection')
        selection=json.loads(args.speed_selection.read_text())
        selected=[r for r in selection.get('candidates',[]) if r['candidate']==args.candidate]
        if (selection.get('status')!='SELECTED_SYNTHETIC_SPEED_ONLY'
            or selection.get('selected_candidate')!=args.candidate
            or selection.get('quality_feedback_used') is not False or len(selected)!=1
            or selected[0]['receipt_sha256']!=file_sha256(args.parity_receipt)):
            raise ValueError('synthetic speed selection does not admit this exact candidate receipt')
    payload=json.loads((args.inputs/'inputs.json').read_text());validate_payload(payload)
    if file_sha256(args.inputs/'inputs.npz')!=payload['inputs_sha256']:
        raise ValueError('permitted input transfer identity differs')
    with np.load(args.inputs/'inputs.npz',allow_pickle=False) as archive:
        if set(archive.files)!={'demo_input','demo_output','query_input'}:
            raise ValueError('input archive contains nonpermitted fields')
        arrays={k:archive[k] for k in archive.files}
    for kind,value in arrays.items():
        if value.shape!=(221,256,256,3) or value.dtype!=np.uint8:
            raise ValueError('input array contract differs')
        for i,ep in enumerate(payload['episodes']):
            if array_sha256(value[i])!=ep[kind+'_sha256']:
                raise ValueError('permitted input binding differs')
    output=fresh_output(args.output)
    state={'schema':'promptdiffusion_fresh_generation_v1','status':'STARTING','fresh_predictions':True,
           'started_utc':start_utc,'candidate':args.candidate,'recipe':asdict(recipe),
           'input_identity':payload['input_identity'],'source':source,'weights':pins['ASSET_SHA256'],
           'worker_index':args.worker_index,'workers':args.workers,'completed':0,'timings':{}}
    def stop_handler(signum,frame):
        state['status']='DEADLINE_OR_TERMINATION';state['signal']=signum
        write(output/'status.json',state)
        raise SystemExit(124)
    signal.signal(signal.SIGALRM,stop_handler);signal.signal(signal.SIGTERM,stop_handler)
    signal.alarm(max(1,min(int(remaining_seconds()),allocation_seconds)))
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    import torch_xla.debug.metrics as metrics
    environment={'torch':torch.__version__,'torch_xla':torch_xla.__version__,
                 'diffusers':importlib.metadata.version('diffusers'),'transformers':importlib.metadata.version('transformers'),
                 'numpy':np.__version__,'precision':'bfloat16'}
    if not environment['torch'].startswith('2.7.0') or not environment['torch_xla'].startswith('2.7.0') or environment['diffusers']!='0.31.0' or environment['transformers']!='4.44.0':
        raise ValueError('runtime differs from frozen direct numerical dependencies')
    complete_environment=runtime_identity()
    environment['complete_environment_sha256']=identity(complete_environment)
    if (parity.get('complete_environment_sha256')!=environment['complete_environment_sha256']
        or parity.get('accelerator_type')!=launch['accelerator_type']
        or parity.get('devices')!=args.devices):
        raise ValueError('parity software/flags or accelerator layout differs from inference')
    write(output/'environment.json',complete_environment)
    prompts=pins['TASK_PROMPTS'];tasks=list(prompts)
    cache_identity=cache_key(stage='xla_compilation',inputs=payload['input_identity'],
        weights=pins['ASSET_SHA256'],source=source,environment=environment,recipe=asdict(recipe),
        prompts=prompts,tokenization={k:v for k,v in pins['ASSET_SHA256'].items() if k.startswith('tokenizer/')},
        seeds=[e['seed'] for e in payload['episodes']])
    # Includes hardware identity: a different placement cannot inherit kernels.
    cache_identity=identity({'numerical':cache_identity,'accelerator':launch['accelerator_type'],
                             'topology':launch['topology'],'devices':args.devices})
    cache=args.cache/cache_identity;cache.mkdir(parents=True,exist_ok=True)
    xr.initialize_cache(str(cache.resolve()),readonly=False);xr.use_spmd()
    if xr.device_type()!='TPU' or xr.addressable_runtime_device_count()!=args.devices or xr.global_runtime_device_count()!=args.devices:
        raise ValueError('runtime accelerator does not match the launch receipt')
    device=xm.xla_device();mesh=xs.Mesh(np.arange(args.devices),(args.devices,),('data',));xs.set_global_mesh(mesh)
    def shard(tensor):xs.mark_sharding(tensor,mesh,('data',)+(None,)*(tensor.ndim-1))
    from diffusers import AutoencoderKL,DDIMScheduler,UNet2DConditionModel
    from transformers import CLIPTextModel,CLIPTokenizer
    # Tokenization and encoding are CPU float32, once per process, as in B0.
    text_start=time.monotonic()
    tokenizer=CLIPTokenizer.from_pretrained(args.model/'tokenizer',local_files_only=True)
    encoder=CLIPTextModel.from_pretrained(args.model/'text_encoder',local_files_only=True,
        torch_dtype=torch.float32,low_cpu_mem_usage=False).eval()
    tokens=tokenizer([prompts[t] for t in tasks]+[''],padding='max_length',
        max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt')
    with torch.no_grad():embeddings=encoder(tokens.input_ids)[0].float().cpu().contiguous()
    token_hash=array_sha256(tokens.input_ids.numpy());del encoder,tokenizer
    state['timings']['text_encoding_seconds']=time.monotonic()-text_start
    state['token_ids_sha256']=token_hash
    sys.path.insert(0,str(args.official_source))
    if recipe.cache_conditioning:
        from cached_controlnet import CachedPromptDiffusionControlNetModel as Control
    else:
        from promptdiffusioncontrolnet import PromptDiffusionControlNetModel as Control
    kwargs=dict(torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=False)
    load_started=time.monotonic()
    control=Control.from_pretrained(args.model/'controlnet',**kwargs).eval().to(device)
    unet=UNet2DConditionModel.from_pretrained(args.model/'unet',**kwargs).eval().to(device)
    vae=AutoencoderKL.from_pretrained(args.model/'vae',**kwargs).eval().to(device)
    scheduler=DDIMScheduler.from_pretrained(args.model/'scheduler',local_files_only=True)
    xm.mark_step();xm.wait_device_ops()
    state['timings']['model_loading_seconds']=time.monotonic()-load_started
    from engine import Engine
    engine=Engine(control,unet,vae,scheduler,device=device,dtype=torch.bfloat16,
        cache_conditioning=recipe.cache_conditioning,dynamic_coefficients=recipe.dynamic_coefficients,
        barrier=xm.mark_step,shard=shard)
    state.update(status='RUNNING',environment=environment,compilation_cache_key=cache_identity)
    write(output/'status.json',state)
    assigned=[i for i in range(221) if (i//recipe.batch_size)%args.workers==args.worker_index]
    batch_timings=[];records=[];memory_samples=[]
    for offset in range(0,len(assigned),recipe.batch_size):
        require_time(30)
        real=assigned[offset:offset+recipe.batch_size];indices=real+[real[-1]]*(recipe.batch_size-len(real))
        host_started=time.monotonic()
        di,do,q=[normalize_rgb(arrays[k][indices],recipe.normalization) for k in ['demo_input','demo_output','query_input']]
        noise=np.stack([torch.randn((4,32,32),generator=torch.Generator(device='cpu').manual_seed(payload['episodes'][i]['seed']),dtype=torch.float32).numpy() for i in indices])
        positive=torch.cat([embeddings[tasks.index(payload['episodes'][i]['task']):tasks.index(payload['episodes'][i]['task'])+1] for i in indices])
        begin=time.monotonic();predictions=engine.predict(di,do,q,noise,positive,embeddings[-1:]);infer_seconds=time.monotonic()-begin
        for j,i in enumerate(real):
            # Only generated pixels are written. The scorer handles targets later.
            path=output/f'prediction_{i:03d}.npy';np.save(path,predictions[j],allow_pickle=False)
            records.append({'ordinal':i,'file':path.name,'sha256':file_sha256(path),
                            'pixels_sha256':array_sha256(predictions[j])})
        batch_timings.append({'count':len(real),'inference_seconds':infer_seconds,'with_io_seconds':time.monotonic()-host_started})
        try:memory_samples.append(xm.get_memory_info(device))
        except Exception:pass
        state['completed']+=len(real);write(output/'status.json',state)
        print(json.dumps({'completed':state['completed'],'expected_shard':len(assigned),'batch_seconds':infer_seconds}),flush=True)
    state.update(status='COMPLETE',expected_shard=len(assigned),records=records,
        batch_timings=batch_timings,total_seconds=time.monotonic()-started,
        peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hbm_samples=memory_samples,hbm_peak_bytes=None,
        hbm_note='Samples are not a verified peak; do not report them as one.')
    write(output/'generation.json',state);write(output/'status.json',{k:v for k,v in state.items() if k!='records'})
    (output/'xla_metrics.txt').write_text(metrics.metrics_report())
    signal.alarm(0)

if __name__=='__main__':main()
