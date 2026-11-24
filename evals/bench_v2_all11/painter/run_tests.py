"""Run every frozen Bench v2 test query with archived Painter FP32 batch4."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image
import painter_native as P
from episode_p1 import episode, pixel_sha
from qualify import ROOT, PREFIX, MANIFEST_SHA, Store
from tools.bench_v2_runtime_v2 import stage_pack

OUTPUT = 'gs://kmh-gcp-us-central2/kristine/eval_runs/2026-11-10/bench_v2_all11_test/painter/test_native_fp32_v1'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def upload(source, destination):
    subprocess.run(['gsutil', '-o', 'GSUtil:check_hashes=always', 'cp', str(source), destination], check=True)


def publish(directory, progress):
    save(directory/'progress.json', progress)
    archive = ROOT/'outputs'/f'{directory.name}.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(directory, arcname=directory.name)
    upload(archive, OUTPUT+'/packs/'+archive.name)
    upload(directory/'progress.json', OUTPUT+'/progress/'+directory.name+'.json')


def main():
    assert os.uname().nodename == 't1v-n-7ff0e5a7-w-0'
    subprocess.run(['sha256sum','-c','SOURCE_SHA256SUMS'],cwd=ROOT,check=True)
    gate = json.loads((ROOT/'receipts/painter_qualification.json').read_text())
    source_sha = P.sha256_file(ROOT/'SOURCE_SHA256SUMS')
    assert gate['status']=='PASS' and gate['validation_episodes_replayed']==221
    assert gate['source_manifest_sha256']==source_sha and gate['raw_output_finite']
    assert gate['unchanged_call_repeat_exact'] and gate['checkpoint_sha256']==P.CHECKPOINT_SHA256
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    assert torch.__version__==gate['torch_version'] and torch_xla.__version__==gate['torch_xla_version']
    torch.set_num_threads(8); torch.manual_seed(42); np.random.seed(42)
    xr.initialize_cache(str(ROOT/'xla_cache'),readonly=False); xr.use_spmd()
    assert xr.device_type()=='TPU' and xr.global_runtime_device_count()==xr.addressable_runtime_device_count()==4
    mesh=xs.Mesh(np.arange(4),(4,),('data',));xs.set_global_mesh(mesh)
    device=xm.xla_device()
    model=P.load_model(ROOT/'official',ROOT/'weights/painter_vit_large.pth',device)
    identity=dict(recipe='painter-p1-depth-decoded',p=1,batch_size=4,seed=42,
                  dtype='float32',backend='TPU',accelerator='v4-8',
                  checkpoint_sha256=P.CHECKPOINT_SHA256,source_manifest_sha256=source_sha,
                  test_runner_sha256=P.sha256_file(Path(__file__)),
                  qualification_sha256=P.sha256_file(ROOT/'receipts/painter_qualification.json'))
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(ROOT/'receipts/test_run_identity.json',dict(identity=identity,run_identity=run_id))
    upload(ROOT/'receipts/test_run_identity.json',OUTPUT+'/provenance/test_run_identity.json')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=[e for e in manifest['packs'] if e['rel'].startswith('bench_v2_test/')]
    assert len(entries)==26 and sum(e['n'] for e in entries)==17511
    raw=[]
    original_decoder=model.forward_decoder
    def observed_decoder(*args,**kwargs):
        value=original_decoder(*args,**kwargs);raw.append(value.detach().cpu());return value
    model.forward_decoder=observed_decoder
    total=0
    for entry in entries:
        rel=entry['rel']
        pack=stage_pack(rel,uri_root=PREFIX,manifest_path=ROOT/'manifest.json',manifest_sha256=MANIFEST_SHA,
                        local_root=Path('/dev/shm/kristine_bench_all11_painter_packs'),store=Store())
        queries=[(task,q) for task in dict.fromkeys(r['task'] for r in pack['records'])
                 for q in range(sum(r['task']==task for r in pack['records']))]
        assert len(queries)==entry['n']
        directory=ROOT/'outputs/test'/rel.replace('/','__');directory.mkdir(parents=True,exist_ok=True)
        count=0
        for start in range(0,len(queries),4):
            batch=queries[start:start+4]
            built=[episode(pack,task,q) for task,q in batch]
            paths=[directory/f'{task}_q{q:05d}' for task,q in batch]
            existing=[]
            for path,(_,binding) in zip(paths,built):
                receipt=path/'receipt.json'
                if receipt.exists():
                    old=json.loads(receipt.read_text())
                    assert old['run_identity']==run_id and old['binding']==binding and old['pack_id']==entry['pack_id']
                    assert P.sha256_file(path/'prediction.png')==old['prediction_sha256']
                    existing.append(True)
                else: existing.append(False)
            if not all(existing):
                inputs=[data._asdict() for data,_ in built]
                inputs+= [inputs[0]]*(4-len(inputs))
                x,y=P.preprocess(inputs);raw.clear();began=time.monotonic()
                predictions=P.predict(model,x,y,device,mesh)
                assert len(raw)==1 and raw[0].dtype==torch.float32 and bool(torch.isfinite(raw[0]).all()), f'Nonfinite output: {rel} {batch}'
                elapsed=time.monotonic()-began
                for path,(_,binding),prediction,exists in zip(paths,built,predictions,existing):
                    if exists:continue
                    path.mkdir(parents=True,exist_ok=True)
                    temp=path/'prediction.tmp.png';Image.fromarray(prediction).save(temp);temp.replace(path/'prediction.png')
                    save(path/'receipt.json',dict(run_identity=run_id,binding=binding,pack_rel=rel,pack_id=entry['pack_id'],
                         prediction_sha256=P.sha256_file(path/'prediction.png'),prediction_pixels_sha256=pixel_sha(prediction),
                         raw_output_finite=True,batch_seconds=elapsed,completed_utc=now(),
                         model_inputs=['demo_input','coded_demo_target','query_input'],scoring_complete=False))
            count+=len(batch);total+=len(batch)
            progress=dict(status='RUNNING',model='painter',run_identity=run_id,pack_rel=rel,
                          pack_completed=count,pack_expected=entry['n'],total_completed=total,total_expected=17511,updated_utc=now())
            save(ROOT/'receipts/test_progress.json',progress);print(json.dumps(progress),flush=True)
            if count%64==0:publish(directory,progress)
        assert count==entry['n'];progress['status']='PACK_INFERENCE_COMPLETE';publish(directory,progress)
        del pack
    assert total==17511
    complete=dict(status='INFERENCE_COMPLETE',run_identity=run_id,queries=total,packs=26,scoring_complete=False,completed_utc=now())
    save(ROOT/'receipts/test_inference_complete.json',complete)
    upload(ROOT/'receipts/test_inference_complete.json',OUTPUT+'/test_inference_complete.json')


if __name__=='__main__':
    main()
