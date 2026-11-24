"""Run every frozen Bench v2 test query with selected IMProv CPU FP32 batch4."""
import argparse
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
import improv_cpu as P
from episode_p1 import episode, pixel_sha
ROOT=Path(__file__).resolve().parent
PREFIX='gs://kmh-gcp-us-central2/data/xtiange/mt_eval_packs'
MANIFEST_SHA='89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d'
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
class Store:
    scheme='gs://'
    def download(self,uri,destination):
        if not uri.startswith(PREFIX+'/'):raise ValueError('Nonregional pack')
        subprocess.run(['gsutil','-o','GSUtil:check_hashes=always','cp',uri,str(destination)],check=True)
from tools.bench_v2_runtime_v2 import stage_pack

OUTPUT = 'gs://kmh-gcp-us-central2/kristine/eval_runs/2026-11-10/bench_v2_all11_test/improv/test_cpu_fp32_v1'


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
    archive = directory.parent/f'{directory.name}.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(directory, arcname=directory.name)
    upload(archive, OUTPUT+'/packs/'+archive.name)
    upload(directory/'progress.json', OUTPUT+'/progress/'+directory.name+'.json')


def main():
    global OUTPUT
    ap=argparse.ArgumentParser();ap.add_argument('--shard-index',type=int,required=True);ap.add_argument('--shard-count',type=int,default=8);a=ap.parse_args()
    assert a.shard_count==8 and 0<=a.shard_index<a.shard_count
    tag=f'shard_{a.shard_index:02d}_of_{a.shard_count:02d}'
    OUTPUT += '/shards/'+tag
    assert os.uname().nodename == 't1v-n-7ff0e5a7-w-0'
    subprocess.run(['sha256sum','-c','SOURCE_SHA256SUMS'],cwd=ROOT,check=True)
    gate=json.loads((ROOT/'receipts/model_qualification.json').read_text())
    assert gate['status']=='PASS' and gate['backend']=='CPU' and gate['batch_size']==4
    assert gate['source_sha256']==file_sha(ROOT/'improv_cpu.py')
    assert gate['qualification_sha256']==file_sha(ROOT/'qualify_model.py')
    assert gate['pins_sha256']==file_sha(ROOT/'model_source_pins.json')
    assert len(gate['records'])==4 and all(x['tokens_exact'] and x['raw_decoding_exact'] and x['pixels_exact'] and x['raw_finite'] for x in gate['records'])
    pins=json.loads((ROOT/'model_source_pins.json').read_text())
    for name,expected in pins.items():assert file_sha(ROOT/'assets'/name)==expected
    import torch
    assert torch.__version__==gate['torch_version']
    torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(0);np.random.seed(0)
    runtime=P.load_runtime(ROOT/'assets/source',ROOT/'assets/model',ROOT/'assets/vq')
    for model in runtime[:3]:model.eval().to('cpu')
    embedding,embedding_sha=P.encode_prompts(ROOT/'assets/source',ROOT/'assets/model')
    assert embedding_sha==gate['embedding_sha256']
    identity=dict(recipe='improv-cpu-reference-edge-sqrt',p=1,batch_size=4,episode_seed=42,
                  dtype='float32',backend='CPU',seed=0,shard_count=a.shard_count,
                  source_manifest_sha256=file_sha(ROOT/'SOURCE_SHA256SUMS'),
                  test_runner_sha256=file_sha(Path(__file__)),
                  qualification_sha256=file_sha(ROOT/'receipts/model_qualification.json'))
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    save(ROOT/'receipts'/tag/'test_run_identity.json',dict(identity=identity,run_identity=run_id))
    upload(ROOT/'receipts'/tag/'test_run_identity.json',OUTPUT+'/provenance/test_run_identity.json')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    entries=[e for e in manifest['packs'] if e['rel'].startswith('bench_v2_test/')]
    assert len(entries)==26 and sum(e['n'] for e in entries)==17511
    expected_total=sum(min(4,e['n']-start) for e in entries for start in range(0,e['n'],4) if (start//4)%a.shard_count==a.shard_index)
    raw=[]
    def observed_decoder(module,args,value):
        raw.append(value.detach())
    runtime[1].register_forward_hook(observed_decoder)
    total=0
    for entry in entries:
        rel=entry['rel']
        pack=stage_pack(rel,uri_root=PREFIX,manifest_path=ROOT/'manifest.json',manifest_sha256=MANIFEST_SHA,
                        local_root=Path('/dev/shm/kristine_bench_all11_improv_packs'),store=Store())
        queries=[(task,q) for task in dict.fromkeys(r['task'] for r in pack['records'])
                 for q in range(sum(r['task']==task for r in pack['records']))]
        assert len(queries)==entry['n']
        directory=ROOT/'outputs'/tag/rel.replace('/','__');directory.mkdir(parents=True,exist_ok=True)
        expected_pack=sum(min(4,len(queries)-start) for start in range(0,len(queries),4) if (start//4)%a.shard_count==a.shard_index)
        count=0
        for start in range(0,len(queries),4):
            if (start//4)%a.shard_count!=a.shard_index:continue
            batch=queries[start:start+4]
            built=[episode(pack,task,q) for task,q in batch]
            paths=[directory/f'{task}_q{q:05d}' for task,q in batch]
            existing=[]
            for path,(_,binding) in zip(paths,built):
                receipt=path/'receipt.json'
                if receipt.exists():
                    old=json.loads(receipt.read_text())
                    assert old['run_identity']==run_id and old['binding']==binding and old['pack_id']==entry['pack_id']
                    assert file_sha(path/'prediction.png')==old['prediction_sha256']
                    existing.append(True)
                else: existing.append(False)
            if not all(existing):
                inputs=[data._asdict() for data,_ in built]
                inputs+= [inputs[0]]*(4-len(inputs))
                tiles=[P.build_canvas(x['demo_input'],x['demo_target'],x['query_input']) for x in inputs]
                arrays=(np.stack([x[0] for x in tiles]),np.stack([x[1] for x in tiles]))
                raw.clear();began=time.monotonic()
                predictions=P.predict_batch(runtime,arrays,embedding,torch.device('cpu'),None)
                assert len(raw)==1 and raw[0].dtype==torch.float32 and bool(torch.isfinite(raw[0]).all()), f'Nonfinite output: {rel} {batch}'
                elapsed=time.monotonic()-began
                for path,(_,binding),prediction,exists in zip(paths,built,predictions,existing):
                    if exists:continue
                    path.mkdir(parents=True,exist_ok=True)
                    temp=path/'prediction.tmp.png';Image.fromarray(prediction).save(temp);temp.replace(path/'prediction.png')
                    save(path/'receipt.json',dict(run_identity=run_id,binding=binding,pack_rel=rel,pack_id=entry['pack_id'],
                         prediction_sha256=file_sha(path/'prediction.png'),prediction_pixels_sha256=pixel_sha(prediction),
                         raw_output_finite=True,batch_seconds=elapsed,completed_utc=now(),
                         model_inputs=['demo_input','coded_demo_target','query_input'],scoring_complete=False))
            count+=len(batch);total+=len(batch)
            progress=dict(status='RUNNING',model='improv',shard_index=a.shard_index,shard_count=a.shard_count,run_identity=run_id,pack_rel=rel,
                          pack_completed=count,pack_expected=expected_pack,global_pack_expected=entry['n'],total_completed=total,total_expected=expected_total,global_total_expected=17511,updated_utc=now())
            save(ROOT/'receipts'/tag/'test_progress.json',progress);print(json.dumps(progress),flush=True)
            if count%64==0:publish(directory,progress)
        assert count==expected_pack;progress['status']='PACK_INFERENCE_COMPLETE';publish(directory,progress)
        del pack
    assert total==expected_total
    complete=dict(status='INFERENCE_COMPLETE',run_identity=run_id,queries=total,packs=26,scoring_complete=False,completed_utc=now())
    save(ROOT/'receipts'/tag/'test_inference_complete.json',complete)
    upload(ROOT/'receipts'/tag/'test_inference_complete.json',OUTPUT+'/test_inference_complete.json')


if __name__=='__main__':
    main()
