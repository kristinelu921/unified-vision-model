"""Full test evaluation with selected VICT R0; resume replays every prior update."""
import datetime,hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
import numpy as np
from PIL import Image
import vict_native as V
import adaptation as A
from test_episode_p1 import episode,pixel_sha
from tools import bench_episode_v1 as BE
from tools.bench_v2_runtime_v2 import stage_pack
ROOT=Path(__file__).resolve().parent
MANIFEST_SHA='89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d'
PREFIX='gs://kmh-gcp-us-central2/data/xtiange/mt_eval_packs'
OUTPUT='gs://kmh-gcp-us-central2/kristine/eval_runs/2026-11-10/bench_v2_all11_test/vict/test_r0_60step_v1'
sha=V.sha256_file
class Store:
    scheme='gs://'
    def download(self,uri,destination):
        assert uri.startswith(PREFIX+'/')
        subprocess.run(['gsutil','-o','GSUtil:check_hashes=always','cp',uri,str(destination)],check=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def upload(path,uri):subprocess.run(['gsutil','-o','GSUtil:check_hashes=always','cp',str(path),uri],check=True)
def publish(directory,progress):
    save(directory/'progress.json',progress)
    archive=ROOT/'test_outputs'/(directory.name+'.tar')
    with tarfile.open(archive,'w') as out:out.add(directory,arcname=directory.name)
    upload(archive,OUTPUT+'/packs/'+archive.name)
    upload(directory/'progress.json',OUTPUT+'/progress/'+directory.name+'.json')
def main():
    assert os.uname().nodename=='t1v-n-7ff0e5a7-w-0'
    for manifest in ['SOURCE_SHA256SUMS','TEST_SHA256SUMS']:
        subprocess.run(['sha256sum','-c',manifest],cwd=ROOT,check=True)
    gate=json.loads((ROOT/'qualification.json').read_text())
    assert gate['status']=='PASS' and gate['full_60_step_same_backend_exact']
    assert gate['checkpoint_sha256']==V.CHECKPOINT_SHA256 and gate['source_manifest_sha256']==sha(ROOT/'SOURCE_SHA256SUMS')
    inputs_gate=json.loads((ROOT/'vict_input_qualification.json').read_text())
    assert inputs_gate['status']=='PASS' and inputs_gate['episodes']==221
    assert inputs_gate['plan_identity']=='3bd6ccf3a6b3697bd71afffc1a68cf5176c96b0a833f2fd1b28e880ae87480d2'
    assert sha(ROOT/'test_manifest.json')==MANIFEST_SHA
    entries=[e for e in json.loads((ROOT/'test_manifest.json').read_text())['packs'] if e['rel'].startswith('bench_v2_test/')]
    assert len(entries)==26 and sum(e['n'] for e in entries)==17511
    import torch,torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    from torch_xla.amp import syncfree
    assert torch.__version__==gate['torch'] and torch_xla.__version__==gate['torch_xla']
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    xr.initialize_cache(str(ROOT/'compiler_cache'),readonly=False);xr.use_spmd()
    assert xr.device_type()=='TPU' and xr.global_runtime_device_count()==xr.addressable_runtime_device_count()==4
    xs.set_global_mesh(xs.Mesh(np.arange(4),(4,),('replica',)))
    device=xm.xla_device();torch.manual_seed(0);np.random.seed(0)
    model=V.load_model(ROOT/'official',Path('/tmp/kristine_bench_all11_painter_20261110/weights/painter_vit_large.pth'))
    names=V.configure_encoder(model);wanted=set(names)
    base={n:p.detach().clone() for n,p in model.named_parameters() if n in wanted}
    model=model.to(device);xm.mark_step();xm.wait_device_ops()
    parameters=dict(model.named_parameters())
    optimizer=syncfree.AdamW([parameters[n] for n in names],lr=V.LR,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    sentinel=parameters['blocks.0.attn.qkv.weight']
    mask=torch.zeros((1,model.patch_embed.num_patches),dtype=torch.bool,device=device);mask[:,model.patch_embed.num_patches//2:]=True
    original_predict=A.predict;raw=[]
    def observed_predict(*args,**kwargs):
        module=args[0];original_decoder=module.forward_decoder
        def decoder(*values,**options):
            result=original_decoder(*values,**options);raw.append(result.detach().cpu());return result
        module.forward_decoder=decoder
        try:return original_predict(*args,**kwargs)
        finally:module.forward_decoder=original_decoder
    A.predict=observed_predict
    identity=dict(recipe='vict-p1-depth-decoded',steps=60,episode_seed=42,model_seed=0,p=1,
                  checkpoint_sha256=V.CHECKPOINT_SHA256,source_manifest_sha256=sha(ROOT/'SOURCE_SHA256SUMS'),
                  test_manifest_sha256=sha(ROOT/'TEST_SHA256SUMS'),qualification_sha256=sha(ROOT/'qualification.json'),
                  input_qualification_sha256=sha(ROOT/'vict_input_qualification.json'),resume_policy='full_ordered_update_replay')
    run_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    record=ROOT/'test_receipts/run_identity.json'
    if record.exists():assert json.loads(record.read_text())==dict(identity=identity,run_identity=run_id)
    else:save(record,dict(identity=identity,run_identity=run_id))
    upload(record,OUTPUT+'/provenance/run_identity.json')
    total=0
    for entry in entries:
        rel=entry['rel'];slug=rel.replace('/','__')
        pack=stage_pack(rel,uri_root=PREFIX,manifest_path=ROOT/'test_manifest.json',manifest_sha256=MANIFEST_SHA,
                        local_root=ROOT/'test_packs',store=Store())
        directory=ROOT/'test_outputs'/slug;directory.mkdir(parents=True,exist_ok=True)
        queries=[(task,q) for task in dict.fromkeys(r['task'] for r in pack['records'])
                 for q in range(sum(r['task']==task for r in pack['records']))]
        assert len(queries)==entry['n']
        for index,(task,q) in enumerate(queries):
            data,binding=episode(pack,task,q);assert binding['cid']==0
            support=BE.code_episode([pack['masks'][binding['demo_global'][0]]],task,0)[0]
            assert np.array_equal(support,data.demo_target),'Support-only coding changed'
            tiles=dict(demo_input=data.demo_input,demo_target=support,query_input=data.query_input)
            path=directory/f'{task}_q{q:05d}';receipt=path/'receipt.json'
            old=json.loads(receipt.read_text()) if receipt.exists() else None
            if old:
                assert old['run_identity']==run_id and old['binding']==binding and old['pack_id']==entry['pack_id']
                assert sha(path/'prediction.png')==old['prediction_sha256']
            # Never skip completed queries: all60 updates consume the exact original RNG stream.
            raw.clear();started=time.monotonic()
            prediction,evidence=A.adapt_predict(model,parameters,names,base,optimizer,sentinel,mask,device,tiles,
                                               dict(task=task,episode_id=f'{slug}:{task}:{q}'))
            assert len(raw)==1 and bool(torch.isfinite(raw[0]).all())
            assert prediction.shape==(256,256,3) and prediction.dtype==np.uint8
            stable={k:v for k,v in evidence.items() if k not in ['step_seconds','inference_seconds']}
            raw_sha=hashlib.sha256(raw[0].contiguous().numpy().tobytes()).hexdigest()
            if old:
                assert old['prediction_pixels_sha256']==pixel_sha(prediction) and old['adaptation']==stable
                assert old['raw_prediction_sha256']==raw_sha
            else:
                path.mkdir(parents=True,exist_ok=True)
                temp=path/'prediction.tmp.png';Image.fromarray(prediction).save(temp);temp.replace(path/'prediction.png')
                save(receipt,dict(run_identity=run_id,binding=binding,pack_rel=rel,pack_id=entry['pack_id'],
                                 prediction_sha256=sha(path/'prediction.png'),prediction_pixels_sha256=pixel_sha(prediction),
                                 raw_prediction_sha256=raw_sha,raw_output_finite=True,adaptation=stable,
                                 elapsed_seconds=time.monotonic()-started,completed_utc=now(),
                                 model_inputs=['demo_input','support_only_coded_demo_target','query_input'],scoring_complete=False))
            total+=1
            progress=dict(status='RUNNING',model='vict',run_identity=run_id,pack_rel=rel,pack_completed=index+1,
                          pack_expected=entry['n'],total_completed=total,total_expected=17511,updated_utc=now())
            save(ROOT/'test_receipts/progress.json',progress);print(json.dumps(progress),flush=True)
            if (index+1)%8==0:publish(directory,progress)
        progress['status']='PACK_INFERENCE_COMPLETE';publish(directory,progress);del pack
    assert total==17511
    complete=ROOT/'test_receipts/inference_complete.json'
    save(complete,dict(status='INFERENCE_COMPLETE',run_identity=run_id,queries=total,packs=26,scoring_complete=False,completed_utc=now()))
    upload(complete,OUTPUT+'/test_inference_complete.json')
if __name__=='__main__':main()
