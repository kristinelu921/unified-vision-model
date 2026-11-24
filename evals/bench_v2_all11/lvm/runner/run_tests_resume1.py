"""Run every selected LVM test episode with unchanged numerical inference."""
import datetime,hashlib,json,os,socket,subprocess,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/tmp/kristine_bench_all11_lvm_20261110')
sys.path.insert(0,str(ROOT))
import lvm_bench_v2 as L
BASE='gs://kmh-gcp-us-central2/kristine/eval_runs/2026-11-10/bench_v2_all11_test/lvm/test_native_fp16_v1'
def upload(path,uri):
 subprocess.run(['gsutil','-o','GSUtil:check_hashes=always','cp',str(path),uri],check=True)
def main():
 assert socket.gethostname()=='t1v-n-94cd5cd6-w-0'
 launch=json.loads((ROOT/'test_resume1_launch.json').read_text())
 assert launch['status']=='AVAILABLE' and launch['all_workers_available']
 assert launch['name']=='kmh-tpuvm-v4-8-spot-kristine-10-23-08-42-N-us-central2-b' and launch['zone']=='us-central2-b'
 age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(launch['checked_utc'])).total_seconds();assert 0<=age<120
 assert (ROOT/'qualification_retry1_controller.exit').read_text().strip()=='0'
 qualification=json.loads((ROOT/'qualification_retry1.json').read_text())
 assert qualification['status']=='PASS' and qualification['fresh_processes']==2 and qualification['episodes_per_trial']==13
 assert qualification['exact_contexts_tokens_quantized_latents_generated_codes_raw_decodes_and_pixels']
 assert qualification['qualification_retry1_script_sha256']==L.sha256_file(ROOT/'qualify_retry1.py')
 assert qualification['qualification_retry1_script_sha256']=='905bb2fed38b7d4dd144dea1972ebebdf8d3528a5b637adac53a85427db5d256'
 subprocess.run(['sha256sum','-c','SOURCE_SHA256SUMS'],cwd=ROOT,check=True)
 assert L.sha256_file(ROOT/'lvm_bench_v2.py')=='1926c1a1f2d196a4dfd4836b3b134ff05ad581535bd74714dd906c324dc7e854'
 setup=json.loads((ROOT/'setup_retry1_complete.json').read_text());assert setup['status']=='PASS'
 assert setup['source_manifest_sha256']==qualification['source_manifest_sha256']==L.sha256_file(ROOT/'SOURCE_SHA256SUMS')
 assets=Path(setup['assets']);assert assets==ROOT/'assets_retry2'
 for name,pin in setup['asset_sha256'].items():assert L.sha256_file(assets/name)==pin
 receipt=json.loads((ROOT/'test_plan_complete.json').read_text())
 assert receipt['status']=='PASS' and receipt['episodes']==17511 and receipt['packs']==26 and receipt['selected_validation_binding_parity']==221
 assert receipt['builder_sha256']=='65385572da3d06f7a09a2e5385b0e689738c497aad741ba560b5c38218242100'
 assert receipt['plan_sha256']==L.sha256_file(ROOT/'test_plan.json')=='ad01eadf3369f04b68d81ca5e00c505031917657516cda3995755b3ca789fc52'
 plan=json.loads((ROOT/'test_plan.json').read_text());pid=plan.pop('plan_identity');assert hashlib.sha256(L.canonical_bytes(plan)).hexdigest()==pid==receipt['plan_identity'];plan['plan_identity']=pid
 assert len(plan['episodes'])==17511 and len(plan['packs'])==26
 manifest=json.loads((ROOT/'manifest.json').read_text());expected={x['rel']:x for x in manifest['packs'] if x['rel'].startswith('bench_v2_test/')}
 assert {x['rel'] for x in plan['packs']}==set(expected)
 for p in plan['packs']:
  assert p['n_episodes']==expected[p['rel']]['n'] and p['data_identity']['pack_id']==expected[p['rel']]['pack_id']
  assert Path(p['local_dir']).is_relative_to(ROOT/'packs_retry1')
 output=ROOT/'test_outputs';assert output.is_dir()
 identity=dict(plan_identity=pid,qualification_sha256=L.sha256_file(ROOT/'qualification_retry1.json'),source_manifest_sha256=setup['source_manifest_sha256'],test_runner_sha256=L.sha256_file(ROOT/'run_tests.py'),asset_sha256=setup['asset_sha256'],recipe='lvm_native_7b_fp16_p1_seed0_t1_topp09_topk50_one_candidate')
 run=hashlib.sha256(L.canonical_bytes(identity)).hexdigest()
 atomic=L.atomic_json
 assert L.sha256_file(ROOT/'run_tests.py')=='fb9713c9395e23189faf41864372767b7244cab2881bf0ca98c8a655fc99f05b'
 assert json.loads((ROOT/'test_run_identity.json').read_text())==dict(run_identity=run,identity=identity)
 assert run=='5a05bdeacaae9a653bc9c601b4e0ce396147231b9ef522f3b5f102dd43e53efc'
 # Preserve terminal evidence and every pre-resume producer manifest.
 import shutil
 audit=ROOT/'test_resume1_preexisting';audit.mkdir(exist_ok=False)
 prior_counts={};prior_keys=set();files_verified=0
 for p in plan['packs']:
  view_path=output/'views'/p['slug']/'results.json'
  view=json.loads(view_path.read_text())
  assert view['plan_identity']==pid and view['data_identity']==p['data_identity']
  assert set(view['episodes']).issubset(p['expected_episode_keys'])
  prior_counts[p['slug']]=len(view['episodes'])
  shutil.copy2(view_path,audit/(p['slug']+'.json'))
  for key,e in view['episodes'].items():
   prior_keys.add(p['rel']+'|'+key)
   assert e['inference_run_identity']==run
   assert e['numerical_evidence']['raw_decode_finite'] and e['numerical_evidence']['tokenizer_finite']
   art=e['artifacts'];items=[art['query_input'],art['query_target'],art['readouts'][L.READOUT]['real']]
   for item in items:
    assert L.sha256_file(view_path.parent/item['path'])==item['sha256'];files_verified+=1
 expected_prefix={e['pack_rel']+'|'+e['key'] for e in plan['episodes'][:2356]}
 assert prior_keys==expected_prefix and sum(prior_counts.values())==2356 and files_verified==7068
 for name in ['test_controller.exit','test_worker.exit','test_worker.log','test_run_identity.json']:
  shutil.copy2(ROOT/name,audit/name)
 for name in ['backend.json','runtime_config.json','progress.json']:
  shutil.copy2(output/name,audit/name)
 recovery=dict(status='SAVED_OUTPUTS_VERIFIED_NATIVE_RESUME',completed=2356,artifact_files_verified=files_verified,run_identity=run,resume_runner_sha256=L.sha256_file(Path(__file__)),original_runner_sha256=L.sha256_file(ROOT/'run_tests.py'),seed_contract='Unchanged L.infer uses PRNGKey(MODEL_SEED) independently per episode',original_terminal_returncode=(ROOT/'test_worker.exit').read_text().strip())
 atomic(ROOT/'test_resume1_provenance.json',recovery);upload(ROOT/'test_resume1_provenance.json',BASE+'/provenance/resume1.json')
 sys.path.insert(0,str(ROOT/'official'))
 import torch
 from evaluation.vqlm_demo.vqvae_muse import VQGANModel
 encode,decode=VQGANModel.encode,VQGANModel.decode_code
 observation={}
 def checked_encode(model,values,*args,**kwargs):
  assert tuple(values.shape)==(3,3,256,256) and bool(torch.isfinite(values).all())
  observation.clear();finite=[]
  def observe(module,arguments,result):
   assert bool(torch.isfinite(result).all());finite.append(True)
  hook=model.quant_conv.register_forward_hook(observe)
  try:result=encode(model,values,*args,**kwargs)
  finally:hook.remove()
  quantized,tokens=result[:2]
  assert len(finite)==1 and bool(torch.isfinite(quantized).all()) and tuple(tokens.shape)==(3,256) and int(tokens.min())>=0 and int(tokens.max())<8192
  observation.update(context_sha256=L.sha256_array(values.detach().cpu().numpy()),token_ids_sha256=L.sha256_array(tokens.detach().cpu().numpy()),quantized_sha256=L.sha256_array(quantized.detach().cpu().numpy()),tokenizer_finite=True)
  return result
 def checked_decode(model,tokens,*args,**kwargs):
  assert tuple(tokens.shape)==(1,256) and int(tokens.min())>=0 and int(tokens.max())<8192
  result=decode(model,tokens,*args,**kwargs)
  assert tuple(result.shape)==(1,3,256,256) and bool(torch.isfinite(result).all())
  assert observation and 'raw_decode_sha256' not in observation
  observation.update(generated_tokens_sha256=L.sha256_array(tokens.detach().cpu().numpy()),raw_decode_sha256=L.sha256_array(result.detach().cpu().numpy()),raw_decode_finite=True)
  return result
 VQGANModel.encode,VQGANModel.decode_code=checked_encode,checked_decode
 save_episode=L.save_episode
 def checked_save(ep,prediction,qi,qt,pack,view,results_root,timings):
  assert observation.get('raw_decode_finite') and observation.get('tokenizer_finite')
  save_episode(ep,prediction,qi,qt,pack,view,results_root,timings)
  view['episodes'][ep['key']]['numerical_evidence']=dict(observation)
  view['episodes'][ep['key']]['inference_run_identity']=run
 L.save_episode=checked_save
 packs={p['slug']:p for p in plan['packs']};counts=dict(prior_counts);published={s:prior_counts[s] if prior_counts[s]==packs[s]['n_episodes'] else 0 for s in packs}
 def checked_atomic(path,value):
  atomic(path,value)
  if path.name!='results.json' or path.parent.name not in packs or path.parent.parent!=output/'views':return
  slug=path.parent.name;p=packs[slug];n=len(value['episodes']);counts[slug]=n
  progress=dict(status='COMPLETE' if n==p['n_episodes'] else 'RUNNING',model='lvm',run_identity=run,pack_rel=p['rel'],pack_id=p['data_identity']['pack_id'],completed=n,expected=p['n_episodes'],total_completed=sum(counts.values()),total_expected=17511,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  atomic(output/'progress.json',progress)
  if n and n!=published[slug] and (n%128==0 or n==p['n_episodes']):
   atomic(path.parent/'progress.json',progress)
   archive=output/(slug+'.tar')
   with tarfile.open(archive,'w') as tar:tar.add(path.parent,arcname=slug)
   upload(archive,BASE+'/packs/'+archive.name);upload(path.parent/'progress.json',BASE+'/progress/'+slug+'.json');published[slug]=n
 L.atomic_json=checked_atomic;L.LIVE_REPO=ROOT
 result=L.infer(SimpleNamespace(plan=ROOT/'test_plan.json',results_root=output,official_repo=ROOT/'official',checkpoint=assets/'easylm_streaming_params',vq_checkpoint=assets,expected_tpu=launch['name'],expected_zone=launch['zone'],expected_devices=4,torch_threads=8))
 assert result==0 and sum(counts.values())==17511 and all(counts[s]==published[s]==packs[s]['n_episodes'] for s in packs)
 upload(output/'INFERENCE_COMPLETE.json',BASE+'/provenance/INFERENCE_COMPLETE.json')
if __name__=='__main__':main()
