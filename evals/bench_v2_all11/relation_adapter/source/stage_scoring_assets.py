"""Stage only frozen validation assets for the certified CPU scorer.

Copies private GCS objects server-to-server to the scorer region, downloads from
that region, and hashes bytes. It never parses or renders a validation example.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/tmp/relationadapter_autoresearch_010427/scoring')
SOURCE='gs://kmh-gcp-us-east5/data/xtiange/mt_eval_packs'
DEST='gs://kmh-gcp-us-central2/kristine/eval_runs/2026-10-30/relationadapter_autoresearch_010427/scoring_assets'
SHARED=Path('/kmh-nfs-ssd-us-mount/code/xtiange/jflux2')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    bucket=json.loads((ROOT/'receipts/scoring_regional_bucket.json').read_text())
    assert bucket['location']=='US-CENTRAL2'
    jobs=[]
    for name,sha in [('samples.json','7a130a55c9928b3509d95d03f3a3c9a63ffb5331c03335b2bee211374bb59322'),('samples.npz','d36685bbd8b95e720f03980b910712cd99d51f4933a4d1e9cf2e118d2235548e')]:
        jobs.append({'source':SOURCE+'/bench_v2_val/'+name,'destination':DEST+'/bench_v2_val/'+name,
                     'local':str(WORK/'packs/89f2501aa722cd12/bench_v2_val'/name),'sha256':sha})
    for category,rel in [('bench_v2_gt_v1','docs/eval_v2_2026-10-06/gt_sidecar_manifest_v1.json'),('bench_v2_gt_v1_bsds','docs/eval_v2_2026-10-06/bsds_v1/gt_bsds_manifest_v1.json')]:
        manifest=json.loads((SHARED/rel).read_text())
        for key,value in manifest['files'].items():
            if not key.startswith('bench_v2_val/'):continue
            jobs.append({'source':SOURCE+'/'+category+'/'+key,'destination':DEST+'/'+category+'/'+key,
                         'local':str(WORK/'corrected_v3/assets'/category/key),'sha256':value['sha256']})
    assert len(jobs)==11 and all('bench_v2_test' not in x['source'] for x in jobs)
    start=time.time()
    (ROOT/'receipts/scoring_asset_stage_plan.json').write_text(json.dumps({'files':jobs,'source_region':'us-east5','scorer_region':'us-central2','mode':'server_side_copy_then_region_local_download','test_split_included':False},indent=2)+'\n')
    def stage(item):
        path=Path(item['local']);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise RuntimeError('fresh stage destination already exists')
        for argv in [['gcloud','storage','cp',item['source'],item['destination']],['gcloud','storage','cp',item['destination'],str(path)]]:
            result=subprocess.run(argv,capture_output=True,text=True,timeout=300)
            if result.returncode:raise RuntimeError('GCS staging failed with exit '+str(result.returncode))
        if digest(path)!=item['sha256']:raise RuntimeError('validation asset hash mismatch')
        return {**item,'bytes':path.stat().st_size,'verified':True}
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(stage,jobs))
    result={'status':'COMPLETE_SHA256_VERIFIED','started_epoch':start,'elapsed_seconds':time.time()-start,'files':files,'validation_examples_inspected':False,'source_region':'us-east5','download_region':'us-central2'}
    (ROOT/'receipts/scoring_asset_stage.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'files':len(files),'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
