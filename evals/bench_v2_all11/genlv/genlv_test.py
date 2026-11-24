"""Frozen CPU GenLV inference on every Bench v2 test query, with resumable receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source/optimized'))
import genlv_runtime as G
from tools import bench_episode_v1 as BE
from tools.bench_v2_runtime_v2 import stage_pack

MANIFEST_SHA = '89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d'
PACK_URI = 'gs://kmh-gcp-us-west4/data/xtiange/mt_eval_packs'
OUTPUT_URI = 'gs://kmh-gcp-us-west4/kristine/eval_runs/2026-11-10/bench_v2_all11_test/genlv_cpu_reference_v1'
TPU = 'kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def pixel_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class GcloudStore:
    scheme = 'gs://'

    def download(self, uri, dst):
        if not uri.startswith(PACK_URI + '/'):
            raise ValueError('Only the pinned regional pack prefix is permitted')
        subprocess.run(['gcloud', 'storage', 'cp', uri, str(dst)], check=True)


def load_pack(rel):
    return stage_pack(rel, uri_root=PACK_URI, manifest_path=ROOT/'manifest.json',
                      manifest_sha256=MANIFEST_SHA,
                      local_root=Path('/dev/shm/kristine_bench_all11_packs'), store=GcloudStore())


def episode(pack, task, q):
    rows = [i for i, r in enumerate(pack['records']) if r['task'] == task]
    records = [pack['records'][i] for i in rows]
    sources = [str(r.get('source')) for r in records]
    same = [i for i, s in enumerate(sources) if s == sources[q]]
    if len(same) < 2:
        raise ValueError('No same-source demonstration')
    rng = np.random.default_rng([42, BE.task_seed(task), q, 1])
    cid = 0
    if BE.family_of(task) == 'palette':
        pool = BE.convention_pools(task)[1]
        cid = int(pool[int(rng.integers(0, len(pool)))])
    demo = [int(i) for i in BE.draw_demo_indices(len(rows), q, 1, rng, candidates=same)]
    # Frozen benchmark coding includes query labels in palette assignment.
    # Only the resulting demo target, demo input and query input reach the model.
    coded = BE.code_episode([pack['masks'][rows[demo[0]]], pack['masks'][rows[q]]], task, cid)
    inputs = G.Inputs(pack['images'][rows[demo[0]]], coded[0], pack['images'][rows[q]])
    digest = hashlib.sha256()
    for tile in (inputs.demo_input, inputs.demo_target, inputs.query_input, coded[1]):
        digest.update(np.ascontiguousarray(tile).tobytes())
    binding = dict(task=task, q=q, p=1, cid=cid, query_global=rows[q],
                   query_record=records[q]['record_id'], query_source=sources[q],
                   demo_global=[rows[demo[0]]], demo_records=[records[demo[0]]['record_id']],
                   episode_pixels_sha256=digest.hexdigest(),
                   query_input_pixels_sha256=pixel_sha(inputs.query_input),
                   query_target_pixels_sha256=pixel_sha(coded[1]))
    return inputs, binding


def source_identity():
    import torch
    paths = ['genlv_test.py', 'manifest.json', 'validation_plan.json',
             'tools/bench_episode_v1.py', 'tools/bench_v2_runtime_v1.py',
             'tools/bench_v2_runtime_v2.py', 'source/optimized/genlv_runtime.py',
             'source/baseline/genlv_compat.py',
             'source/official/extended_version/models/xrestormer_prompt_crossattn_wores.py']
    return dict(recipe='genlv-cpu-reference', backend='cpu', dtype='float32',
                batch_size=1, threads=8, seed=0, episode_seed=42, p=1,
                checkpoint_sha256=G.CHECKPOINT_SHA256,
                torch_version=torch.__version__, numpy_version=np.__version__,
                source_sha256={p:G.sha256_file(ROOT/p) for p in paths})


def configure():
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    if os.uname().nodename != 't1v-n-4b587646-w-0':
        raise ValueError('This launch is bound to the selected authorized TPU worker')
    if G.sha256_file(ROOT/'manifest.json') != MANIFEST_SHA:
        raise ValueError('Manifest changed')
    return source_identity()


def qualify(identity):
    import torch
    pack = load_pack('bench_v2_val')
    original = json.loads((ROOT/'validation_plan.json').read_text())
    assert original['plan_identity'] == 'f5110d7f425dc6e9a1c339f8ed780245c325e5199f099c1dabcc769a3da52e73'
    assert len(original['episodes']) == 221
    checked = []
    for e in original['episodes']:
        _, binding = episode(pack, e['task'], e['q'])
        for k, v in binding.items():
            if v != e[k]:
                raise ValueError(f'Validation episode replay differs: {e["key"]} {k}')
        checked.append(e['episode_id'])
    print('VALIDATION_BINDINGS_MATCH 221', flush=True)
    model = G.load_cpu_model(ROOT/'weights/huge-checkpoint-49.pth')
    smokes = []
    for task in ['colorization', 'semantic_segmentation']:
        inputs, binding = episode(pack, task, 0)
        arrays = G.preprocess([inputs])
        started = time.monotonic()
        raw = G.predict_raw(model, arrays)
        assert raw.dtype == torch.float32 and bool(torch.isfinite(raw).all())
        assert tuple(raw.shape) == (1, 3, 256, 256)
        pred = G.postprocess(raw)[0]
        out = ROOT/'outputs/qualification'/task
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred).save(out/'prediction.png')
        # Independent direct released-model call verifies wrapper input order and quantization.
        tensors = [torch.from_numpy(a) for a in arrays]
        with torch.no_grad():
            _, direct, _ = model(imgs=[*tensors, tensors[2]], input_is_list=True)
        assert torch.equal(raw, direct)
        direct_u8 = np.clip(direct.detach().numpy()[0].transpose(1,2,0)*255, 0, 255).astype(np.int32).astype(np.uint8)[...,::-1]
        assert np.array_equal(pred, direct_u8)
        smokes.append(dict(task=task, q=0, binding=binding,
                           prediction_pixels_sha256=pixel_sha(pred),
                           seconds=time.monotonic()-started, finite=True,
                           direct_model_exact=True, direct_quantization_exact=True))
        print(json.dumps(smokes[-1]), flush=True)
    receipt = dict(status='PASS', identity=identity, completed_utc=now(),
                   validation_episode_replays=len(checked), validation_smokes=smokes,
                   scope='Frozen episode replay and CPU wrapper/direct-model equivalence; no TPU or historical CPU numerical parity claim')
    atomic_json(ROOT/'receipts/genlv_qualification.json', receipt)
    return model


def publish(pack_dir, receipt, shard_tag):
    atomic_json(pack_dir/'progress.json', receipt)
    archive = ROOT/'outputs'/f'{shard_tag}__{pack_dir.name}.tar'
    with tarfile.open(archive, 'w') as t:
        t.add(pack_dir, arcname=pack_dir.name)
    subprocess.run(['gcloud', 'storage', 'cp', str(archive),
                    OUTPUT_URI+'/packs/'+archive.name], check=True)
    subprocess.run(['gcloud', 'storage', 'cp', str(pack_dir/'progress.json'),
                    OUTPUT_URI+'/progress/'+shard_tag+'__'+pack_dir.name+'.json'], check=True)


def run(identity, model=None, shard_index=0, shard_count=1):
    import torch
    gate = json.loads((ROOT/'receipts/genlv_qualification.json').read_text())
    if gate['status'] != 'PASS' or gate['identity'] != identity or gate['validation_episode_replays'] != 221:
        raise ValueError('Matching successful qualification required')
    model = model or G.load_cpu_model(ROOT/'weights/huge-checkpoint-49.pth')
    manifest = json.loads((ROOT/'manifest.json').read_text())
    entries = [p for p in manifest['packs'] if p['rel'].startswith('bench_v2_test/')]
    assert len(entries) == 26 and sum(p['n'] for p in entries) == 17511
    if not 0 <= shard_index < shard_count <= 8:
        raise ValueError('Invalid inference shard')
    shard_tag = f'shard_{shard_index:02d}_of_{shard_count:02d}'
    run_id = G.canonical_sha(dict(identity=identity, shard_index=shard_index, shard_count=shard_count))
    expected_total = len(range(shard_index, 17511, shard_count))
    completed = 0
    ordinal = 0
    for entry in entries:
        rel = entry['rel']
        pack = load_pack(rel)
        out = ROOT/'outputs'/shard_tag/rel.replace('/', '__')
        out.mkdir(parents=True, exist_ok=True)
        expected_pack = sum(i % shard_count == shard_index for i in range(ordinal, ordinal + entry['n']))
        count = 0
        for task in dict.fromkeys(r['task'] for r in pack['records']):
            n = sum(r['task'] == task for r in pack['records'])
            for q in range(n):
                assigned = ordinal % shard_count == shard_index
                ordinal += 1
                if not assigned:
                    continue
                inputs, binding = episode(pack, task, q)
                dest = out/f'{task}_q{q:05d}'
                meta = dest/'receipt.json'
                if meta.exists():
                    old = json.loads(meta.read_text())
                    if old['run_identity'] != run_id or old['binding'] != binding:
                        raise ValueError('Resume identity changed')
                    if G.sha256_file(dest/'prediction.png') != old['prediction_sha256']:
                        raise ValueError('Resume prediction changed')
                else:
                    started = time.monotonic()
                    raw = G.predict_raw(model, G.preprocess([inputs]))
                    if raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
                        raise ValueError(f'Nonfinite/wrong dtype: {rel} {q}')
                    pred = G.postprocess(raw)[0]
                    dest.mkdir(parents=True, exist_ok=True)
                    tmp = dest/'prediction.tmp.png'
                    Image.fromarray(pred).save(tmp)
                    os.replace(tmp, dest/'prediction.png')
                    atomic_json(meta, dict(run_identity=run_id, binding=binding, pack_rel=rel,
                                           pack_id=entry['pack_id'], prediction_sha256=G.sha256_file(dest/'prediction.png'),
                                           prediction_pixels_sha256=pixel_sha(pred), seconds=time.monotonic()-started,
                                           completed_utc=now(), model_inputs=['demo_input','coded_demo_target','query_input']))
                count += 1
                completed += 1
                progress = dict(status='RUNNING', model='genlv', run_identity=run_id,
                                shard_index=shard_index, shard_count=shard_count,
                                pack_rel=rel, pack_completed=count, pack_expected=expected_pack,
                                total_completed=completed, total_expected=expected_total, updated_utc=now())
                atomic_json(ROOT/f'receipts/test_progress_{shard_tag}.json', progress)
                print(json.dumps(progress), flush=True)
                if count % 25 == 0:
                    publish(out, progress, shard_tag)
        assert count == expected_pack
        progress['status'] = 'PACK_INFERENCE_COMPLETE'
        publish(out, progress, shard_tag)
        del pack
    assert ordinal == 17511 and completed == expected_total
    atomic_json(ROOT/f'receipts/test_inference_complete_{shard_tag}.json', dict(status='SHARD_INFERENCE_COMPLETE',
                identity=identity, shard_index=shard_index, shard_count=shard_count,
                queries=completed, packs=26, completed_utc=now(), scoring_complete=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['qualify','run','qualify-and-run'], required=True)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    args = ap.parse_args()
    identity = configure()
    atomic_json(ROOT/'receipts/run_identity.json', identity)
    model = qualify(identity) if args.mode in ['qualify','qualify-and-run'] else None
    if args.mode in ['run','qualify-and-run']:
        run(identity, model, args.shard_index, args.shard_count)
