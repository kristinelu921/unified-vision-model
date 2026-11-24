"""Selected A1 full-test inference; numerical engine and permitted inputs are frozen."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source_optimized'))
from contracts import Recipe, array_sha256, file_sha256, identity, normalize_rgb, runtime_identity
from prepare_promptdiffusion_inputs import permitted, stage_pack, Store, PREFIX, MANIFEST_SHA

OUTPUT = 'gs://kmh-gcp-us-west4/kristine/eval_runs/2026-11-10/bench_v2_all11_test/prompt_diffusion/test_a1_native8_v1'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def upload(path, uri):
    subprocess.run(['gsutil', '-o', 'GSUtil:check_hashes=always', 'cp', str(path), uri], check=True)

def publish(directory, progress):
    save(directory / 'progress.json', progress)
    archive = directory.with_suffix('.tar')
    with tarfile.open(archive, 'w') as out:
        out.add(directory, arcname=directory.name)
    upload(archive, OUTPUT + '/packs/' + archive.name)
    upload(directory / 'progress.json', OUTPUT + '/progress/' + directory.name + '.json')

def main():
    assert socket.gethostname() == 't1v-n-4b587646-w-0'
    for name in ['SOURCE_SHA256SUMS', 'TEST_RETRY1_SHA256SUMS']:
        subprocess.run(['sha256sum', '-c', name], cwd=ROOT, check=True)
    launch = json.loads((ROOT / 'test_retry1_launch.json').read_text())
    assert launch['name'] == 'kmh-tpuvm-v5e-8-spot-kristine-04-02-37-45-861043662-0-us-west4-a'
    assert launch['zone'] == 'us-west4-a' and launch['status'] == 'AVAILABLE'
    assert launch['all_workers_available'] and launch['devices'] == 8
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(launch['checked_utc'])).total_seconds()
    assert 0 <= age < 120
    gate = json.loads((ROOT / 'qualification_native8/parity.json').read_text())
    assert gate['status'] == 'PASS' and gate['exact_uint8_predictions'] == gate['fixtures'] == 16
    assert gate['raw_output_finite'] and gate['raw_finite_batches'] == 20
    assert len(gate['warm_timing']['repetitions']) == 3
    assert all(r['exact_uint8_predictions'] == 16 for r in gate['warm_timing']['repetitions'])
    assert gate['qualification_program_sha256'] == file_sha256(ROOT / 'qualify_native8.py')
    recipe = Recipe(candidate='A1', normalization='minus_one_one')
    assert gate['recipe'] == recipe.validate()
    source = {f.name: file_sha256(f) for f in (ROOT / 'source_optimized').glob('*.py')}
    assert source == gate['source']
    assert gate['devices'] == 8 and gate['accelerator_type'] == launch['accelerator_type']
    input_gate = json.loads((ROOT / 'promptdiffusion_input_qualification.json').read_text())
    assert input_gate['status'] == 'PASS' and input_gate['episodes'] == 221
    assert input_gate['all_permitted_pixels_exact'] and input_gate['query_masks_accessed'] is False
    assert input_gate['preparer_sha256'] == file_sha256(ROOT / 'prepare_promptdiffusion_inputs.py')
    manifest = ROOT / 'runner/manifest.json'
    assert file_sha256(manifest) == MANIFEST_SHA
    entries = [p for p in json.loads(manifest.read_text())['packs'] if p['rel'].startswith('bench_v2_test/')]
    assert len(entries) == 26 and sum(p['n'] for p in entries) == 17511
    setup = json.loads((ROOT / 'setup_complete.json').read_text())
    assert setup['status'] == 'PASS'
    model = Path(setup['model'])
    official = ROOT / 'sources/official'
    pins = json.loads((ROOT / 'source_optimized/frozen_assets.json').read_text())
    assert pins['ASSET_SHA256'] == gate['weights']
    for name, expected in pins['ASSET_SHA256'].items():
        assert file_sha256(model / name) == expected, name
    for name, expected in pins['SOURCE_SHA256'].items():
        assert file_sha256(official / name) == expected, name
    run = dict(recipe=recipe.validate(), source=source, weights=gate['weights'],
               manifest_sha256=MANIFEST_SHA, test_source_sha256=file_sha256(ROOT / 'TEST_RETRY1_SHA256SUMS'),
               qualification_sha256=file_sha256(ROOT / 'qualification_native8/parity.json'),
               input_qualification_sha256=file_sha256(ROOT / 'promptdiffusion_input_qualification.json'),
               mesh=[4, 2], devices=8, episode_seed=42, demonstrations=1)
    run_id = identity(run)
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.debug.metrics
    # Qualification records flags after importing XLA, which installs defaults.
    assert gate['complete_environment_sha256'] == identity(runtime_identity())
    output = ROOT / 'test_outputs'
    output.mkdir(exist_ok=False)
    save(output / 'run_identity.json', dict(run_identity=run_id, identity=run))
    upload(output / 'run_identity.json', OUTPUT + '/provenance/run_identity.json')
    from engine import Engine
    spec = importlib.util.spec_from_file_location('frozen_reference', ROOT / 'source_reference/bundle/prompt_diffusion_bench_v2.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cache = ROOT / 'cache' / identity(dict(run=run_id, environment=gate['complete_environment_sha256']))
    cache.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(cache), readonly=False)
    xr.use_spmd()
    assert xr.device_type() == 'TPU' and xr.global_runtime_device_count() == xr.addressable_runtime_device_count() == 8
    device = xm.xla_device()
    mesh = xs.Mesh(np.arange(8), (4, 2), ('data', 'replica'))
    xs.set_global_mesh(mesh)
    def shard(tensor):
        xs.mark_sharding(tensor, mesh, ('data',) + (None,) * (tensor.ndim - 1))
    positive, negative, embedding_hashes = reference.encode_prompts(official, model)
    assert embedding_hashes == gate['embedding_hashes']
    runtime = reference.load_runtime(official, model, torch.bfloat16)
    runtime = tuple(m.eval().to(device) for m in runtime[:3]) + (runtime[3],)
    xm.mark_step()
    xm.wait_device_ops()
    raw = []
    original_decode = runtime[2].decode
    def checked_decode(*values, **options):
        result = original_decode(*values, **options)
        value = result[0] if isinstance(result, tuple) else result.sample
        observed = value.detach().float().cpu().numpy()
        assert np.isfinite(observed).all(), 'Nonfinite raw VAE output'
        raw.append(array_sha256(observed))
        return result
    runtime[2].decode = checked_decode
    engine = Engine(*runtime, device=device, dtype=torch.bfloat16,
                    cache_conditioning=False, dynamic_coefficients=False, barrier=xm.mark_step, shard=shard)
    total = 0
    for entry in entries:
        rel = entry['rel']
        slug = rel.replace('/', '__')
        pack = stage_pack(rel, uri_root=PREFIX, manifest_path=manifest, manifest_sha256=MANIFEST_SHA,
                          local_root=Path('/dev/shm/kristine_bench_all11_promptdiffusion_test_packs'), store=Store())
        directory = output / slug
        directory.mkdir()
        iterator = iter(permitted(pack))
        completed = 0
        while completed < entry['n']:
            real = [next(iterator) for _ in range(min(4, entry['n'] - completed))]
            batch = real + [real[-1]] * (4 - len(real))
            arrays = [normalize_rgb(np.stack([row[1][i] for row in batch]), 'minus_one_one') for i in range(3)]
            noise = np.stack([torch.randn((4, 32, 32), generator=torch.Generator(device='cpu').manual_seed(ep['seed']), dtype=torch.float32).numpy() for ep, _ in batch])
            embeddings = torch.cat([positive[ep['task']] for ep, _ in batch])
            raw.clear()
            predictions = engine.predict(*arrays, noise, embeddings, negative)
            assert len(raw) == 1 and predictions.shape == (4, 256, 256, 3) and predictions.dtype == np.uint8
            for index, (ep, _) in enumerate(real):
                ordinal = completed + index
                ep = dict(ep, ordinal=ordinal)
                path = directory / f'prediction_{ordinal:05d}.png'
                Image.fromarray(predictions[index]).save(path)
                save(directory / f'prediction_{ordinal:05d}.json', dict(
                    run_identity=run_id, pack_rel=rel, pack_id=pack['data_identity']['pack_id'],
                    episode=ep, file=path.name, file_sha256=file_sha256(path),
                    pixels_sha256=array_sha256(predictions[index]), raw_output_finite=True,
                    raw_batch_sha256=raw[0], batch_start=completed, batch_position=index))
            completed += len(real)
            total += len(real)
            progress = dict(status='COMPLETE' if completed == entry['n'] else 'RUNNING',
                            model='prompt_diffusion', run_identity=run_id, pack_rel=rel,
                            pack_id=pack['data_identity']['pack_id'], completed=completed, expected=entry['n'],
                            total_completed=total, total_expected=17511,
                            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if completed % 128 == 0 or completed == entry['n']:
                publish(directory, progress)
        assert next(iterator, None) is None
        del pack
    assert total == 17511
    save(output / 'complete.json', dict(status='INFERENCE_COMPLETE', run_identity=run_id, completed=total, packs=26))
    upload(output / 'complete.json', OUTPUT + '/provenance/complete.json')

if __name__ == '__main__':
    main()
