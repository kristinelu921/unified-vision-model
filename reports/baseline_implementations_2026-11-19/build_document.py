from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import zipfile
import markdown

ROOT = Path('/kmh-nfs-ssd-us-mount/code/kristine/lvm')
OUT = ROOT / 'reports/baseline_implementations_2026-11-19'
RUN = ROOT / 'eval_runs/2026-11-10/bench_v2_all11_test'
SITE = ROOT / 'visualcloze-results-site'
TITLE = 'Final implemented adjustments for the eleven baselines'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def r(path):
    return RUN / path

def s(path):
    return SITE / path

complete = ['visualcloze', 'painter', 'genlv', 'promptdiffusion', 'improv', 'lvm']
receipts = {name: read(r(name+'_test_completion.json')) for name in complete}
(OUT/'verified_test_receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')

SOURCES = {
'S01': s('dist/10-12-validations/best-results.json'),
'S02': r('all11_test_configuration_audit.json'),
'S03': r('campaign.json'),
'S04': OUT/'verified_test_receipts.json',
'S05': ROOT/'AGENTS.md',
'S06': ROOT/'experiment_log/2026-09-26/visualcloze-open-model-tpu-evals.md',
'S07': r('runner/tools/bench_episode_v1.py'),
'S08': s('dist/bench-v2-instamanip-p3/data.json'),
'S09': r('selected_metric_backend.py'),
'S10': r('recovered/visualcloze_recovery/source/scorer_snapshot/jflux2/fasteval/eval_v2/protocol_v1.py'),
'S11': r('recovered/visualcloze_recovery/source/scorer_snapshot/jflux2/fasteval/eval_v2/postprocess_v2.py'),
'S12': s('dist/data/vict-edge-aggregation-correction.json'),
'S13': s('dist/bench-v2-signed-depth/signed_depth_metric.py'),
'S14': r('recovery_20261113/remaining_four_completion_audit.json'),
'S15': r('recovered/visualcloze_recovery/source/baseline/p2/visualcloze_p2_task_text.py'),
'S16': r('visualcloze_runtime_source_verification.json'),
'S17': r('visualcloze_decode_failure_localization.json'),
'S18': r('run_visualcloze_tests.py'),
'S19': r('visualcloze_diagnostic9_qualification_decision.json'),
'S20': s('dist/data/bench-v2-validation-demo-counts.json'),
'S21': s('dist/bench-v2-review/data.json'),
'S22': r('painter_runner/painter_native.py'),
'S23': r('painter_runner/official/Painter/models_painter.py'),
'S24': s('scripts/depth_output_adapter.py'),
'S25': r('painter_qualification.json'),
'S26': r('vict_runner/vict_native.py'),
'S27': r('vict_runner/adaptation.py'),
'S28': r('vict_runner/run_tests.py'),
'S29': r('runner/source/optimized/genlv_runtime.py'),
'S30': r('runner/source/official/extended_version/models/xrestormer_prompt_crossattn_wores.py'),
'S31': r('runner/genlv_test.py'),
'S32': r('recovered/lavin_p8_retry18/defaults.json'),
'S33': r('recovered/lavin_p8_retry18/official/lib/model/dit.py'),
'S34': r('recovered/lavin_p8_retry18/lavin_dit_bench_v2.py'),
'S35': r('recovered/lavin_p8_retry18/ARCHIVED_COMPLETION_VERIFIED.json'),
'S36': r('promptdiffusion_runner/source_optimized/contracts.py'),
'S37': r('promptdiffusion_runner/source_optimized/frozen_assets.json'),
'S38': r('promptdiffusion_recipe_binding_audit.json'),
'S39': r('promptdiffusion_runner/source_optimized/engine.py'),
'S40': r('promptdiffusion_runner/run_tests_retry1.py'),
'S41': r('recovered/relation/plan_val.json'),
'S42': r('recovered/relation/source/runtime_loader.py'),
'S43': r('recovered/relation/source/baseline_archive/relation_adapter_bench_v2.py'),
'S44': r('recovered/relation/source/run_target_free.py'),
'S45': r('recovered/instamanip_p3/defaults.json'),
'S46': r('recovered/instamanip_p3/validation_plan.json'),
'S47': r('recovered/instamanip_p3/instamanip_bench_v2.py'),
'S48': r('improv_runner/improv_cpu.py'),
'S49': r('improv_model_qualification.json'),
'S50': s('scripts/edge_contrast_adapter.py'),
'S51': r('recovered/analogist_selected/analogist_q4_v5p16.py'),
'S52': r('recovered/analogist_selected/q2_analogist_bench_v2.py'),
'S53': r('recovered/analogist_selected/CURRENT_VISION_PROMPT_CACHE_RECEIPT.json'),
'S54': r('recovered/lvm/lvm_bench_v2.py'),
'S55': r('lvm_runner/official/EasyLM/models/llama/llama_model.py'),
'S56': r('recovered/lvm/lvm_runtime_config.py'),
'S57': r('recovered/relation/source/full_runtime.lock'),
'S58': r('recovered/lavin_p8_retry18/requirements.runtime.lock'),
}
for key, p in SOURCES.items():
    assert p.is_file(), (key, str(p))

models = [
('VisualCloze','visualcloze-p2',2,'TPU BF16 with one FP32 decoder normalization','384','1','Full test verified'),
('Painter','painter-p1-depth-decoded',1,'TPU FP32','448','1','Full test verified'),
('VICT','vict-p1-depth-decoded',1,'TPU BF16 adaptation; 60 updates/query','448','1','Validation complete; test receipt absent'),
('GenLV 101 Huge','genlv-cpu-reference',1,'CPU FP32','256','1','Full test verified'),
('LaVin-DiT','lavin-dit-p8',8,'TPU BF16','512','1','Validation complete; test receipt absent'),
('PromptDiffusion','promptdiffusion-range-updated / A1',1,'TPU BF16','256','1','Full test verified'),
('RelationAdapter','relation-adapter-p1',1,'TPU BF16','256','1','Validation complete; test receipt absent'),
('InstaManip','instamanip-p3',3,'TPU BF16','448 encoder; 1024 output','1','Validation complete; test receipt absent'),
('IMProv','improv-cpu-reference-edge-sqrt',1,'CPU FP32','224 grid; 111 output tile','1','Full test verified'),
('Analogist','analogist-p1',1,'TPU BF16','512 grid; 253 tiles','8','Validation complete; test receipt absent'),
('LVM 7B','lvm-p1',1,'TPU FP16; CPU VQ','256','1','Full test verified'),
]

def table(headers, rows):
    def clean(x):
        return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(clean(c) for c in row)+' |' for row in rows])

model_table = table(['Model','Selected recipe','P','Saved test evidence'],
                    [(name,recipe,p,status) for name,recipe,p,execution,pixels,candidates,status in models])
selected = read(SOURCES['S01'])
metric_table = table(['Task','Primary metric','Preferred direction'],
                     [(x['task'],x['metric'],x['direction']) for x in selected['tasks']])
campaign = read(SOURCES['S03'])
assert len(campaign['packs']) == 26 and sum(x['n'] for x in campaign['packs']) == 17511
pack_table = table(['Task','Dataset pack','Queries','Pack identity'],
                   [(x['rel'].split('/')[1],x['rel'].split('/')[2],x['n'],x['pack_id']) for x in campaign['packs']])

groups = read(SOURCES['S20'])['groups']
variants = [m for g in groups for m in g['models']]
variant_table = table(['Recipe','P','Saved status'],[(x['key'],x['p'],x['status']) for x in variants])

prompt_maps = {
    'VisualCloze task prompts': read(r('recovered/visualcloze_recovery/source/baseline/p2/task_prompts.json')),
    'PromptDiffusion task prompts': read(SOURCES['S37'])['TASK_PROMPTS'],
    'InstaManip task instructions': read(SOURCES['S46'])['task_instructions'],
}
for name, data in prompt_maps.items():
    assert isinstance(data,dict) and len(data) == 13, (name, list(data))
prompt_tables = '\n\n'.join('## '+name+'\n\n'+table(['Task','Exact text'],sorted(data.items())) for name,data in prompt_maps.items())

source_rows=[]
for key,p in SOURCES.items():
    source_rows.append(f'**{key}** — [{p.name}]({p}). SHA-256 `{sha(p)}`.')
source_index='\n\n'.join(source_rows)

body=(OUT/'report_body.md').read_text()
for key,value in {'MODEL_TABLE':model_table,'METRIC_TABLE':metric_table,'PACK_TABLE':pack_table,
                  'VARIANT_TABLE':variant_table,'PROMPT_TABLES':prompt_tables,'SOURCE_INDEX':source_index}.items():
    body=body.replace('{{'+key+'}}',value)
assert not re.search(r'\{\{\w+\}\}',body)
assert set(re.findall(r'\[(S\d+)\]',body)) <= set(SOURCES)
body=body.replace('All eleven families have saved validation measurements;',
    'These are evaluations of released pretrained checkpoints, not newly reproduced pretraining runs. VICT additionally optimizes its encoder at test time; the other selected recipes keep learned weights fixed. All eleven families have saved validation measurements;')
body=body.replace('The source bundle preserves the inspected files with an index',
    'The source bundle preserves the inspected files with an index')
body=body.replace('Full model assets are identified by their original manifests and hashes.',
    'Full model assets are identified by their original manifests and hashes. A recovered working source can contain later changes while its frozen plan still records the earlier source hash; the bundle records the bytes inspected here and does not assert that every recovered file equals every historical plan hash.')

# Main body stays native and editable in Pages. Keep its title outside the blocks.
(OUT/'baseline_model_implementation_reference.md').write_text('# '+TITLE+'\n\n'+body)
blocks = re.split(r'(?m)(?=^# )',body)
blocks = [b.strip() for b in blocks if b.strip()]
assert len(blocks) <= 100
(OUT/'page_payload.json').write_text(json.dumps({'title':TITLE,'initial_blocks':blocks,
    'idempotency_key':'baseline-implementation-reference-20261119-01a1f748-v1'},ensure_ascii=False))

# A clean standalone reading/export copy.
html_body=markdown.markdown('# '+TITLE+'\n\n'+body,extensions=['tables','fenced_code','toc'])
style='''
@page { size: A4; margin: 1.8cm; }
body { font: 10.5pt Calibri, Arial, sans-serif; line-height: 1.42; color: #172335; }
h1 { font-size: 19pt; color: #153e62; margin-top: 24pt; page-break-after: avoid; }
h2 { font-size: 13pt; color: #215678; margin-top: 16pt; page-break-after: avoid; }
p { margin: 0 0 8pt; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 8pt; table-layout: auto; }
th { background: #e6eef5; color: #153e62; }
td, th { border: 1px solid #bdcbd5; padding: 5pt; vertical-align: top; overflow-wrap: anywhere; }
tr { page-break-inside: avoid; }
pre { background: #f2f5f7; padding: 8pt; white-space: pre-wrap; font-size: 8pt; }
code { font-family: Consolas, monospace; font-size: 9pt; overflow-wrap: anywhere; }
a { color: #176897; overflow-wrap: anywhere; }
'''
(OUT/'baseline_model_implementation_reference.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(TITLE)+'</title><style>'+style+'</style></head><body>'+html_body+'</body></html>')

# Explicit source allowlist. No credentials, model weights, images or launch execution.
files=set(SOURCES.values())
extras=[r(x+'_test_completion.json') for x in complete]
extras += [
r('selected_validation_snapshot.json'),r('painter_runner/painter_compat.py'),r('painter_runner/run_tests.py'),
r('painter_runner/episode_p1.py'),r('vict_runner/source_receipt.json'),r('vict_runner/adaptation_source_receipt.json'),
r('genlv_final_source_qualification.json'),r('improv_runner/cpu_source_receipt.json'),
r('improv_runner/run_tests.py'),r('improv_runner/episode_p1.py'),
r('recovered/lavin_p8_retry18/validation_plan.json'),r('recovered/lavin_p8_retry18/conversion_manifest.json'),
r('recovered/lavin_p8_retry18/lavin_dit_parallel.py'),r('recovered/lavin_p8_retry18/tpu_compat.py'),
r('recovered/lvm/validation_plan.json'),r('lvm_runner/runtime_compat_patch.json'),r('lvm_runner/run_tests_resume1.py'),
r('recovered/analogist_selected/plan_val.json'),r('recovered/analogist_selected/analogist_loader.py'),
r('recovered/analogist_selected/analogist_compat.py'),r('recovered/analogist_selected/CURRENT_VISION_PROMPT_CACHE.json'),
r('recovered/analogist_selected/current_vision_prompt_embeddings.index.json'),
r('recovered/visualcloze_recovery/source/baseline/p2/task_prompts.json'),
r('visualcloze_selected_MODEL_SHA256SUMS'),r('visualcloze_diagnostic9_archived_manifest_comparison.json'),
r('visualcloze_diagnostic7_diagnostic9_comparison.json'),
r('recovered/instamanip_p3/asset_contract.py'),r('recovered/instamanip_p3/instamanip_loader.py'),
r('recovered/instamanip_p3/xla_static_generation.py'),r('recovered/instamanip_p3/xla_dynamic_euler.py'),
r('recovered/instamanip_p3/apply_xla_diffusion_patch.py'),
s('dist/bench-v2-cpu-reference/data.json'),s('dist/bench-v2-painter-reference/data.json'),
s('dist/bench-v2-recovery/instamanip-diffusion.json'),s('dist/bench-v2-recovery/completed-checks.json'),
s('dist/data/promptdiffusion-reference-parity.json'),
]
for p in extras:
    assert p.is_file(),str(p)
    files.add(p)
# Include implementation modules for the chosen sources and native scorer.
for directory in [r('promptdiffusion_runner/source_optimized'),r('recovered/relation/source'),
                  r('recovered/lavin_p8_retry18/official/lib/model'),r('improv_original_source/improv/modeling'),
                  r('recovered/visualcloze_recovery/source/scorer_snapshot/jflux2/fasteval/eval_v2')]:
    for p in directory.rglob('*.py'):
        files.add(p)
files.add(OUT/'baseline_model_implementation_reference.md')
files.add(OUT/'baseline_model_implementation_reference.html')
files.add(OUT/'verified_test_receipts.json')
index=[]
for p in sorted(files):
    rel = str(p.relative_to(ROOT))
    index.append({'source_ids':[key for key,val in SOURCES.items() if val==p],
                  'original_path':str(p),'archive_path':'workspace/'+rel,
                  'bytes':p.stat().st_size,'sha256':sha(p)})
(OUT/'source_index.json').write_text(json.dumps(index,indent=2)+'\n')
with zipfile.ZipFile(OUT/'baseline_implementation_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    z.write(OUT/'source_index.json','source_index.json')
    for row in index:z.write(row['original_path'],row['archive_path'])
    z.writestr('README.txt',
        TITLE+'\n\nInspected implementation and evidence snapshot, November 19, 2026 America/New_York.\n'
        'Read workspace/reports/baseline_implementations_2026-11-19/baseline_model_implementation_reference.html or .md.\n'
        'source_index.json records original paths and SHA-256 hashes for every included file.\n'
        'No checkpoints or benchmark image packs are included. Historical launch scripts are source evidence, not automatic launch instructions.\n'
        'A current recovered source file can differ from the older source hash inside a frozen plan. Preserve that version distinction.\n')

summary={'models':len(models),'body_words':len(body.split()),'page_blocks':len(blocks),
         'sources_cited':len(SOURCES),'source_files_in_bundle':len(index),
         'completed_test_models':len(receipts),'validation_variants_in_demo_table':len(variants),
         'test_packs':len(campaign['packs']),'test_queries_per_model':sum(x['n'] for x in campaign['packs']),
         'bundle_bytes':(OUT/'baseline_implementation_sources.zip').stat().st_size}
(OUT/'document_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
