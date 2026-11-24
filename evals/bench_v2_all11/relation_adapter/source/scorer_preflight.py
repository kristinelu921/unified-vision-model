"""Instantiate the frozen scorer on its certified host without opening examples."""
from pathlib import Path
import json
import time
from jflux2.fasteval.eval_v3.scoring_v2 import create_scorer

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/kmh-nfs-ssd-us-mount/code/xtiange/jflux2/docs/bench_v2_gt_ceiling_tpu_v1_2026-10-07')
start=time.monotonic()
scorer=create_scorer(
    release_pin={'path':str(BASE/'corrected_metric_release_v3/release_manifest.json'),'sha256':'bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792'},
    host_build_pin={'path':str(BASE/'cpu_reference_v2/bsds_matching_review_v1/build_receipt_v3.json'),'sha256':'55112291ec00a2cadde799409809d08133de3e3dabc681d03bd96225c1bfcc16'},
    host_admission_pin={'path':str(BASE/'corrected_metric_release_v1_preparation_v1/LOCAL_HOST_CERTIFICATE_V3.json'),'sha256':'710d3ec5ac665c42bec5884ae08556e7251a6771b074bdb3159ee7c9b001637b'},
    appearance_mode='preload')
receipt={'status':'FROZEN_SCORER_ADMISSION_PASS','elapsed_seconds':time.monotonic()-start,'validation_examples_opened':False,'metrics_computed':False,'release_sha256':scorer.release_sha256,'host_build_sha256':scorer.host_build_sha256}
(ROOT/'receipts/scorer_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
