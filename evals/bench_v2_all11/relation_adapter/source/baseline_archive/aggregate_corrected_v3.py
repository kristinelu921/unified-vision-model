#!/usr/bin/env python3
"""Aggregate authoritative corrected-V3 RelationAdapter scores by split."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, shutil, sys
from pathlib import Path
import numpy as np

JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2"); sys.path.insert(0, str(JFLUX))
from jflux2.fasteval.eval_v2.protocol_v1 import PRIMARY  # noqa: E402
from tools.bench_eval_v3_backend_v1 import create_backend  # noqa: E402

PROTOCOL = "bench_v2_PP2_CSA_seed1_AP_FIRST_v1"
RELEASE = "bd08e05f1d0e9dd077db683f4f124ac49a2c45cd0ec5728653bd1ad66cc13792"
READOUT = "RelationAdapter@native_p1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def finite(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{where}: nonfinite/non-numeric")
    return float(value)


def average(values):
    if not values: raise ValueError("empty group")
    return math.fsum(values) / len(values)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--units-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); args = p.parse_args()
    if args.out_dir.exists() or args.out_dir.is_symlink(): raise SystemExit("fresh aggregate output required")
    units = sorted(path for path in args.units_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if len(units) != 27: raise SystemExit(f"expected 27 units, got {len(units)}")
    pack_root = Path(os.environ["BENCH_PACK_ROOT"])
    groups, edge_counts, unit_rows = {}, {}, []
    metric_sha = metric_identity = None; episodes = pairs = 0
    for unit in units:
        producer_path, score_root = unit / "results.json", unit / "scored_eval_v2"
        producer = json.loads(producer_path.read_text()); scored = json.loads((score_root / "results.json").read_text())
        receipt = json.loads((score_root / "score_manifest.json").read_text())
        if receipt.get("status") != "score_complete" or receipt.get("authoritative") is not True: raise ValueError(f"{unit}: score incomplete")
        ident = receipt.get("metric_identity", {})
        if ident.get("metric_protocol") != PROTOCOL or ident.get("release_sha256") != RELEASE or receipt.get("requested_contract_satisfied") is not True: raise ValueError(f"{unit}: metric identity differs")
        if metric_sha is None: metric_sha, metric_identity = receipt["metric_identity_sha256"], ident
        elif metric_sha != receipt.get("metric_identity_sha256") or metric_identity != ident: raise ValueError("metric identity varies")
        for name, digest in receipt.get("outputs", {}).items():
            if sha256(score_root / name) != digest: raise ValueError(f"{unit}: score output differs")
        if set(scored["episodes"]) != set(producer["episodes"]): raise ValueError(f"{unit}: scored coverage differs")
        pack_rel = producer["data_identity"]["pack_rel"]; split = "val" if pack_rel == "bench_v2_val" else "test"
        metadata = json.loads((pack_root / pack_rel / "samples.json").read_text()); task_rows = {}
        for row in metadata["records"]: task_rows.setdefault(row["task"], []).append(row)
        for key in producer["expected_episode_keys"]:
            ep = scored["episodes"][key]
            if ep.get("complete") is not True or ep.get("score_complete") is not True: raise ValueError(f"{key}: incomplete")
            task, q = ep["task"], int(ep["q"]); primary, lower = PRIMARY[task]
            for arm, record in (("copy", ep["controls"]["copy"]), ("model", ep["readouts"][READOUT]["real"])):
                if record.get("native_name") != primary or bool(record.get("native_lower")) != bool(lower): raise ValueError(f"{key}.{arm}: primary differs")
                value, native = finite(record.get(primary), f"{key}.{arm}"), finite(record.get("native"), f"{key}.{arm}.native")
                if not math.isclose(value, native, rel_tol=0, abs_tol=1e-12): raise ValueError(f"{key}.{arm}: native differs")
                groups.setdefault((split, pack_rel, task), {"copy": [], "model": []})[arm].append(value)
                if task == "edge_detection":
                    dataset, counts = str(task_rows[task][q]["dataset"]), record.get("_edge_counts")
                    if np.asarray(counts).shape != (99, 4): raise ValueError(f"{key}.{arm}: edge counts differ")
                    edge_counts.setdefault((split, pack_rel, dataset, arm), []).append(counts)
            episodes += 1; pairs += 2
        unit_rows.append({"unit": unit.name, "pack_rel": pack_rel, "episodes": len(producer["episodes"]),
                          "producer_sha256": sha256(producer_path), "score_manifest_sha256": sha256(score_root / "score_manifest.json"),
                          "scored_results_sha256": sha256(score_root / "results.json"), "run_identity": producer["run_identity"]})
    if (episodes, pairs) != (17_732, 35_464): raise ValueError(f"coverage differs: {(episodes, pairs)}")
    primary_groups = []
    for (split, pack_rel, task), arms in sorted(groups.items()):
        if len(arms["copy"]) != len(arms["model"]): raise ValueError("arm coverage differs")
        cm, mm, lower = average(arms["copy"]), average(arms["model"]), bool(PRIMARY[task][1])
        primary_groups.append({"split": split, "pack_rel": pack_rel, "task": task, "primary": PRIMARY[task][0],
                               "lower_is_better": lower, "n": len(arms["model"]), "copy_mean": cm, "model_mean": mm,
                               "model_improvement": cm - mm if lower else mm - cm})
    backend = create_backend()
    if backend.identity.get("metric_protocol") != PROTOCOL: raise ValueError("edge backend differs")
    edge_groups = []
    for (split, pack_rel, dataset, arm), counts in sorted(edge_counts.items()):
        result = backend.dataset_edge_readout(dataset, counts)
        for name in ("ods", "ois", "ap", "ap_frozen", "ap_last_diagnostic"): finite(result.get(name), f"edge.{name}")
        edge_groups.append({"split": split, "pack_rel": pack_rel, "dataset": dataset, "arm": arm, "n": len(counts), **result})
    args.out_dir.mkdir(parents=True)
    summary = {"schema": "relation_adapter_bench_v2_native_p1_corrected_v3_summary_v1",
               "metric_protocol": PROTOCOL, "release_sha256": RELEASE, "metric_identity_sha256": metric_sha,
               "metric_identity": metric_identity, "aggregation": {"primary": "mean per-query named primary within pack/task",
               "edge": "corrected backend dataset readout with FIRST duplicate-recall AP"}, "episodes": episodes,
               "scored_pairs_including_copy": pairs, "primary_groups": primary_groups, "edge_dataset_readouts": edge_groups,
               "units": unit_rows}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.out_dir / "primary_groups.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(primary_groups[0])); writer.writeheader(); writer.writerows(primary_groups)
    (args.out_dir / "edge_dataset_readouts.json").write_text(json.dumps({"metric_protocol": PROTOCOL, "groups": edge_groups}, indent=2, sort_keys=True) + "\n")
    publication = args.out_dir / "publication"
    for unit in units:
        target = publication / "units" / unit.name; target.mkdir(parents=True, exist_ok=True); shutil.copy2(unit / "results.json", target / "results.json")
        for name in ("results.json", "score_manifest.json", "lpips_v1.json", "keypoint_localisation_v1.json"):
            dst = target / "scored_eval_v2" / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(unit / "scored_eval_v2" / name, dst)
    for name in ("summary.json", "primary_groups.csv", "edge_dataset_readouts.json"): shutil.copy2(args.out_dir / name, publication / name)
    objects = {str(path.relative_to(publication)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(publication.rglob("*")) if path.is_file()}
    (publication / "publication_manifest.json").write_text(json.dumps({"schema": "relation_adapter_bench_v2_native_p1_corrected_v3_publication_manifest_v1",
        "metric_protocol": PROTOCOL, "release_sha256": RELEASE, "objects": objects}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"episodes": episodes, "primary_groups": len(primary_groups), "edge_groups": len(edge_groups)}, sort_keys=True))


if __name__ == "__main__": main()
