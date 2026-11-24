"""
TFDS builder for generated Hypersim combo transformation pairs.

One example = one (base image, transformed image, task) pair.
Each split corresponds to one category-pinned collection, for example:

  - bathroom_bathtub_sink
  - livingroom_sofa_table
  - dining_chair_table

By default, the builder indexes the checked-in combo summary JSON files under
`evals/combo/`. Override with:

  HYPERSIM_COMBO_SUMMARIES=/abs/a.json,/abs/b.json

Build programmatically:

  from evals.tfds.hypersim_combo_pairs import HypersimComboPairs
  builder = HypersimComboPairs(data_dir="/mnt/klum/combo_hypersim_tfds")
  builder.download_and_prepare()
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

try:
    from .eval_common import relpath_posix
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.tfds.eval_common import relpath_posix



_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SUMMARY_PATHS = [
    _REPO_ROOT / "evals" / "combo" / "hypersim_scaled_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_livingroom_sofa_table_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_bedroom_bed_nightstand_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_kitchen_refrigerator_cabinet_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_dining_chair_table_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_restaurant_chair_table_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_blinds_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_chair_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_lamp_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_picture_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_shelves_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_single_object_table_summary.json",
]


def _summary_paths() -> list[Path]:
    raw = os.environ.get("HYPERSIM_COMBO_SUMMARIES", "").strip()
    if not raw:
        return [p for p in _DEFAULT_SUMMARY_PATHS if p.is_file()]
    return [Path(part).expanduser().resolve() for part in raw.split(",") if part.strip()]


def _sanitize_split_name(text: str) -> str:
    chars = []
    for ch in text.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _split_name(summary: dict) -> str:
    secondary = str(summary.get("secondary_category") or "").strip()
    if secondary:
        return _sanitize_split_name(
            f"{summary['scene_type']}_{summary['primary_category']}_{secondary}"
        )
    return _sanitize_split_name(f"{summary['scene_type']}_{summary['primary_category']}")


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class HypersimComboPairs(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.4.0")

    def _info(self) -> tfds.core.DatasetInfo:
        role_feats = tfds.features.FeaturesDict(
            {
                "instance_id": tfds.features.Text(),
                "category": tfds.features.Text(),
                "bbox_xywh": tfds.features.Tensor(shape=(4,), dtype=np.int32),
            }
        )
        all_role_feats = tfds.features.FeaturesDict(
            {
                "instance_id": tfds.features.Text(),
                "category": tfds.features.Text(),
                "area": tfds.features.Tensor(shape=(), dtype=np.int32),
                "bbox_xywh": tfds.features.Tensor(shape=(4,), dtype=np.int32),
                "rel_mask_path": tfds.features.Text(),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Generated Hypersim combo transformation pairs. "
                "Each example contains a base RGB image, a transformed target image, "
                "the task id/description, the pinned primary/secondary roles, and all "
                "instances matching the primary/secondary categories for the source frame. "
                "Transform layers are applied category-wide across all matching role instances."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "base_image": tfds.features.Image(shape=(None, None, 3)),
                    "transformed_image": tfds.features.Image(shape=(None, None, 3)),
                    "roles": tfds.features.FeaturesDict(
                        {
                            "primary": role_feats,
                            "secondary": role_feats,
                        }
                    ),
                    "all_primary_instances": tfds.features.Sequence(all_role_feats),
                    "all_secondary_instances": tfds.features.Sequence(all_role_feats),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "dataset_name": tfds.features.Text(),
                            "split_name": tfds.features.Text(),
                            "scene_type": tfds.features.Text(),
                            "primary_category": tfds.features.Text(),
                            "secondary_category": tfds.features.Text(),
                            "record_id": tfds.features.Text(),
                            "task_id": tfds.features.Text(),
                            "task_description": tfds.features.Text(),
                            "scene": tfds.features.Text(),
                            "camera": tfds.features.Text(),
                            "frame": tfds.features.Text(),
                            "rel_base_path": tfds.features.Text(),
                            "rel_transformed_path": tfds.features.Text(),
                            "rel_source_image_path": tfds.features.Text(),
                            "rel_depth_path": tfds.features.Text(),
                            "rel_normal_path": tfds.features.Text(),
                            "num_primary_instances": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "num_secondary_instances": tfds.features.Tensor(shape=(), dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("base_image", "transformed_image"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        splits = {}
        for summary_path in _summary_paths():
            summary = _load_json(summary_path)
            split_name = _split_name(summary)
            splits[split_name] = self._generate_examples(summary_path)
        return splits

    def _generate_examples(self, summary_path: Path):
        summary = _load_json(summary_path)
        dataset_name = summary["dataset_name"]
        scene_type = summary["scene_type"]
        split_name = _split_name(summary)
        samples_dir = Path(summary["full_pairs_dir"]).resolve()
        samples_manifest = Path(summary["full_pairs_manifest"]).resolve()
        subset_dir = Path(summary["raw_subset_dir"]).resolve()
        combo_root = Path(summary["full_manifest"]).resolve().parent
        combo_manifest = _load_json(Path(summary["full_manifest"]).resolve())
        records_by_id = {record["id"]: record for record in combo_manifest["records"]}

        with samples_manifest.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                record = records_by_id[row["record_id"]]
                roles = row["roles"]
                all_primary_instances = []
                all_secondary_instances = []
                for item in record["instances"]:
                    encoded = {
                        "instance_id": str(item["id"]),
                        "category": str(item["category"]),
                        "area": np.int32(item["area"]),
                        "bbox_xywh": np.asarray(item["bbox"], dtype=np.int32),
                        "rel_mask_path": relpath_posix(item["mask"], str(combo_root)),
                    }
                    if item["category"] == summary["primary_category"]:
                        all_primary_instances.append(encoded)
                    if item["category"] == summary.get("secondary_category"):
                        all_secondary_instances.append(encoded)
                key = f"{split_name}:{row['record_id']}:{row['task_id']}"
                yield key, {
                    "base_image": row["base"],
                    "transformed_image": row["transformed"],
                    "roles": {
                        "primary": {
                            "instance_id": roles["primary"]["id"],
                            "category": roles["primary"]["category"],
                            "bbox_xywh": np.asarray(roles["primary"]["bbox"], dtype=np.int32),
                        },
                        "secondary": {
                            "instance_id": roles["secondary"]["id"],
                            "category": roles["secondary"]["category"],
                            "bbox_xywh": np.asarray(roles["secondary"]["bbox"], dtype=np.int32),
                        },
                    },
                    "all_primary_instances": all_primary_instances,
                    "all_secondary_instances": all_secondary_instances,
                    "metadata": {
                        "dataset_name": str(dataset_name),
                        "split_name": str(split_name),
                        "scene_type": str(scene_type),
                        "primary_category": str(summary["primary_category"]),
                        "secondary_category": str(summary["secondary_category"]),
                        "record_id": str(row["record_id"]),
                        "task_id": str(row["task_id"]),
                        "task_description": str(row["description"]),
                        "scene": str(record["scene"]),
                        "camera": str(record["camera"]),
                        "frame": str(record["frame"]),
                        "rel_base_path": relpath_posix(row["base"], str(samples_dir)),
                        "rel_transformed_path": relpath_posix(row["transformed"], str(samples_dir)),
                        "rel_source_image_path": relpath_posix(record["image"], str(subset_dir)),
                        "rel_depth_path": relpath_posix(record["depth"], str(subset_dir)),
                        "rel_normal_path": relpath_posix(record["normal"], str(subset_dir)),
                        "num_primary_instances": np.int32(len(all_primary_instances)),
                        "num_secondary_instances": np.int32(len(all_secondary_instances)),
                    },
                }


def load_data(
    split: str = "bathroom_bathtub_sink",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("hypersim_combo_pairs", **load_kw)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _decode_text(x) -> str:
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def sanity_check(
    tfds_data_dir: str,
    split: str = "bathroom_bathtub_sink",
    num_batches: int = 1,
    batch_size: int = 1,
    max_show: int = 1,
):
    ds = tfds.load("hypersim_combo_pairs", split=split, data_dir=tfds_data_dir)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        base = ex["base_image"].numpy()
        transformed = ex["transformed_image"].numpy()
        record_ids = ex["metadata"]["record_id"]
        task_ids = ex["metadata"]["task_id"]
        b = min(max_show, base.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(base[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            title = f"{_decode_text(record_ids[j])} | {_decode_text(task_ids[j])}"
            plt.title(title[:100])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(np.clip(transformed[j], 0, 255).astype(np.uint8))
            plt.axis("off")
        plt.tight_layout()
        out = f"hypersim_combo_pairs_sanity_{split}_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")
