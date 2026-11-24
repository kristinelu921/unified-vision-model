"""TFDS builder for 9-image grouped Hypersim combo transformations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

try:
    from .eval_common import relpath_posix
    from .hypersim_combo_pairs import _load_json, _sanitize_split_name, _split_name
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.tfds.eval_common import relpath_posix
    from evals.tfds.hypersim_combo_pairs import _load_json, _sanitize_split_name, _split_name


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DUO_SUMMARIES = [
    _REPO_ROOT / "evals" / "combo" / "hypersim_scaled_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_livingroom_sofa_table_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_bedroom_bed_nightstand_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_kitchen_refrigerator_cabinet_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_dining_chair_table_summary.json",
    _REPO_ROOT / "evals" / "combo" / "hypersim_restaurant_chair_table_summary.json",
]


def _summary_paths() -> list[Path]:
    raw = os.environ.get("HYPERSIM_GROUPED9_SUMMARIES", "").strip()
    if raw:
        return [Path(part).expanduser().resolve() for part in raw.split(",") if part.strip()]
    return [path for path in _DEFAULT_DUO_SUMMARIES if path.is_file()]


def _include_partial_groups() -> bool:
    return os.environ.get("HYPERSIM_GROUPED9_INCLUDE_PARTIAL", "").strip().lower() in {"1", "true", "yes"}


def _task_id_from_split(split_name: str) -> str:
    marker = "__task_"
    if marker not in split_name:
        raise ValueError(f"Grouped split must contain {marker!r}: {split_name}")
    return split_name.split(marker, 1)[1]


class HypersimComboGrouped9(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.4")

    def _info(self) -> tfds.core.DatasetInfo:
        role_feats = tfds.features.FeaturesDict(
            {
                "instance_id": tfds.features.Text(),
                "category": tfds.features.Text(),
                "bbox_xywh": tfds.features.Tensor(shape=(4,), dtype=np.int32),
            }
        )
        example_metadata = tfds.features.FeaturesDict(
            {
                "record_id": tfds.features.Text(),
                "scene": tfds.features.Text(),
                "camera": tfds.features.Text(),
                "frame": tfds.features.Text(),
                "rel_base_path": tfds.features.Text(),
                "rel_transformed_path": tfds.features.Text(),
                "annotation_json": tfds.features.Text(),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Hypersim combo examples grouped into samples of 9 images. "
                "Each TFDS split is one duo category plus one transformation task, "
                "with split names formatted as '<duo_split>__task_<task_id>'."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "base_images": tfds.features.Sequence(tfds.features.Image(shape=(None, None, 3))),
                    "transformed_images": tfds.features.Sequence(tfds.features.Image(shape=(None, None, 3))),
                    "primary_masks": tfds.features.Sequence(tfds.features.Image(shape=(None, None, 1))),
                    "secondary_masks": tfds.features.Sequence(tfds.features.Image(shape=(None, None, 1))),
                    "example_metadata": tfds.features.Sequence(example_metadata),
                    "group_metadata": tfds.features.FeaturesDict(
                        {
                            "dataset_name": tfds.features.Text(),
                            "duo_split_name": tfds.features.Text(),
                            "task_split_name": tfds.features.Text(),
                            "scene_type": tfds.features.Text(),
                            "primary_category": tfds.features.Text(),
                            "secondary_category": tfds.features.Text(),
                            "task_id": tfds.features.Text(),
                            "task_description": tfds.features.Text(),
                            "group_id": tfds.features.Text(),
                            "group_size": tfds.features.Tensor(shape=(), dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("base_images", "transformed_images"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        splits = {}
        for summary_path in _summary_paths():
            summary = _load_json(summary_path)
            duo_split = _split_name(summary)
            tasks = _task_ids(Path(summary["full_pairs_manifest"]))
            for task_id in tasks:
                count = _task_group_count(summary_path, task_id)
                if count == 0 and not _include_partial_groups():
                    continue
                split = f"{duo_split}__task_{_sanitize_split_name(task_id)}"
                splits[split] = self._generate_examples(summary_path, task_id)
        return splits

    def _generate_examples(self, summary_path: Path, split_task_id: str):
        summary = _load_json(summary_path)
        duo_split = _split_name(summary)
        samples_dir = Path(summary["full_pairs_dir"]).resolve()
        combo_root = Path(summary["full_manifest"]).resolve().parent
        combo_manifest = _load_json(Path(summary["full_manifest"]).resolve())
        records_by_id = {record["id"]: record for record in combo_manifest["records"]}
        rows = []
        with Path(summary["full_pairs_manifest"]).resolve().open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["task_id"] == split_task_id:
                    rows.append(row)
        groups = _group_by_unique_scenes(rows, records_by_id)

        for group_index, group in enumerate(groups):
            first = group[0]
            yield f"{duo_split}:{split_task_id}:{group_index:05d}", self._encode_group(
                summary=summary,
                duo_split=duo_split,
                samples_dir=samples_dir,
                combo_root=combo_root,
                records_by_id=records_by_id,
                group=group,
                group_index=group_index,
                task_description=first["description"],
            )

    def _encode_group(
        self,
        *,
        summary: dict,
        duo_split: str,
        samples_dir: Path,
        combo_root: Path,
        records_by_id: dict[str, dict],
        group: list[dict],
        group_index: int,
        task_description: str,
    ) -> dict:
        base_images = []
        transformed_images = []
        primary_masks = []
        secondary_masks = []
        metadata = []
        for row in group:
            record = records_by_id[row["record_id"]]
            primary_instances = [
                item for item in record["instances"] if item["category"] == summary["primary_category"]
            ]
            secondary_instances = [
                item for item in record["instances"] if item["category"] == summary["secondary_category"]
            ]
            primary_masks.append(primary_instances[0]["mask"])
            secondary_masks.append(secondary_instances[0]["mask"])
            base_images.append(row["base"])
            transformed_images.append(row["transformed"])
            metadata.append(
                {
                    "record_id": str(row["record_id"]),
                    "scene": str(record["scene"]),
                    "camera": str(record["camera"]),
                    "frame": str(record["frame"]),
                    "rel_base_path": relpath_posix(row["base"], str(samples_dir)),
                    "rel_transformed_path": relpath_posix(row["transformed"], str(samples_dir)),
                    "annotation_json": json.dumps(
                        {
                            "roles": row["roles"],
                            "role_groups": row["role_groups"],
                            "primary_instances": [
                                {
                                    "id": str(item["id"]),
                                    "category": str(item["category"]),
                                    "bbox": item["bbox"],
                                    "area": item["area"],
                                    "rel_mask_path": relpath_posix(item["mask"], str(combo_root)),
                                }
                                for item in primary_instances
                            ],
                            "secondary_instances": [
                                {
                                    "id": str(item["id"]),
                                    "category": str(item["category"]),
                                    "bbox": item["bbox"],
                                    "area": item["area"],
                                    "rel_mask_path": relpath_posix(item["mask"], str(combo_root)),
                                }
                                for item in secondary_instances
                            ],
                        },
                        separators=(",", ":"),
                    ),
                }
            )

        return {
            "base_images": base_images,
            "transformed_images": transformed_images,
            "primary_masks": primary_masks,
            "secondary_masks": secondary_masks,
            "example_metadata": metadata,
            "group_metadata": {
                "dataset_name": str(summary["dataset_name"]),
                "duo_split_name": str(duo_split),
                "task_split_name": f"{duo_split}__task_{_sanitize_split_name(group[0]['task_id'])}",
                "scene_type": str(summary["scene_type"]),
                "primary_category": str(summary["primary_category"]),
                "secondary_category": str(summary["secondary_category"]),
                "task_id": str(group[0]["task_id"]),
                "task_description": str(task_description),
                "group_id": f"{duo_split}:{group[0]['task_id']}:{group_index:05d}",
                "group_size": np.int32(len(group)),
            },
        }


def _task_ids(samples_manifest: Path) -> list[str]:
    seen = {}
    with samples_manifest.resolve().open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            seen.setdefault(row["task_id"], None)
    return list(seen)


def _task_count(samples_manifest: Path, task_id: str) -> int:
    count = 0
    with samples_manifest.resolve().open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task_id"] == task_id:
                count += 1
    return count


def _task_group_count(summary_path: Path, task_id: str) -> int:
    summary = _load_json(summary_path)
    combo_manifest = _load_json(Path(summary["full_manifest"]).resolve())
    records_by_id = {record["id"]: record for record in combo_manifest["records"]}
    rows = []
    with Path(summary["full_pairs_manifest"]).resolve().open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["task_id"] == task_id:
                rows.append(row)
    return len(_group_by_unique_scenes(rows, records_by_id))


def _group_by_unique_scenes(
    rows: list[dict],
    records_by_id: dict[str, dict],
    group_size: int = 9,
) -> list[list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        record = records_by_id[row["record_id"]]
        scene = str(record["scene"])
        buckets.setdefault(scene, []).append(row)

    for scene, bucket in buckets.items():
        bucket.sort(
            key=lambda row: (
                str(records_by_id[row["record_id"]]["camera"]),
                int(records_by_id[row["record_id"]]["frame"]),
                row["record_id"],
            )
        )

    groups = []
    include_partial = _include_partial_groups()
    while buckets:
        scenes = sorted(buckets, key=lambda scene: (-len(buckets[scene]), scene))
        if len(scenes) < group_size and not include_partial:
            break
        chosen = scenes[:group_size]
        group = [buckets[scene].pop(0) for scene in chosen]
        groups.append(group)
        buckets = {scene: bucket for scene, bucket in buckets.items() if bucket}
    return groups


def _interleave_by_scene_camera(rows: list[dict], records_by_id: dict[str, dict]) -> list[dict]:
    buckets = {}
    for row in rows:
        record = records_by_id[row["record_id"]]
        key = (str(record["scene"]), str(record["camera"]))
        buckets.setdefault(key, []).append(row)

    for key, bucket in buckets.items():
        bucket.sort(key=lambda row: (int(records_by_id[row["record_id"]]["frame"]), row["record_id"]))

    ordered = []
    keys = sorted(buckets, key=lambda key: (-len(buckets[key]), key))
    while keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def load_data(split: str, data_dir: str | None = None):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    return tfds.load("hypersim_combo_grouped9", **load_kw)
