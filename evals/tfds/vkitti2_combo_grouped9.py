"""TFDS builder for grouped-9 VKITTI2 24-task combo transformations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

try:
    from .eval_common import relpath_posix
    from .hypersim_combo_pairs import _sanitize_split_name
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.tfds.eval_common import relpath_posix
    from evals.tfds.hypersim_combo_pairs import _sanitize_split_name


_DEFAULT_ROOT = Path("/mnt/klum/vkitti2_diverse/combo_24task/selected_pair_sets_24")


def _root() -> Path:
    return Path(os.environ.get("VKITTI2_GROUPED9_ROOT", str(_DEFAULT_ROOT))).expanduser().resolve()


def _pair_dirs() -> list[Path]:
    root = _root()
    raw = os.environ.get("VKITTI2_GROUPED9_PAIRS", "").strip()
    if raw:
        return [root / part.strip() for part in raw.split(",") if part.strip()]
    return sorted(path for path in root.iterdir() if (path / "grouped9_manifest.json").is_file())


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _task_ids(grouped_manifest: dict) -> list[str]:
    seen = {}
    for group in grouped_manifest["groups"]:
        for task_id in group["tasks"]:
            seen.setdefault(task_id, None)
    return list(seen)


class Vkitti2ComboGrouped9(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        example_metadata = tfds.features.FeaturesDict(
            {
                "record_id": tfds.features.Text(),
                "scene": tfds.features.Text(),
                "variation": tfds.features.Text(),
                "camera": tfds.features.Text(),
                "frame": tfds.features.Text(),
                "scene_context": tfds.features.Text(),
                "rel_base_path": tfds.features.Text(),
                "rel_transformed_path": tfds.features.Text(),
                "annotation_json": tfds.features.Text(),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "VKITTI2 combo examples grouped into samples of 9 images. "
                "Each split is one category pair plus one of the 24 transformations."
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
                            "official_scene_count": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "scene_context_count": tfds.features.Tensor(shape=(), dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("base_images", "transformed_images"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        splits = {}
        for pair_dir in _pair_dirs():
            grouped = _load_json(pair_dir / "grouped9_manifest.json")
            for task_id in _task_ids(grouped):
                split = f"{_sanitize_split_name(pair_dir.name)}__task_{_sanitize_split_name(task_id)}"
                splits[split] = self._generate_examples(pair_dir, task_id)
        return splits

    def _generate_examples(self, pair_dir: Path, split_task_id: str):
        combo_manifest = _load_json(pair_dir / "combo_manifest.json")
        grouped = _load_json(pair_dir / "grouped9_manifest.json")
        records_by_id = {record["id"]: record for record in combo_manifest["records"]}
        pair_name = _sanitize_split_name(pair_dir.name)
        for group in grouped["groups"]:
            rows = group["tasks"].get(split_task_id)
            if not rows:
                continue
            yield (
                f"{pair_name}:{split_task_id}:{group['group_id']}",
                self._encode_group(pair_dir, combo_manifest, records_by_id, group, rows, split_task_id),
            )

    def _encode_group(
        self,
        pair_dir: Path,
        combo_manifest: dict,
        records_by_id: dict[str, dict],
        group: dict,
        rows: list[dict],
        task_id: str,
    ) -> dict:
        samples_dir = pair_dir / "samples_24"
        base_images = []
        transformed_images = []
        primary_masks = []
        secondary_masks = []
        metadata = []
        primary = str(combo_manifest["primary_category"])
        secondary = str(combo_manifest["secondary_category"])

        for row in rows:
            record = records_by_id[row["record_id"]]
            primary_instances = [item for item in record["instances"] if item["category"] == primary]
            secondary_instances = [item for item in record["instances"] if item["category"] == secondary]
            primary_masks.append(primary_instances[0]["mask"])
            secondary_masks.append(secondary_instances[0]["mask"])
            base_images.append(row["base"])
            transformed_images.append(row["transformed"])
            metadata.append(
                {
                    "record_id": str(row["record_id"]),
                    "scene": str(record["scene"]),
                    "variation": str(record["variation"]),
                    "camera": str(record["camera"]),
                    "frame": str(record["frame"]),
                    "scene_context": f"{record['scene']}/{record['variation']}/{record['camera']}",
                    "rel_base_path": relpath_posix(row["base"], str(samples_dir)),
                    "rel_transformed_path": relpath_posix(row["transformed"], str(samples_dir)),
                    "annotation_json": json.dumps(
                        {
                            "roles": row["roles"],
                            "role_groups": row["role_groups"],
                            "primary_instances": primary_instances,
                            "secondary_instances": secondary_instances,
                        },
                        separators=(",", ":"),
                    ),
                }
            )

        task_description = rows[0].get("description", "")
        duo_split = _sanitize_split_name(pair_dir.name)
        return {
            "base_images": base_images,
            "transformed_images": transformed_images,
            "primary_masks": primary_masks,
            "secondary_masks": secondary_masks,
            "example_metadata": metadata,
            "group_metadata": {
                "dataset_name": "vkitti2_combo_grouped9",
                "duo_split_name": duo_split,
                "task_split_name": f"{duo_split}__task_{_sanitize_split_name(task_id)}",
                "scene_type": "vkitti2",
                "primary_category": primary,
                "secondary_category": secondary,
                "task_id": str(task_id),
                "task_description": str(task_description),
                "group_id": str(group["group_id"]),
                "group_size": np.int32(len(rows)),
                "official_scene_count": np.int32(len(group["unique_official_scenes"])),
                "scene_context_count": np.int32(len(group["unique_scene_contexts"])),
            },
        }


def load_data(split: str, data_dir: str | None = None):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    return tfds.load("vkitti2_combo_grouped9", **load_kw)
