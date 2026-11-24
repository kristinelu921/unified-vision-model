"""TFDS builder for strict 256x256 grouped-9 UrbanSyn combo transformations."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

try:
    from .eval_common import relpath_posix
    from .gso_replica_combo_grouped9 import _sanitize_split_name
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.tfds.eval_common import relpath_posix
    from evals.tfds.gso_replica_combo_grouped9 import _sanitize_split_name


_DEFAULT_INDEX_ROOT = Path("/mnt/klum/lvm/diverse_dataset/urban_syn_tfds_index")


def _index_root() -> Path:
    return Path(os.environ.get("URBAN_SYN_TFDS_INDEX", str(_DEFAULT_INDEX_ROOT))).expanduser().resolve()


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class UrbanSynTfds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
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
                "Strict UrbanSyn combo examples grouped into samples of 9 scene-diverse "
                "256x256 images. Each split is one category pair plus one transformation task."
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
                            "scene_count": tfds.features.Tensor(shape=(), dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("base_images", "transformed_images"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        summary = _load_json(_index_root() / "summary.json")
        splits = {}
        for item in summary["splits"]:
            if int(item["groups"]) <= 0:
                continue
            split_name = _sanitize_split_name(item["split_name"])
            splits[split_name] = self._generate_examples(Path(item["path"]))
        return splits

    def _generate_examples(self, split_path: Path):
        for group_index, group in enumerate(_read_jsonl(split_path)):
            yield group["group_id"], self._encode_group(group, group_index)

    def _encode_group(self, group: dict, group_index: int) -> dict:
        base_images = []
        transformed_images = []
        primary_masks = []
        secondary_masks = []
        metadata = []
        for item in group["items"]:
            sample_root = Path(item["base"]).resolve().parent.parent
            base_images.append(item["base"])
            transformed_images.append(item["transformed"])
            primary_masks.append(item["primary_mask"])
            secondary_masks.append(item["secondary_mask"])
            metadata.append(
                {
                    "record_id": str(item["record_id"]),
                    "scene": str(item["scene"]),
                    "camera": str(item["camera"]),
                    "frame": str(item["frame"]),
                    "rel_base_path": relpath_posix(item["base"], str(sample_root)),
                    "rel_transformed_path": relpath_posix(item["transformed"], str(sample_root)),
                    "annotation_json": json.dumps(
                        {
                            "roles": item["roles"],
                            "role_groups": item["role_groups"],
                            "primary_instances": [item["primary_instance"]],
                            "secondary_instances": [item["secondary_instance"]],
                            "primary_masks": [item["primary_mask"]],
                            "secondary_masks": [item["secondary_mask"]],
                        },
                        separators=(",", ":"),
                    ),
                }
            )

        split_name = str(group["split_name"])
        duo_split = split_name.split("__task_", 1)[0]
        return {
            "base_images": base_images,
            "transformed_images": transformed_images,
            "primary_masks": primary_masks,
            "secondary_masks": secondary_masks,
            "example_metadata": metadata,
            "group_metadata": {
                "dataset_name": "urban_syn_tfds",
                "duo_split_name": duo_split,
                "task_split_name": split_name,
                "scene_type": "urbansyn",
                "primary_category": str(group["duo"][0]),
                "secondary_category": str(group["duo"][1]),
                "task_id": str(group["task_id"]),
                "task_description": str(group.get("task_description", "")),
                "group_id": str(group.get("group_id", f"{split_name}:{group_index:05d}")),
                "group_size": np.int32(group["group_size"]),
                "scene_count": np.int32(group["scene_count"]),
            },
        }


def load_data(split: str, data_dir: str | None = None):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    return tfds.load("urban_syn_tfds", **load_kw)
