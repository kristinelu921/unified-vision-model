"""Combined strict 256x256 grouped-9 TFDS for UrbanSyn, Hypersim, and Replica-GSO."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

try:
    from .gso_replica_combo_grouped9 import _sanitize_split_name
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from evals.tfds.gso_replica_combo_grouped9 import _sanitize_split_name

try:
    from evals.tfds import gso_replica_tfds, hypersim_combo_grouped9, urban_syn_tfds  # noqa: F401
except ImportError:  # pragma: no cover
    pass


URBAN_SYN_NAME = "urban_syn_tfds"
GSO_NAME = "gso_replica_tfds"
HYPERSIM_NAME = "hypersim_combo_grouped9"

URBAN_SYN_DIR = Path(os.environ.get("COMBINED_URBAN_SYN_DIR", "/mnt/klum/lvm/diverse_dataset/urban_syn_tfds"))
GSO_DIR = Path(os.environ.get("COMBINED_GSO_DIR", "/mnt/klum/lvm/gso_replica_tfds"))
HYPERSIM_DIR = Path(os.environ.get("COMBINED_HYPERSIM_DIR", "/mnt/klum/lvm/hypersim_combo"))

TARGET_SIZE = 256
GROUP_SIZE = 9
MIN_BOX_AREA = TARGET_SIZE * TARGET_SIZE / 50.0


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _source_builder(name: str, data_dir: Path):
    return tfds.builder(name, data_dir=str(data_dir))


def _source_splits(name: str, data_dir: Path) -> list[str]:
    builder = _source_builder(name, data_dir)
    return sorted(builder.info.splits.keys())


def _prefixed_split(source: str, split: str) -> str:
    return _sanitize_split_name(f"{source}__{split}")


def _tfds_numpy(name: str, data_dir: Path, split: str):
    return tfds.as_numpy(tfds.load(name, split=split, data_dir=str(data_dir)))


def _metadata_dict(sequence: dict, index: int) -> dict:
    return {key: _decode(value[index]) for key, value in sequence.items()}


def _group_metadata_dict(metadata: dict) -> dict:
    return {key: _decode(value) for key, value in metadata.items()}


def _copy_group(example: dict, source_name: str) -> dict:
    group_metadata = _group_metadata_dict(example["group_metadata"])
    group_metadata["source_dataset"] = source_name
    scene_count = int(group_metadata.get("scene_count", GROUP_SIZE))
    return {
        "base_images": example["base_images"],
        "transformed_images": example["transformed_images"],
        "primary_masks": example["primary_masks"],
        "secondary_masks": example["secondary_masks"],
        "example_metadata": [
            _metadata_dict(example["example_metadata"], index) for index in range(len(example["base_images"]))
        ],
        "group_metadata": {
            "dataset_name": "combined_diverse_tfds",
            "source_dataset": source_name,
            "duo_split_name": group_metadata.get("duo_split_name", ""),
            "task_split_name": group_metadata.get("task_split_name", ""),
            "scene_type": group_metadata.get("scene_type", source_name),
            "primary_category": group_metadata.get("primary_category", ""),
            "secondary_category": group_metadata.get("secondary_category", ""),
            "task_id": group_metadata.get("task_id", ""),
            "task_description": group_metadata.get("task_description", ""),
            "group_id": group_metadata.get("group_id", ""),
            "group_size": np.int32(len(example["base_images"])),
            "scene_count": np.int32(scene_count),
        },
    }


def _bbox_union(boxes: list[list[int]]) -> tuple[int, int, int, int]:
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return x1, y1, x2, y2


def _crop_square(union_xyxy: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    size = min(width, height)
    ux1, uy1, ux2, uy2 = union_xyxy
    cx = (ux1 + ux2) / 2.0
    cy = (uy1 + uy2) / 2.0
    x1 = int(round(cx - size / 2.0))
    y1 = int(round(cy - size / 2.0))
    x1 = max(0, min(x1, width - size))
    y1 = max(0, min(y1, height - size))
    return x1, y1, x1 + size, y1 + size


def _clip_bbox_to_crop(bbox: list[int], crop: tuple[int, int, int, int], scale: float) -> tuple[list[int] | None, float, int]:
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + w, y + h
    cx1, cy1, cx2, cy2 = crop
    ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
    clipped_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    original_area = max(0, w) * max(0, h)
    retained = clipped_area / float(original_area) if original_area else 0.0
    if clipped_area <= 0:
        return None, retained, 0
    out = [
        int(round((ix1 - cx1) * scale)),
        int(round((iy1 - cy1) * scale)),
        int(round((ix2 - ix1) * scale)),
        int(round((iy2 - iy1) * scale)),
    ]
    return out, retained, out[2] * out[3]


def _resize_image(array: np.ndarray, crop: tuple[int, int, int, int], resample: int) -> np.ndarray:
    image = Image.fromarray(array)
    image = image.crop(crop).resize((TARGET_SIZE, TARGET_SIZE), resample=resample)
    out = np.asarray(image)
    if out.ndim == 2:
        out = out[:, :, None]
    return out


def _strict_hypersim_item(example: dict, index: int) -> dict | None:
    metadata = _metadata_dict(example["example_metadata"], index)
    annotation = json.loads(metadata["annotation_json"])
    primary_instances = annotation.get("primary_instances", [])
    secondary_instances = annotation.get("secondary_instances", [])
    if not primary_instances or not secondary_instances:
        return None
    primary = primary_instances[0]
    secondary = secondary_instances[0]
    image = example["base_images"][index]
    height, width = image.shape[:2]
    crop = _crop_square(_bbox_union([primary["bbox"], secondary["bbox"]]), width, height)
    scale = TARGET_SIZE / float(crop[2] - crop[0])
    primary_bbox, primary_retained, primary_area = _clip_bbox_to_crop(primary["bbox"], crop, scale)
    secondary_bbox, secondary_retained, secondary_area = _clip_bbox_to_crop(secondary["bbox"], crop, scale)
    if primary_bbox is None or secondary_bbox is None:
        return None
    if primary_retained < 0.5 or secondary_retained < 0.5:
        return None
    if primary_area < MIN_BOX_AREA or secondary_area < MIN_BOX_AREA:
        return None

    primary_out = dict(primary)
    secondary_out = dict(secondary)
    primary_out["bbox_before_crop"] = primary["bbox"]
    secondary_out["bbox_before_crop"] = secondary["bbox"]
    primary_out["bbox"] = primary_bbox
    secondary_out["bbox"] = secondary_bbox
    primary_out["retained_bbox_fraction"] = primary_retained
    secondary_out["retained_bbox_fraction"] = secondary_retained

    role_groups = annotation.get("role_groups", {})
    return {
        "base_image": _resize_image(example["base_images"][index], crop, Image.Resampling.BILINEAR),
        "transformed_image": _resize_image(example["transformed_images"][index], crop, Image.Resampling.BILINEAR),
        "primary_mask": _resize_image(example["primary_masks"][index].squeeze(-1), crop, Image.Resampling.NEAREST),
        "secondary_mask": _resize_image(example["secondary_masks"][index].squeeze(-1), crop, Image.Resampling.NEAREST),
        "metadata": {
            **metadata,
            "annotation_json": json.dumps(
                {
                    "roles": annotation.get("roles", {}),
                    "role_groups": role_groups,
                    "primary_instances": [primary_out],
                    "secondary_instances": [secondary_out],
                    "crop_xyxy": crop,
                },
                separators=(",", ":"),
            ),
        },
        "scene": metadata["scene"],
    }


def _hypersim_groups(split: str):
    split_metadata: dict | None = None
    buckets: dict[str, list[dict]] = {}
    for example in _tfds_numpy(HYPERSIM_NAME, HYPERSIM_DIR, split):
        split_metadata = _group_metadata_dict(example["group_metadata"])
        for index in range(len(example["base_images"])):
            item = _strict_hypersim_item(example, index)
            if item is None:
                continue
            buckets.setdefault(item["scene"], []).append(item)
    if split_metadata is None:
        return
    while len(buckets) >= GROUP_SIZE:
        scenes = sorted(buckets, key=lambda scene: (-len(buckets[scene]), scene))[:GROUP_SIZE]
        group = [buckets[scene].pop(0) for scene in scenes]
        buckets = {scene: items for scene, items in buckets.items() if items}
        yield split_metadata, group


class CombinedDiverseTfds(tfds.core.GeneratorBasedBuilder):
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
                "Combined strict 256x256 grouped-9 combo dataset from UrbanSyn, Hypersim, "
                "and Omnidata Replica-GSO."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "base_images": tfds.features.Sequence(tfds.features.Image(shape=(TARGET_SIZE, TARGET_SIZE, 3))),
                    "transformed_images": tfds.features.Sequence(
                        tfds.features.Image(shape=(TARGET_SIZE, TARGET_SIZE, 3))
                    ),
                    "primary_masks": tfds.features.Sequence(tfds.features.Image(shape=(TARGET_SIZE, TARGET_SIZE, 1))),
                    "secondary_masks": tfds.features.Sequence(tfds.features.Image(shape=(TARGET_SIZE, TARGET_SIZE, 1))),
                    "example_metadata": tfds.features.Sequence(example_metadata),
                    "group_metadata": tfds.features.FeaturesDict(
                        {
                            "dataset_name": tfds.features.Text(),
                            "source_dataset": tfds.features.Text(),
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
        splits = {}
        for split in _source_splits(URBAN_SYN_NAME, URBAN_SYN_DIR):
            splits[_prefixed_split("urban_syn", split)] = self._generate_existing(URBAN_SYN_NAME, URBAN_SYN_DIR, split)
        for split in _source_splits(GSO_NAME, GSO_DIR):
            splits[_prefixed_split("gso_replica", split)] = self._generate_existing(GSO_NAME, GSO_DIR, split)
        for split in _source_splits(HYPERSIM_NAME, HYPERSIM_DIR):
            splits[_prefixed_split("hypersim", split)] = self._generate_hypersim(split)
        return splits

    def _generate_existing(self, source_name: str, data_dir: Path, split: str):
        for index, example in enumerate(_tfds_numpy(source_name, data_dir, split)):
            yield f"{source_name}:{split}:{index:05d}", _copy_group(example, source_name)

    def _generate_examples(self):
        raise NotImplementedError("combined_diverse_tfds uses source-specific split generators")

    def _generate_hypersim(self, split: str):
        for group_index, (metadata, group) in enumerate(_hypersim_groups(split)):
            base_images = [item["base_image"] for item in group]
            transformed_images = [item["transformed_image"] for item in group]
            primary_masks = [item["primary_mask"] for item in group]
            secondary_masks = [item["secondary_mask"] for item in group]
            task_split = metadata.get("task_split_name", split)
            yield (
                f"hypersim:{split}:{group_index:05d}",
                {
                    "base_images": base_images,
                    "transformed_images": transformed_images,
                    "primary_masks": primary_masks,
                    "secondary_masks": secondary_masks,
                    "example_metadata": [item["metadata"] for item in group],
                    "group_metadata": {
                        "dataset_name": "combined_diverse_tfds",
                        "source_dataset": HYPERSIM_NAME,
                        "duo_split_name": metadata.get("duo_split_name", ""),
                        "task_split_name": task_split,
                        "scene_type": metadata.get("scene_type", "hypersim"),
                        "primary_category": metadata.get("primary_category", ""),
                        "secondary_category": metadata.get("secondary_category", ""),
                        "task_id": metadata.get("task_id", ""),
                        "task_description": metadata.get("task_description", ""),
                        "group_id": f"hypersim:{split}:{group_index:05d}",
                        "group_size": np.int32(GROUP_SIZE),
                        "scene_count": np.int32(GROUP_SIZE),
                    },
                },
            )


def load_data(split: str, data_dir: str | None = None):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    return tfds.load("combined_diverse_tfds", **load_kw)
