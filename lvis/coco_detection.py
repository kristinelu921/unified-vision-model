"""
TFDS builder: COCO 2017 cat detection (bounding boxes).

Requires local data prepared by ``lvis/download_coco_detection_cat.py``.

Expected layout:
- ``/mnt/klum/lvm/coco_detection/images/{train2017,val2017}/*.jpg``
- ``/mnt/klum/lvm/coco_detection/annotations.jsonl``

Build::

  tfds build lvis/coco_detection.py --data_dir=/path/to/tfds_out

Load::

  tfds.load("coco_detection", data_dir=..., split="train")
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_REPO_ROOT = "/mnt/klum/lvm/coco_detection"
DATA_ROOT = os.environ.get("COCO_DETECTION_OUT_DIR", _REPO_ROOT)
IMAGES_ROOT = os.path.join(DATA_ROOT, "images")
ANNOTATIONS_JSONL = os.path.join(DATA_ROOT, "annotations.jsonl")

_MAX_EXAMPLES = os.environ.get("COCO_DETECTION_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None
_SAMPLE_SEED = int(os.environ.get("COCO_DETECTION_SAMPLE_SEED", "42"))

_MIN_FRAC_RAW = os.environ.get("COCO_DETECTION_MIN_BBOX_AREA_FRAC", "").strip()
MIN_BBOX_AREA_FRAC = float(_MIN_FRAC_RAW) if _MIN_FRAC_RAW else (1.0 / 30.0)


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _bbox_area_xywh(ann: dict) -> float:
    _x, _y, bw, bh = [float(v) for v in ann["bbox_xywh"]]
    return max(0.0, bw) * max(0.0, bh)


def _encode_ann(ann: dict) -> dict:
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox_xywh"]], dtype=np.float32),
        "area": np.float32(float(ann.get("area", 0.0))),
        "category_id": np.int64(int(ann.get("category_id", 0))),
        "iscrowd": np.int64(int(ann.get("iscrowd", 0))),
    }


def preprocess_func(example):
    image = example["image"]
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    example["image"] = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    return example


def _load_records() -> list[dict]:
    if not os.path.isfile(ANNOTATIONS_JSONL):
        raise FileNotFoundError(
            f"Missing annotations file: {ANNOTATIONS_JSONL}. "
            "Run lvis/download_coco_detection_cat.py first."
        )
    rows: list[dict] = []
    with open(ANNOTATIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CocoDetection(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "COCO 2017 subset: images containing at least one cat annotation. "
                "All cat bounding boxes for an image are grouped in `annotations`."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "annotations": tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                "bbox": tfds.features.Tensor(shape=(4,), dtype=np.float32),
                                "area": tfds.features.Tensor(shape=(), dtype=np.float32),
                                "category_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                                "class_idx": tfds.features.Tensor(shape=(), dtype=np.int64),
                                "category_name": tfds.features.Text(),
                                "iscrowd": tfds.features.Tensor(shape=(), dtype=np.int64),
                            }
                        ),
                    ),
                    "categories": tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                "category_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                                "class_idx": tfds.features.Tensor(shape=(), dtype=np.int64),
                                "category_name": tfds.features.Text(),
                            }
                        ),
                    ),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "image_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "file_name": tfds.features.Text(),
                            "width": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "height": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "rel_image_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            homepage="https://cocodataset.org/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples("train"),
            "validation": self._generate_examples("validation"),
        }

    def _generate_examples(self, split_name: str):
        rows = [r for r in _load_records() if r.get("split") == split_name]
        random.Random(_SAMPLE_SEED).shuffle(rows)

        category_name = "cat"
        category_feats = [
            {
                "category_id": np.int64(0),
                "class_idx": np.int64(0),
                "category_name": category_name,
            }
        ]

        n = 0
        for rec in rows:
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            file_name = rec["file_name"]
            image_id = int(rec.get("image_id", n))
            w = int(rec["width"])
            h = int(rec["height"])
            rgb_path = os.path.join(IMAGES_ROOT, rec["image_rel_path"])
            if not os.path.isfile(rgb_path):
                continue

            anns = list(rec.get("annotations", []))
            if MIN_BBOX_AREA_FRAC > 0:
                img_area = float(max(w, 0) * max(h, 0))
                if img_area > 0:
                    need = MIN_BBOX_AREA_FRAC * img_area
                    anns = [a for a in anns if _bbox_area_xywh(a) >= need]
            if not anns:
                continue

            key = f"{split_name}_{image_id}_{file_name}"
            n += 1
            yield key, {
                "image": rgb_path,
                "annotations": [
                    {
                        **_encode_ann(a),
                        "class_idx": np.int64(0),
                        "category_name": category_name,
                    }
                    for a in anns
                ],
                "categories": category_feats,
                "metadata": {
                    "image_id": np.int64(image_id),
                    "file_name": file_name,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": _relpath(rgb_path, IMAGES_ROOT),
                },
            }


def load_data(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
):
    kw: dict = {"split": split}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load("coco_detection", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    pass
