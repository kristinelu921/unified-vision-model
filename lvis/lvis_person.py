"""
TFDS builder: LVIS v1 examples that contain at least one ``person`` annotation.

Standalone module (does **not** import ``lvis.py``). Paths default to the same
layout as ``lvis.py`` so you can point at the same LVIS images and JSON:

- ``LVIS_IMAGES_ROOT``, ``LVIS_TRAIN_JSON``, ``LVIS_VAL_JSON``
- ``LVIS_MAX_EXAMPLES_PER_SPLIT``, ``LVIS_SAMPLE_SEED``

Metadata omits ``json_path`` (unlike ``Lvis``).

Only keeps annotations whose COCO-style bbox (``[x,y,w,h]``) has area ``w*h`` at
least ``MIN_BBOX_AREA_FRAC`` times the image area (default ``1/30``). Images with
no such annotations are dropped. Override with env ``LVIS_MIN_BBOX_AREA_FRAC``.

Build::

  tfds build lvis/lvis_person.py --data_dir=/path/to/tfcds_out

Load::

  tfds.load('lvis_person', data_dir=..., split='train')
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

_REPO_ROOT = "/mnt/klum/lvis"
DATA_ROOT = _REPO_ROOT
IMAGES_ROOT = os.environ.get("LVIS_IMAGES_ROOT", DATA_ROOT)
TRAIN_JSON = os.environ.get(
    "LVIS_TRAIN_JSON",
    os.path.join(DATA_ROOT, "annotations", "lvis_v1_train.json"),
)
VAL_JSON = os.environ.get(
    "LVIS_VAL_JSON",
    os.path.join(DATA_ROOT, "annotations", "lvis_v1_val.json"),
)

_MAX_EXAMPLES = os.environ.get("LVIS_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None
_SAMPLE_SEED = int(os.environ.get("LVIS_SAMPLE_SEED", "0"))

# Min bbox area (w×h in px²) as a fraction of image area (W×H); default 1/30.
_MIN_FRAC_RAW = os.environ.get("LVIS_MIN_BBOX_AREA_FRAC", "").strip()
MIN_BBOX_AREA_FRAC = float(_MIN_FRAC_RAW) if _MIN_FRAC_RAW else (1.0 / 30.0)


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _encode_ann(ann: dict) -> dict:
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox"]], dtype=np.float32),
        "area": np.float32(ann.get("area", 0.0)),
        "category_id": np.int64(ann["category_id"]),
        "iscrowd": np.int64(ann.get("iscrowd", 0)),
    }


def preprocess_func(example):
    image = example["image"]
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    example["image"] = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    return example


def _category_name_has_person(name: str) -> bool:
    #print(name)
    return name.lower() == "person"


def _bbox_area_xywh(ann: dict) -> float:
    _x, _y, bw, bh = (float(t) for t in ann["bbox"])
    return max(0.0, bw) * max(0.0, bh)


def _filter_anns_by_min_bbox_frac(
    anns: list, *, image_w: int, image_h: int, min_frac: float
) -> list:
    """Keep annotations whose bbox rectangle area is >= ``min_frac * W * H``."""
    img_area = float(max(0, image_w) * max(0, image_h))
    if img_area <= 0.0 or min_frac <= 0.0:
        return list(anns)
    need = min_frac * img_area
    return [a for a in anns if _bbox_area_xywh(a) >= need]


class LvisPerson(tfds.core.GeneratorBasedBuilder):    
    VERSION = tfds.core.Version("1.2.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "LVIS v1 subset: images with at least one person-related category "
                "in annotations; annotations list is filtered to those only."
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
            homepage="https://www.lvisdataset.org/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(TRAIN_JSON, "train"),
            "validation": self._generate_examples(VAL_JSON, "validation"),
        }

    def _generate_examples(self, json_path: str, split_name: str):
        with open(json_path, encoding="utf-8") as f:
            lvis = json.load(f)

        categories = sorted(lvis["categories"], key=lambda c: int(c["id"]))
        class_idx_by_id = {int(cat["id"]): i for i, cat in enumerate(categories)}
        name_by_id = {int(cat["id"]): cat["name"] for cat in categories}
        category_feats = [
            {
                "category_id": np.int64(int(cat["id"])),
                "class_idx": np.int64(class_idx_by_id[int(cat["id"])]),
                "category_name": cat["name"],
            }
            for cat in categories
        ]

        by_image: dict[int, list] = {}
        for ann in lvis["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)

        images = list(lvis["images"])
        random.Random(_SAMPLE_SEED).shuffle(images)

        n = 0
        for img in images:
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            image_id = int(img["id"])
            file_name = img["coco_url"].rsplit("/", 2)[-2] + "/" + img["coco_url"].rsplit("/", 1)[-1]
            h, w = int(img["height"]), int(img["width"])
            rgb_path = os.path.join(IMAGES_ROOT, file_name)

            raw_anns = by_image.get(image_id, [])
            person_anns = [
                a
                for a in raw_anns
                if _category_name_has_person(name_by_id[int(a["category_id"])])
            ]
            person_anns = _filter_anns_by_min_bbox_frac(
                person_anns,
                image_w=w,
                image_h=h,
                min_frac=MIN_BBOX_AREA_FRAC,
            )
            if not person_anns:
                continue

            key = f"{split_name}_{image_id}_{file_name}"
            n += 1
            yield key, {
                "image": rgb_path,
                "annotations": [
                    {
                        **_encode_ann(a),
                        "class_idx": np.int64(class_idx_by_id[int(a["category_id"])]),
                        "category_name": name_by_id[int(a["category_id"])],
                    }
                    for a in person_anns
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
    ds = tfds.load("lvis_person", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    pass
