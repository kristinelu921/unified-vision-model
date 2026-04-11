"""
TFDS builder for Open Images V7 — **cat** object detection (bounding boxes).

One TFDS example per image (all cat bboxes grouped).
Schema matches ``lvis_person.py``: ``image``, ``annotations`` (Sequence),
``categories``, ``metadata``.

Reads images from FiftyOne download dir and bbox annotations from the official
``oidv6-train-annotations-bbox.csv``.

Build::

    tfds build lvis/open_images_cat.py --data_dir=/mnt/klum/lvis/lvis_cat_detection_tfds

Load::

    tfds.load("open_images_cat", data_dir=..., split="train")
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image as PILImage

_OI_ROOT = "/mnt/klum/lvis/instance_segmentation/open-images-v7/train"
IMAGES_DIR = os.path.join(_OI_ROOT, "data")
BBOX_CSV = os.path.join(_OI_ROOT, "labels", "oidv6-train-annotations-bbox.csv")
CLASSES_CSV = os.path.join(_OI_ROOT, "metadata", "classes.csv")

CAT_MID = "/m/01yrx"

_MAX_EXAMPLES = os.environ.get("OI_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None
_SAMPLE_SEED = int(os.environ.get("OI_SAMPLE_SEED", "42"))
_VAL_FRACTION = float(os.environ.get("OI_VAL_FRACTION", "0.05"))

_MIN_FRAC_RAW = os.environ.get("OI_MIN_BBOX_AREA_FRAC", "").strip()
MIN_BBOX_AREA_FRAC = float(_MIN_FRAC_RAW) if _MIN_FRAC_RAW else (1.0 / 30.0)


def _load_class_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with open(CLASSES_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                names[row[0]] = row[1]
    return names


def _load_cat_bbox_annotations() -> dict[str, list[dict]]:
    """Return {ImageID: [annotations]} for cat bboxes only."""
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(BBOX_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] == CAT_MID:
                by_image[row["ImageID"]].append(row)
    return dict(by_image)


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _bbox_area(ann: dict, w: int, h: int) -> float:
    xmin = float(ann["XMin"]) * w
    xmax = float(ann["XMax"]) * w
    ymin = float(ann["YMin"]) * h
    ymax = float(ann["YMax"]) * h
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def _encode_ann(ann: dict, w: int, h: int) -> dict:
    xmin = float(ann["XMin"]) * w
    xmax = float(ann["XMax"]) * w
    ymin = float(ann["YMin"]) * h
    ymax = float(ann["YMax"]) * h
    bw = max(xmax - xmin, 0.0)
    bh = max(ymax - ymin, 0.0)
    return {
        "bbox": np.array([xmin, ymin, bw, bh], dtype=np.float32),
        "area": np.float32(bw * bh),
        "category_id": np.int64(0),
        "iscrowd": np.int64(1 if ann.get("IsGroupOf", "0") == "1" else 0),
    }


def preprocess_func(example):
    image = example["image"]
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    example["image"] = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    return example


class OpenImagesCat(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Open Images V7 cat detection: one row per image. "
                "All cat bounding-box annotations grouped per image. "
                "Schema matches lvis_person.py."
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
            homepage="https://storage.googleapis.com/openimages/web/index.html",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples("train"),
            "validation": self._generate_examples("validation"),
        }

    def _generate_examples(self, split_name: str):
        name_by_mid = _load_class_names()
        cat_name = name_by_mid.get(CAT_MID, "Cat")
        category_feats = [
            {
                "category_id": np.int64(0),
                "class_idx": np.int64(0),
                "category_name": cat_name,
            }
        ]

        by_image = _load_cat_bbox_annotations()
        image_ids = sorted(by_image.keys())
        random.Random(_SAMPLE_SEED).shuffle(image_ids)

        val_count = max(1, int(len(image_ids) * _VAL_FRACTION))
        if split_name == "validation":
            image_ids = image_ids[:val_count]
        else:
            image_ids = image_ids[val_count:]

        n = 0
        for idx, image_id in enumerate(image_ids):
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            file_name = f"{image_id}.jpg"
            rgb_path = os.path.join(IMAGES_DIR, file_name)
            if not os.path.isfile(rgb_path):
                continue

            img = PILImage.open(rgb_path)
            w, h = img.size
            img_area = float(w * h)

            anns = by_image[image_id]
            if MIN_BBOX_AREA_FRAC > 0 and img_area > 0:
                need = MIN_BBOX_AREA_FRAC * img_area
                anns = [a for a in anns if _bbox_area(a, w, h) >= need]
            if not anns:
                continue

            key = f"{split_name}_{n}_{image_id}"
            n += 1
            yield key, {
                "image": rgb_path,
                "annotations": [
                    {
                        **_encode_ann(a, w, h),
                        "class_idx": np.int64(0),
                        "category_name": cat_name,
                    }
                    for a in anns
                ],
                "categories": category_feats,
                "metadata": {
                    "image_id": np.int64(n),
                    "file_name": file_name,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": _relpath(rgb_path, IMAGES_DIR),
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
    ds = tfds.load("open_images_cat", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


def sanity_check(
    ds_dir,
    split,
    num_batches=1,
    batch_size=8,
    shuffle=False,
    max_show=4,
    dataset_name: str = "open_images_cat",
):
    """Visualize RGB with all bboxes drawn on."""
    n_take = num_batches * batch_size
    ds = tfds.load(
        dataset_name, split=split, data_dir=ds_dir, shuffle_files=shuffle
    )
    ds = ds.take(n_take)

    examples: list = []
    for ex in ds:
        examples.append(ex)
    if not examples:
        print("sanity_check: no examples from dataset.")
        return

    for bi in range(num_batches):
        start = bi * batch_size
        rows = examples[start : start + batch_size]
        if not rows:
            break

        n = min(max_show, len(rows))
        fig, axes = plt.subplots(n, 1, figsize=(8, 5.5 * n))
        if n == 1:
            axes = [axes]

        for j in range(n):
            ex = rows[j]
            img = np.clip(ex["image"].numpy(), 0, 255).astype(np.uint8)
            h = int(ex["metadata"]["height"].numpy())
            w = int(ex["metadata"]["width"].numpy())
            image_id = int(ex["metadata"]["image_id"].numpy())

            axes[j].imshow(img)
            anns = ex["annotations"]
            num_boxes = anns["bbox"].shape[0]
            for k in range(num_boxes):
                bbox = anns["bbox"][k].numpy()
                x, y, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                nm = anns["category_name"][k].numpy()
                if isinstance(nm, (bytes, np.bytes_)):
                    nm = nm.decode("utf-8", errors="replace")

                axes[j].add_patch(
                    plt.Rectangle(
                        (x, y), bw, bh,
                        fill=False, edgecolor="lime", linewidth=1.5,
                    )
                )
                axes[j].text(
                    x, max(y - 2, 0), nm[:28],
                    color="yellow", fontsize=7, clip_on=True,
                )

            axes[j].set_title(f"img_id={image_id}  {h}x{w}  boxes={num_boxes}")
            axes[j].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"{dataset_name}_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
