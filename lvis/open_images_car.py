"""
Car bounding box detection

Build::
    tfds build lvis/open_images_car.py --data_dir=/mnt/klum/lvis/open_images_car_tfds

Load::
    tfds.load("open_images_car", data_dir=..., split="train")
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from PIL import Image as PILImage

_OI_ROOT = "/mnt/klum/lvis/instance_segmentation/open-images-v7/train"
IMAGES_DIR = os.path.join(_OI_ROOT, "data")
SEG_CSV = os.path.join(_OI_ROOT, "labels", "segmentations.csv")
CLASSES_CSV = os.path.join(_OI_ROOT, "metadata", "classes.csv")

CAR_MID = "/m/0k4j"

_MAX_EXAMPLES = os.environ.get("OI_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None
_SAMPLE_SEED = int(os.environ.get("OI_SAMPLE_SEED", "42"))
_VAL_FRACTION = float(os.environ.get("OI_VAL_FRACTION", "0.05"))


def _load_class_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with open(CLASSES_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                names[row[0]] = row[1]
    return names


def _load_car_annotations() -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    with open(SEG_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] == CAR_MID:
                by_image[row["ImageID"]].append(row)
    return dict(by_image)


class OpenImagesCar(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Open Images V7 car detection: per-image RGB with all car "
                "bounding-box annotations.  Schema matches lvis.py so "
                "lvis_detection_tfds.load_data works directly."
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
                            "json_path": tfds.features.Text(),
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
        car_name = name_by_mid.get(CAR_MID, "Car")
        by_image = _load_car_annotations()

        image_ids = sorted(by_image.keys())
        random.Random(_SAMPLE_SEED).shuffle(image_ids)

        val_count = max(1, int(len(image_ids) * _VAL_FRACTION))
        if split_name == "validation":
            image_ids = image_ids[:val_count]
        else:
            image_ids = image_ids[val_count:]

        category_feats = [
            {
                "category_id": np.int64(0),
                "class_idx": np.int64(0),
                "category_name": car_name,
            }
        ]

        n = 0
        for image_id in image_ids:
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            file_name = f"{image_id}.jpg"
            rgb_path = os.path.join(IMAGES_DIR, file_name)
            if not os.path.isfile(rgb_path):
                continue

            img = PILImage.open(rgb_path)
            w, h = img.size

            anns = by_image[image_id]
            encoded_anns = []
            for ann in anns:
                xmin = float(ann["BoxXMin"]) * w
                xmax = float(ann["BoxXMax"]) * w
                ymin = float(ann["BoxYMin"]) * h
                ymax = float(ann["BoxYMax"]) * h
                bw = max(xmax - xmin, 0.0)
                bh = max(ymax - ymin, 0.0)
                encoded_anns.append(
                    {
                        "bbox": np.array([xmin, ymin, bw, bh], dtype=np.float32),
                        "area": np.float32(bw * bh),
                        "category_id": np.int64(0),
                        "class_idx": np.int64(0),
                        "category_name": car_name,
                        "iscrowd": np.int64(0),
                    }
                )

            key = f"{split_name}_{image_id}"
            n += 1
            yield key, {
                "image": rgb_path,
                "annotations": encoded_anns,
                "categories": category_feats,
                "metadata": {
                    "image_id": np.int64(n),
                    "file_name": file_name,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": file_name,
                    "json_path": "segmentations.csv",
                },
            }


def preprocess_func(example):
    image = example["image"]
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    example["image"] = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    return example


def load_data(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
    ignore_errors: bool = True,
):
    kw: dict = {"split": split}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load("open_images_car", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if ignore_errors:
        ds = ds.ignore_errors()
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
    dataset_name: str = "open_images_car",
):
    """Visualize RGB + bounding box overlay (one image per row)."""
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
        fig, axes = plt.subplots(n, 1, figsize=(8, 4.5 * n))
        if n == 1:
            axes = [axes]

        for j in range(n):
            ex = rows[j]
            img = np.clip(ex["image"].numpy(), 0, 255).astype(np.uint8)
            h = int(ex["metadata"]["height"].numpy())
            w = int(ex["metadata"]["width"].numpy())

            ann_feats = ex["annotations"]
            bbox = ann_feats["bbox"].numpy()
            names = ann_feats["category_name"].numpy()
            nb = int(bbox.shape[0])

            axes[j].imshow(img)
            for k in range(nb):
                x, y, bw, bh = [float(t) for t in bbox[k]]
                axes[j].add_patch(
                    plt.Rectangle(
                        (x, y), bw, bh,
                        fill=False, edgecolor="lime", linewidth=1.5,
                    )
                )
                nm = names[k]
                if isinstance(nm, (bytes, np.bytes_)):
                    nm = nm.decode("utf-8", errors="replace")
                axes[j].text(
                    x, max(y - 2, 0), nm[:28],
                    color="yellow", fontsize=7, clip_on=True,
                )
            axes[j].set_title(f"image  {h}x{w}  boxes={nb}")
            axes[j].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"{dataset_name}_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
