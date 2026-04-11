"""
TFDS builder: COCO 2017 cat detection (bounding boxes).

Compatible with ``jflux/jflux2/dataloaders/lvis_detection_tfds.py``:
- top-level has ``image``, ``annotations``, ``metadata``
- ``annotations`` contains ``bbox`` (xywh in pixels) used by bbox_mask builder

Requires local data prepared by ``lvis/download_coco_detection_cat.py``.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
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


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _encode_ann(ann: dict) -> dict:
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox_xywh"]], dtype=np.float32),
        "area": np.float32(float(ann.get("area", 0.0))),
        "category_id": np.int64(int(ann.get("category_id", 17))),
        "iscrowd": np.int64(int(ann.get("iscrowd", 0))),
    }


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


class CocoCatDetection(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.1.0")

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
            image_id = int(rec["image_id"])
            file_name = rec["file_name"]
            h, w = int(rec["height"]), int(rec["width"])
            rel_path = rec["image_rel_path"]
            rgb_path = os.path.join(IMAGES_ROOT, rel_path)
            if not os.path.isfile(rgb_path):
                continue

            anns = list(rec.get("annotations", []))
            if not anns:
                continue
            key = f"{split_name}_{image_id}_{file_name}"
            n += 1
            yield key, {
                "image": rgb_path,
                "annotations": [
                    {
                        **_encode_ann(ann),
                        "class_idx": np.int64(0),
                        "category_name": category_name,
                    }
                    for ann in anns
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
):
    kw: dict = {"split": split}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load("coco_cat_detection", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _normalize_style(style: str) -> str:
    normalized = str(style).strip().lower()
    if normalized not in {"fill", "outline"}:
        raise ValueError(f"`style` must be 'fill' or 'outline', got {style!r}.")
    return normalized


def _box_mask(
    xs: tf.Tensor,
    ys: tf.Tensor,
    *,
    x0: tf.Tensor,
    y0: tf.Tensor,
    x1: tf.Tensor,
    y1: tf.Tensor,
    style: str,
    border_width: float,
) -> tf.Tensor:
    inside_x = tf.logical_and(xs >= x0, xs < x1)
    inside_y = tf.logical_and(ys >= y0, ys < y1)
    inside = tf.logical_and(inside_x, inside_y)
    if style == "fill":
        return inside

    near_left = tf.abs(xs - x0) < border_width
    near_right = tf.abs(xs - x1) < border_width
    near_top = tf.abs(ys - y0) < border_width
    near_bottom = tf.abs(ys - y1) < border_width
    border = tf.logical_or(
        tf.logical_or(tf.logical_and(inside, near_left), tf.logical_and(inside, near_right)),
        tf.logical_or(tf.logical_and(inside, near_top), tf.logical_and(inside, near_bottom)),
    )
    return border


def bbox_mask_from_annotations(
    example,
    *,
    random_color: bool = False,
    style: str = "outline",
    border_width: float = 4.0,
    overlay_alpha: float = 1.0,
):
    """Copied logic from jflux lvis_detection_tfds for visual parity."""
    image = tf.cast(example["image"], tf.uint8)
    bboxes = tf.cast(example["annotations"]["bbox"], tf.float32)
    style = _normalize_style(style)

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    ys = tf.cast(tf.range(height), tf.float32)[tf.newaxis, :, tf.newaxis]
    xs = tf.cast(tf.range(width), tf.float32)[tf.newaxis, tf.newaxis, :]

    x0 = bboxes[:, 0][:, tf.newaxis, tf.newaxis]
    y0 = bboxes[:, 1][:, tf.newaxis, tf.newaxis]
    x1 = x0 + tf.maximum(bboxes[:, 2][:, tf.newaxis, tf.newaxis], 0.0)
    y1 = y0 + tf.maximum(bboxes[:, 3][:, tf.newaxis, tf.newaxis], 0.0)

    box_masks = _box_mask(
        xs,
        ys,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        style=style,
        border_width=float(border_width),
    )
    if random_color:
        colors = tf.random.stateless_uniform(
            shape=[tf.shape(bboxes)[0], 3],
            seed=tf.constant([1234, 5678], dtype=tf.int32),
            minval=0,
            maxval=256,
            dtype=tf.int32,
        )
        colored = tf.cast(box_masks[..., tf.newaxis], tf.int32) * colors[:, tf.newaxis, tf.newaxis, :]
        box_rgb = tf.cast(tf.reduce_max(colored, axis=0), tf.uint8)
    else:
        mask_2d = tf.reduce_any(box_masks, axis=0)
        red = tf.cast(mask_2d, tf.uint8) * 255
        zeros = tf.zeros_like(red)
        box_rgb = tf.stack([red, zeros, zeros], axis=-1)

    box_rgb_f = tf.cast(box_rgb, tf.float32)
    image_f = tf.cast(image, tf.float32)
    active = tf.reduce_any(box_rgb > 0, axis=-1, keepdims=True)
    overlay = image_f * (1.0 - overlay_alpha) + box_rgb_f * overlay_alpha
    mask = tf.cast(tf.where(active, overlay, image_f), tf.uint8)
    return {
        "image": image,
        "mask": mask,
        "mask_valid": tf.greater(tf.shape(bboxes)[0], 0),
        "metadata": example["metadata"],
    }


def sanity_check(
    ds_dir,
    split,
    num_batches=1,
    batch_size=8,
    shuffle=False,
    max_show=4,
    dataset_name: str = "coco_cat_detection",
    random_color: bool = False,
    style: str = "outline",
    border_width: float = 4.0,
    overlay_alpha: float = 1.0,
):
    """Visualize image + bbox-mask overlay using jflux lvis_detection logic."""
    n_take = num_batches * batch_size
    ds = tfds.load(dataset_name, split=split, data_dir=ds_dir, shuffle_files=shuffle)
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
        fig, axes = plt.subplots(n, 2, figsize=(12, 4.5 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for j in range(n):
            ex = rows[j]
            vis = bbox_mask_from_annotations(
                ex,
                random_color=random_color,
                style=style,
                border_width=border_width,
                overlay_alpha=overlay_alpha,
            )
            img = np.clip(ex["image"].numpy(), 0, 255).astype(np.uint8)
            mask = np.clip(vis["mask"].numpy(), 0, 255).astype(np.uint8)
            h = int(ex["metadata"]["height"].numpy())
            w = int(ex["metadata"]["width"].numpy())
            image_id = int(ex["metadata"]["image_id"].numpy())
            num_boxes = int(ex["annotations"]["bbox"].shape[0])
            valid = bool(vis["mask_valid"].numpy())

            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f"image  img_id={image_id}  {h}x{w}  boxes={num_boxes}")
            axes[j, 0].axis("off")

            axes[j, 1].imshow(mask)
            axes[j, 1].set_title(f"mask overlay  valid={valid}  style={style}")
            axes[j, 1].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"{dataset_name}_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
