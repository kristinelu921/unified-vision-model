"""
TFDS builder for EntitySeg **instance segmentation** only (insseg JSON).

**One dataset example = one instance (one object).** If an image has 5 instances, you get **5**
examples: same ``image`` path repeated, each with a single ``instance`` dict containing that
object’s **COCO RLE** (``segmentation_size`` + ``segmentation_counts``) and a decoded **raster**
``mask_raster`` (H×W×1, ``uint8`` 0/1). Requires **pycocotools** at build time.

Paths:

- ``ENTITYSEG_DATA_ROOT`` — default ``/mnt/klum/entityseg/data``
- ``ENTITYSEG_IMAGES_ROOT`` — where ``file_name`` resolves
- ``ENTITYSEG_TRAIN_JSON`` / ``ENTITYSEG_VAL_JSON`` — insseg JSONs
- ``ENTITYSEG_MAX_EXAMPLES_PER_SPLIT`` — optional cap on **instance rows** yielded (not images)

Missing RGB under ``ENTITYSEG_IMAGES_ROOT`` (no file at ``join(root, file_name)``): **skipped** for all instances on that image.

For **semantic segmentation** (one label map per image), see ``entity_seg_semantic.py``.

  tfds build entityseg/entity_seg_instance.py --data_dir=/path/to/tfds_output
  tfds.load("entity_seg_instance", data_dir=..., split="train")

  ``sanity_check(ds_dir, split=...)`` saves PNGs: RGB | ``mask_raster`` | overlay + bbox.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

try:
    from pycocotools import mask as mask_util
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "entity_seg_instance needs pycocotools to decode RLE masks. "
        "Install with: pip install pycocotools"
    ) from e

Image.MAX_IMAGE_PIXELS = None


DATA_ROOT = os.environ.get("ENTITYSEG_DATA_ROOT", "/mnt/klum/entityseg/data")
IMAGES_ROOT = os.environ.get(
    "ENTITYSEG_IMAGES_ROOT",
    os.path.join(DATA_ROOT, "entityseg_test_images_dl"),
)

TRAIN_JSON = os.environ.get(
    "ENTITYSEG_TRAIN_JSON",
    os.path.join(DATA_ROOT, "entityseg_insseg_train.json"),
)
VAL_JSON = os.environ.get(
    "ENTITYSEG_VAL_JSON",
    os.path.join(DATA_ROOT, "entityseg_insseg_val.json"),
)

_MAX_EXAMPLES = os.environ.get("ENTITYSEG_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


@lru_cache(maxsize=1)
def _image_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not os.path.isdir(IMAGES_ROOT):
        return index
    for dirpath, _dirnames, filenames in os.walk(IMAGES_ROOT):
        for filename in filenames:
            index.setdefault(filename, os.path.join(dirpath, filename))
    return index


def _resolve_image_path(file_name: str) -> str | None:
    direct = os.path.join(IMAGES_ROOT, file_name)
    if os.path.isfile(direct):
        return direct
    return _image_index().get(os.path.basename(file_name))


def _rle_to_raster(ann: dict) -> np.ndarray:
    """Decode COCO RLE to (H, W, 1) uint8 in {0, 1}."""
    seg = ann["segmentation"]
    h, w = int(seg["size"][0]), int(seg["size"][1])
    counts = seg["counts"]
    rle: dict = {"size": [h, w], "counts": counts}
    m = mask_util.decode(rle)
    return m.astype(np.uint8)[..., np.newaxis]


def _encode_instance(ann: dict) -> dict:
    """One COCO annotation: bbox + class + RLE + decoded raster mask."""
    seg = ann["segmentation"]
    h, w = int(seg["size"][0]), int(seg["size"][1])
    bbox = [float(x) for x in ann["bbox"]]
    counts = str(seg["counts"])
    return {
        "bbox": np.asarray(bbox, dtype=np.float32),
        "area": np.float32(ann.get("area", 0.0)),
        "category_id": np.int64(ann["category_id"]),
        "iscrowd": np.int64(ann.get("iscrowd", 0)),
        "segmentation_size": np.asarray([h, w], dtype=np.int64),
        "segmentation_counts": counts,
        "mask_raster": _rle_to_raster(ann),
    }


class EntitySegInstance(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.1.0")

    def _info(self) -> tfds.core.DatasetInfo:
        instance_feats = tfds.features.FeaturesDict(
            {
                "bbox": tfds.features.Tensor(shape=(4,), dtype=np.float32),
                "area": tfds.features.Tensor(shape=(), dtype=np.float32),
                "category_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                "iscrowd": tfds.features.Tensor(shape=(), dtype=np.int64),
                "segmentation_size": tfds.features.Tensor(shape=(2,), dtype=np.int64),
                "segmentation_counts": tfds.features.Text(),
                "mask_raster": tfds.features.Tensor(
                    shape=(None, None, 1),
                    dtype=np.uint8,
                    encoding="zlib",
                    doc="Binary mask decoded from RLE; values 0 or 1.",
                ),
            },
            doc="One object: RLE fields + ``mask_raster`` (H×W×1, same as decode(RLE)).",
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "EntitySeg instance segmentation: **one row per instance**. "
                "RLE mask fields live under ``instance``."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "instance": instance_feats,
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "image_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "annotation_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "file_name": tfds.features.Text(),
                            "width": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "height": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "rel_image_path": tfds.features.Text(),
                            "json_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "instance"),
            homepage="https://arxiv.org/abs/2211.05776",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(TRAIN_JSON, "train"),
            "validation": self._generate_examples(VAL_JSON, "validation"),
        }

    def _generate_examples(self, json_path: str, split_name: str):
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)

        by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)

        n = 0
        for img in coco["images"]:
            image_id = int(img["id"])
            file_name = img["file_name"]
            h, w = int(img["height"]), int(img["width"])
            rgb_path = _resolve_image_path(file_name)
            if rgb_path is None or not os.path.isfile(rgb_path):
                continue

            meta_common = {
                "image_id": np.int64(image_id),
                "file_name": file_name,
                "width": np.int32(w),
                "height": np.int32(h),
                "rel_image_path": _relpath(rgb_path, IMAGES_ROOT),
                "json_path": _relpath(os.path.abspath(json_path), DATA_ROOT),
            }

            for ann in by_image.get(image_id, []):
                if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                    return
                ann_id = int(ann.get("id", n))
                key = f"{split_name}_{image_id}_{ann_id}"
                n += 1
                yield key, {
                    "image": rgb_path,
                    "instance": _encode_instance(ann),
                    "metadata": {
                        **meta_common,
                        "annotation_id": np.int64(ann_id),
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
    ds = tfds.load("entity_seg_instance", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_func(example):
    image = tf.cast(example["image"], tf.float32)
    mask = tf.cast(example["instance"]["mask_raster"], tf.float32)

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    x_scale = 256.0 / w
    y_scale = 256.0 / h

    example["image"] = tf.cast(tf.image.resize(image, [256, 256], method="bilinear"), tf.uint8)
    example["instance"]["mask_raster"] = tf.cast(
        tf.image.resize(mask, [256, 256], method="nearest"),
        tf.uint8,
    )

    bbox = tf.cast(example["instance"]["bbox"], tf.float32)
    example["instance"]["bbox"] = tf.stack(
        [bbox[0] * x_scale, bbox[1] * y_scale, bbox[2] * x_scale, bbox[3] * y_scale]
    )
    return example


def _overlay_mask_on_rgb(
    image: np.ndarray,
    mask_hw1: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.25, 0.2),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary mask (H,W,1) onto an RGB uint8 image."""
    img = np.clip(image.astype(np.float32), 0, 255)
    m = np.clip(mask_hw1[..., 0].astype(np.float32), 0, 1.0)[..., None]
    col = np.array(color, dtype=np.float32) * 255.0
    out = img * (1.0 - alpha * m) + col * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def sanity_check(
    ds_dir: str,
    split: str = "train",
    num_batches: int = 1,
    batch_size: int = 4,
    shuffle: bool = False,
    max_show: int = 4,
):
    """Load a few batches and save a figure per batch: RGB | ``mask_raster`` | overlay + bbox."""
    ds = load_data(
        data_dir=ds_dir,
        split=split,
        batch_size=batch_size,
        repeat=False,
        shuffle=shuffle,
    )
    ds = ds.take(num_batches)

    for bi, batch in enumerate(ds):
        imgs = batch["image"].numpy()
        inst = batch["instance"]
        masks = inst["mask_raster"].numpy()
        bboxes = inst["bbox"].numpy()
        cats = inst["category_id"].numpy()
        n = min(max_show, int(imgs.shape[0]))
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)
        for j in range(n):
            img = np.clip(imgs[j], 0, 255).astype(np.uint8)
            mk = masks[j]
            h, w = int(mk.shape[0]), int(mk.shape[1])
            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f"image  category_id={int(cats[j])}")
            axes[j, 0].axis("off")
            axes[j, 1].imshow(mk[..., 0], cmap="gray", vmin=0, vmax=1)
            axes[j, 1].set_title(f"mask_raster  {h}×{w}")
            axes[j, 1].axis("off")
            ov = _overlay_mask_on_rgb(img, mk)
            axes[j, 2].imshow(ov)
            x, y, bw, bh = [float(t) for t in bboxes[j]]
            rect = plt.Rectangle(
                (x, y),
                bw,
                bh,
                fill=False,
                edgecolor="lime",
                linewidth=1.5,
            )
            axes[j, 2].add_patch(rect)
            axes[j, 2].set_title("overlay + bbox (xywh)")
            axes[j, 2].axis("off")
        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"entity_seg_instance_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
