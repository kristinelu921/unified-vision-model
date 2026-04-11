"""
path is gs://kmh-gcp-asia-northeast1-b/kristine/lvm/lvis_giraffe_tfds/
"""

from __future__ import annotations

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime


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


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _encode_ann(ann: dict, h: int, w: int) -> dict:
    mask = _decode_lvis_segmentation(ann["segmentation"], h, w)
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox"]], dtype=np.float32),
        "area": np.float32(ann.get("area", 0.0)),
        "category_id": np.int64(ann["category_id"]),
        "iscrowd": np.int64(ann.get("iscrowd", 0)),
        "mask_raster": mask.astype(np.uint8)[..., np.newaxis],
    }


class LvisGiraffe(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("2.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        instance_feats = tfds.features.FeaturesDict(
            {
                "bbox": tfds.features.Tensor(shape=(4,), dtype=np.float32),
                "area": tfds.features.Tensor(shape=(), dtype=np.float32),
                "category_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                "class_idx": tfds.features.Tensor(shape=(), dtype=np.int64),
                "category_name": tfds.features.Text(),
                "iscrowd": tfds.features.Tensor(shape=(), dtype=np.int64),
                "mask_raster": tfds.features.Tensor(
                    shape=(None, None, 1),
                    dtype=np.uint8,
                    encoding="zlib",
                    doc="Binary mask for this instance; values 0 or 1.",
                ),
            },
            doc="One LVIS object: bbox, class, decoded segmentation mask.",
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "LVIS giraffe v1: one row per image that has exactly one giraffe instance. "
                "Full image RGB + that single instance bbox and ``mask_raster`` "
                "(H×W×1, same resolution as the image)."
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

        images_by_id: dict[int, dict] = {int(img["id"]): img for img in lvis["images"]}

        anns = list(lvis["annotations"])
        giraffe_anns = [ann for ann in anns if int(ann["category_id"]) == 496]


        n = 0
        for ann in giraffe_anns:
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            image_id = int(ann["image_id"])
            img = images_by_id.get(image_id)
            if img is None:
                continue

            file_name = img["coco_url"].rsplit("/", 2)[-2] + "/" + img["coco_url"].rsplit("/", 1)[-1]
            h, w = int(img["height"]), int(img["width"])
            rgb_path = os.path.join(IMAGES_ROOT, file_name)
            if not os.path.isfile(rgb_path):
                continue

            ann_id = int(ann.get("id", n))
            key = f"{split_name}_{image_id}_{ann_id}"
            n += 1
            encoded = _encode_ann(ann, h, w)
            yield key, {
                "image": rgb_path,
                "instance": {
                    "bbox": encoded["bbox"],
                    "area": encoded["area"],
                    "category_id": encoded["category_id"],
                    "iscrowd": encoded["iscrowd"],
                    "mask_raster": encoded["mask_raster"],
                    "class_idx": np.int64(class_idx_by_id[int(ann["category_id"])]),
                    "category_name": name_by_id[int(ann["category_id"])],
                },
                "metadata": {
                    "image_id": np.int64(image_id),
                    "annotation_id": np.int64(ann_id),
                    "file_name": file_name,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": _relpath(rgb_path, IMAGES_ROOT),
                    "json_path": _relpath(os.path.abspath(json_path), DATA_ROOT),
                },
            }


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
    ds = tfds.load("lvis_giraffe", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if ignore_errors:
        ds = ds.ignore_errors()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _decode_lvis_segmentation(seg, h: int, w: int) -> np.ndarray:
    """Decode LVIS ``segmentation`` (RLE dict or polygon list) to H×W uint8 in ``{0,1}``."""
    from pycocotools import mask as mask_util

    if isinstance(seg, dict) and "counts" in seg:
        return mask_util.decode(seg).astype(np.uint8)
    if isinstance(seg, list):
        if not seg:
            return np.zeros((h, w), dtype=np.uint8)
        rles = mask_util.frPyObjects(seg, h, w)
        rle = mask_util.merge(rles) if isinstance(rles, list) else rles
        return mask_util.decode(rle).astype(np.uint8)
    return np.zeros((h, w), dtype=np.uint8)



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


def sanity_check(
    ds_dir,
    split,
    num_batches=1,
    batch_size=8,
    shuffle=False,
    max_show=4,
    dataset_name: str = "lvis_giraffe",
    ignore_errors: bool = True,
):
    """Visualize RGB and B/W ``instance/mask_raster`` (one instance per row).

    ``dataset_name`` must match the folder under ``data_dir`` (default ``lvis_instance``).
    """
    n_take = num_batches * batch_size
    ds = tfds.load(
        dataset_name, split=split, data_dir=ds_dir, shuffle_files=shuffle
    )
    if ignore_errors:
        ds = ds.ignore_errors()
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
        fig, axes = plt.subplots(n, 2, figsize=(8.5, 3.8 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for j in range(n):
            ex = rows[j]
            img = np.clip(ex["image"].numpy(), 0, 255).astype(np.uint8)
            h, w = int(ex["metadata"]["height"].numpy()), int(ex["metadata"]["width"].numpy())
            image_id = int(ex["metadata"]["image_id"].numpy())
            ann_id = int(ex["metadata"]["annotation_id"].numpy())
            inst = ex["instance"]
            mk = inst["mask_raster"].numpy()
            cat = int(inst["category_id"].numpy())
            nm = inst["category_name"].numpy()
            if isinstance(nm, (bytes, np.bytes_)):
                nm = nm.decode("utf-8", errors="replace")

            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f"image  img_id={image_id} ann_id={ann_id}  {h}×{w}")
            axes[j, 0].axis("off")

            axes[j, 1].imshow(mk[..., 0], cmap="gray", vmin=0, vmax=1)
            axes[j, 1].set_title(f"mask_raster  cat={cat}")
            axes[j, 1].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"{dataset_name}_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
