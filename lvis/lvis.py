"""
TFDS builder for LVIS v1 object detection.

Exact default paths:

- ``/kmh-nfs-ssd-us-mount/code/kristine/lvm/data/lvis/train2017``
- ``/kmh-nfs-ssd-us-mount/code/kristine/lvm/data/lvis/val2017``
- ``/kmh-nfs-ssd-us-mount/code/kristine/lvm/data/lvis/annotations/lvis_v1_train.json``
- ``/kmh-nfs-ssd-us-mount/code/kristine/lvm/data/lvis/annotations/lvis_v1_val.json``

Override with env vars:

- ``LVIS_IMAGES_ROOT``
- ``LVIS_TRAIN_JSON``
- ``LVIS_VAL_JSON``

  tfds build lvis/lvis.py --data_dir=/path/to/tfds_output
  tfds.load("lvis", data_dir=..., split="train")
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


def _encode_ann(ann: dict) -> dict:
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox"]], dtype=np.float32),
        "area": np.float32(ann.get("area", 0.0)),
        "category_id": np.int64(ann["category_id"]),
        "iscrowd": np.int64(ann.get("iscrowd", 0)),
    }


class Lvis(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="LVIS v1 RGB images with object detection annotations.",
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
        for i, img in enumerate(images):
            if i % 3 != 0:
                continue
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break

            image_id = int(img["id"])
            file_name = img["coco_url"].rsplit("/", 2)[-2] + "/" + img["coco_url"].rsplit("/", 1)[-1]
            h, w = int(img["height"]), int(img["width"])
            rgb_path = os.path.join(IMAGES_ROOT, file_name)

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
                    for a in by_image.get(image_id, [])
                ],
                "categories": category_feats,
                "metadata": {
                    "image_id": np.int64(image_id),
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
):
    kw: dict = {"split": split}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load("lvis", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.ragged_batch(batch_size).prefetch(tf.data.AUTOTUNE)


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


def _lvis_colored_instance_masks(anns: list, h: int, w: int) -> np.ndarray:
    """RGB image: each instance mask gets a distinct color (from JSON segmentations)."""
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    golden = 0.3819660112501051
    for i, ann in enumerate(anns):
        m = _decode_lvis_segmentation(ann["segmentation"], h, w)
        hue = (i * golden) % 1.0
        rgb = np.array(plt.cm.hsv(hue)[:3], dtype=np.float32) * 255.0
        canvas[m > 0] = rgb
    return np.clip(canvas, 0, 255).astype(np.uint8)


def preprocess_func(example):
    image = example["image"]
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    example["image"] = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    return example


def sanity_check(
    ds_dir,
    split,
    num_batches=1,
    batch_size=8,
    shuffle=False,
    max_show=4,
    masks_from_json=True,
):
    """Visualize full-resolution RGB, decoded instance masks, and overlay (masks from LVIS JSON).

    TFDS does not store segmentations; masks are decoded from the same JSON as the builder
    (``LVIS_TRAIN_JSON`` / ``LVIS_VAL_JSON``) keyed by ``image_id``. Requires **pycocotools**.
    Set ``masks_from_json=False`` to only draw bounding boxes (no JSON load).
    """
    n_take = num_batches * batch_size
    ds = tfds.load("lvis", split=split, data_dir=ds_dir, shuffle_files=shuffle)
    ds = ds.take(n_take)

    json_path = TRAIN_JSON if split == "train" else VAL_JSON
    by_image: dict[int, list] | None = None
    if masks_from_json:
        try:
            import pycocotools  # noqa: F401
        except ImportError:
            print("pycocotools not installed; falling back to bbox-only (masks_from_json=False).")
            masks_from_json = False
        if masks_from_json:
            if split == "train":
                print("sanity_check: loading train annotations JSON (large); use split='validation' for a lighter check.")
            with open(json_path, encoding="utf-8") as f:
                lvis = json.load(f)
            by_image = {}
            for ann in lvis["annotations"]:
                by_image.setdefault(int(ann["image_id"]), []).append(ann)

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
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for j in range(n):
            ex = rows[j]
            img = np.clip(ex["image"].numpy(), 0, 255).astype(np.uint8)
            h, w = int(ex["metadata"]["height"].numpy()), int(ex["metadata"]["width"].numpy())
            image_id = int(ex["metadata"]["image_id"].numpy())

            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f"image  id={image_id}  {h}×{w}")
            axes[j, 0].axis("off")

            if by_image is not None and masks_from_json:
                json_anns = by_image.get(image_id, [])
                colored = _lvis_colored_instance_masks(json_anns, h, w)
                axes[j, 1].imshow(colored)
                axes[j, 1].set_title(f"masks (JSON, n={len(json_anns)})")
                axes[j, 1].axis("off")
                blend = (img.astype(np.float32) * 0.55 + colored.astype(np.float32) * 0.45).astype(
                    np.uint8
                )
                axes[j, 2].imshow(blend)
            else:
                axes[j, 1].imshow(np.ones((h, w, 3), dtype=np.uint8) * 240)
                axes[j, 1].set_title("(no mask JSON)")
                axes[j, 1].axis("off")
                axes[j, 2].imshow(img)

            ann_feats = ex["annotations"]
            bbox = ann_feats["bbox"].numpy()
            names = ann_feats["category_name"].numpy()
            nb = int(bbox.shape[0])
            ax_ov = axes[j, 2]
            for k in range(nb):
                x, y, bw, bh = [float(t) for t in bbox[k]]
                ax_ov.add_patch(
                    plt.Rectangle(
                        (x, y),
                        bw,
                        bh,
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.0,
                    )
                )
                nm = names[k]
                if isinstance(nm, (bytes, np.bytes_)):
                    nm = nm.decode("utf-8", errors="replace")
                ax_ov.text(
                    x,
                    max(y - 2, 0),
                    nm[:28],
                    color="yellow",
                    fontsize=6,
                    clip_on=True,
                )
            axes[j, 2].set_title("overlay + bboxes")
            axes[j, 2].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"lvis_sanity_{date_str}_{split}_batch_{bi + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
