"""
TFDS builder for LVIS v1 with category-sharded splits.

Rules implemented:
- Category shards for BOTH train and validation splits.
- Validation is a random 5% sample (per category) from all examples.
- Categories with fewer than MIN_CATEGORY_EXAMPLES are dropped.
- Each example is one (image, category) pair containing ONLY annotations
  from that category in that image.
- Same image can appear in multiple category shards.

Build:
  tfds build lvis/lvis_instance_category_sharded.py --data_dir=/path/to/tfds_out

Load example split:
  tfds.load("lvis_instance_category_sharded", split="train_cat_241", data_dir=...)
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

_REPO_ROOT = "/mnt/klum/lvis"
DATA_ROOT = _REPO_ROOT
IMAGES_ROOT = os.environ.get("LVIS_IMAGES_ROOT", DATA_ROOT)
TRAIN_JSON = os.environ.get(
    "LVIS_TRAIN_JSON",
    os.path.join(DATA_ROOT, "annotations", "lvis_v1_train.json"),
)

SPLIT_SEED = int(os.environ.get("LVIS_SHARD_SPLIT_SEED", "42"))
VAL_FRACTION = float(os.environ.get("LVIS_SHARD_VAL_FRACTION", "0.05"))
MIN_CATEGORY_EXAMPLES = int(os.environ.get("LVIS_SHARD_MIN_CATEGORY_EXAMPLES", "20"))
_MAX_CATS_RAW = os.environ.get("LVIS_SHARD_MAX_CATEGORIES", "").strip()
MAX_CATEGORIES = int(_MAX_CATS_RAW) if _MAX_CATS_RAW else None


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _decode_lvis_segmentation(seg, h: int, w: int) -> np.ndarray:
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


def _encode_ann(ann: dict, h: int, w: int) -> dict:
    mask = _decode_lvis_segmentation(ann["segmentation"], h, w)
    return {
        "bbox": np.asarray([float(x) for x in ann["bbox"]], dtype=np.float32),
        "area": np.float32(ann.get("area", 0.0)),
        "category_id": np.int64(ann["category_id"]),
        "iscrowd": np.int64(ann.get("iscrowd", 0)),
        "mask_raster": mask.astype(np.uint8)[..., np.newaxis],
    }


def _merge_masks_or(masks_hw1: list[np.ndarray], h: int, w: int) -> np.ndarray:
    merged = np.zeros((h, w, 1), dtype=np.uint8)
    for m in masks_hw1:
        merged[..., 0] = np.maximum(merged[..., 0], np.asarray(m[..., 0], dtype=np.uint8))
    return merged


class LvisInstanceCategorySharded(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        ann_feats = tfds.features.FeaturesDict(
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
                    doc="Binary mask for this instance.",
                ),
            }
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "LVIS category-sharded instance segmentation. One example = one image and "
                "all instances of ONE category in that image."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "annotations": tfds.features.Sequence(ann_feats),
                    "merged_mask_or": tfds.features.Tensor(
                        shape=(None, None, 1),
                        dtype=np.uint8,
                        encoding="zlib",
                        doc="OR-merged binary mask over all annotations in this example.",
                    ),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "image_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "file_name": tfds.features.Text(),
                            "width": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "height": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "rel_image_path": tfds.features.Text(),
                            "json_path": tfds.features.Text(),
                            "category_id": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "category_name": tfds.features.Text(),
                            "num_instances": tfds.features.Tensor(shape=(), dtype=np.int32),
                        }
                    ),
                }
            ),
            homepage="https://www.lvisdataset.org/",
        )

    def _prepare_index(self):
        if hasattr(self, "_prepared_index"):
            return self._prepared_index

        with open(TRAIN_JSON, encoding="utf-8") as f:
            lvis = json.load(f)

        categories = sorted(lvis["categories"], key=lambda c: int(c["id"]))
        if MAX_CATEGORIES is not None:
            categories = categories[:MAX_CATEGORIES]
        kept_cat_ids = {int(c["id"]) for c in categories}
        class_idx_by_id = {int(cat["id"]): i for i, cat in enumerate(categories)}
        name_by_id = {int(cat["id"]): cat["name"] for cat in categories}
        images_by_id = {int(img["id"]): img for img in lvis["images"]}

        grouped: dict[int, dict[int, list[dict]]] = {}
        for ann in lvis["annotations"]:
            cat_id = int(ann["category_id"])
            if cat_id not in kept_cat_ids:
                continue
            img_id = int(ann["image_id"])
            grouped.setdefault(cat_id, {}).setdefault(img_id, []).append(ann)

        # Build examples as (image_id, anns_of_same_category_in_image).
        examples_by_cat: dict[int, list[tuple[int, list[dict]]]] = {}
        for cat_id, by_img in grouped.items():
            rows = [(img_id, anns) for img_id, anns in by_img.items() if anns]
            if len(rows) >= MIN_CATEGORY_EXAMPLES:
                examples_by_cat[cat_id] = rows

        split_map: dict[str, tuple[int, list[tuple[int, list[dict]]]]] = {}
        rng = random.Random(SPLIT_SEED)
        for cat_id, rows in sorted(examples_by_cat.items()):
            rows = list(rows)
            rng.shuffle(rows)
            val_n = max(1, int(round(len(rows) * VAL_FRACTION)))
            val_rows = rows[:val_n]
            train_rows = rows[val_n:]
            if train_rows:
                split_map[f"train_cat_{cat_id}"] = (cat_id, train_rows)
            if val_rows:
                split_map[f"validation_cat_{cat_id}"] = (cat_id, val_rows)

        self._prepared_index = {
            "split_map": split_map,
            "class_idx_by_id": class_idx_by_id,
            "name_by_id": name_by_id,
            "images_by_id": images_by_id,
        }
        return self._prepared_index

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        prepared = self._prepare_index()
        split_map = prepared["split_map"]
        class_idx_by_id = prepared["class_idx_by_id"]
        name_by_id = prepared["name_by_id"]
        images_by_id = prepared["images_by_id"]

        out = {}
        for split_name, (cat_id, rows) in split_map.items():
            out[split_name] = self._generate_examples(
                split_name=split_name,
                cat_id=cat_id,
                rows=rows,
                class_idx_by_id=class_idx_by_id,
                name_by_id=name_by_id,
                images_by_id=images_by_id,
            )
        return out

    def _generate_examples(
        self,
        *,
        split_name: str,
        cat_id: int,
        rows: list[tuple[int, list[dict]]],
        class_idx_by_id: dict[int, int],
        name_by_id: dict[int, str],
        images_by_id: dict[int, dict],
    ):
        cat_name = name_by_id[cat_id]
        class_idx = class_idx_by_id[cat_id]

        for image_id, anns in rows:
            img = images_by_id.get(int(image_id))
            if img is None:
                continue

            file_name = img["coco_url"].rsplit("/", 2)[-2] + "/" + img["coco_url"].rsplit("/", 1)[-1]
            h, w = int(img["height"]), int(img["width"])
            rgb_path = os.path.join(IMAGES_ROOT, file_name)
            if not os.path.isfile(rgb_path):
                continue

            encoded = []
            masks = []
            for ann in anns:
                enc = _encode_ann(ann, h, w)
                masks.append(enc["mask_raster"])
                encoded.append(
                    {
                        **enc,
                        "class_idx": np.int64(class_idx),
                        "category_name": cat_name,
                    }
                )
            if not encoded:
                continue

            merged_mask = _merge_masks_or(masks, h, w)
            key = f"{split_name}_{image_id}_{cat_id}"
            yield key, {
                "image": rgb_path,
                "annotations": encoded,
                "merged_mask_or": merged_mask,
                "metadata": {
                    "image_id": np.int64(image_id),
                    "file_name": file_name,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": _relpath(rgb_path, IMAGES_ROOT),
                    "json_path": _relpath(os.path.abspath(TRAIN_JSON), DATA_ROOT),
                    "category_id": np.int64(cat_id),
                    "category_name": cat_name,
                    "num_instances": np.int32(len(encoded)),
                },
            }


def sanity_check(
    ds_dir: str,
    split: str,
    num_examples: int = 4,
    dataset_name: str = "lvis_instance_category_sharded",
):
    """Visualize RGB, OR-merged mask, and overlay for a specific category shard split."""
    ds = tfds.load(dataset_name, split=split, data_dir=ds_dir, shuffle_files=False).take(num_examples)
    rows = [ex for ex in ds]
    if not rows:
        print("sanity_check: no examples.")
        return

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, ex in enumerate(rows):
        img = np.asarray(ex["image"].numpy(), dtype=np.uint8)
        merged = np.asarray(ex["merged_mask_or"].numpy(), dtype=np.uint8)[..., 0]
        overlay = img.copy().astype(np.float32)
        overlay[merged > 0] = overlay[merged > 0] * 0.55 + np.array([255, 64, 64], dtype=np.float32) * 0.45
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        cid = int(ex["metadata"]["category_id"].numpy())
        cname = ex["metadata"]["category_name"].numpy()
        if isinstance(cname, (bytes, np.bytes_)):
            cname = cname.decode("utf-8", errors="replace")
        ninst = int(ex["metadata"]["num_instances"].numpy())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"image  cat={cid} ({cname})  n={ninst}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(merged, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("OR merged mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"{dataset_name}_sanity_{stamp}_{split}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
