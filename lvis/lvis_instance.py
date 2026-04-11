"""
TFDS builder for LVIS v1 — **one TFDS example per annotation** (instance).

Same paths / env vars as ``lvis.py`` (``LVIS_IMAGES_ROOT``, ``LVIS_*_JSON``).
``LVIS_MAX_EXAMPLES_PER_SPLIT`` caps **annotation** rows yielded (not images).
Decodes each annotation's segmentation to ``instance/mask_raster`` at build time;
**pycocotools** is required to run ``tfds build``.

  tfds build lvis/lvis_instance.py --data_dir=/path/to/tfds_output
  tfds.load("lvis_instance", data_dir=..., split="train")
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

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


class LvisInstance(tfds.core.GeneratorBasedBuilder):
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
                "LVIS v1: **one row per instance**. Full image RGB + single-instance "
                "bbox and ``mask_raster`` (H×W×1, same resolution as the image)."
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
        random.Random(_SAMPLE_SEED).shuffle(anns)

        n = 0
        for i, ann in enumerate(anns):
            if i % 3 != 0:
                continue
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
    ds = tfds.load("lvis_instance", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if ignore_errors:
        ds = ds.ignore_errors()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _index_cache_path(
    *,
    split: str,
    data_dir: str | None,
    dataset_name: str = "lvis_instance",
    cache_dir: str | None = None,
) -> str:
    base = cache_dir or data_dir or os.path.join(str(Path.home()), ".cache", "lvis_instance")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{dataset_name}_category_index_{split}.json")


def build_category_index_cache(
    split: str = "train",
    data_dir: str | None = None,
    dataset_name: str = "lvis_instance",
    force_rebuild: bool = False,
    cache_dir: str | None = None,
) -> dict[int, list[int]]:
    """Build/load category->example_indices cache for fast random-class sampling."""
    cache_path = _index_cache_path(
        split=split,
        data_dir=data_dir,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    if os.path.isfile(cache_path) and not force_rebuild:
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {
            int(cat_id): [int(i) for i in indices]
            for cat_id, indices in raw["category_to_indices"].items()
        }

    kw: dict = {"split": split, "shuffle_files": False}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load(dataset_name, **kw)

    category_to_indices: dict[int, list[int]] = {}
    category_to_image_indices: dict[int, dict[int, list[int]]] = {}
    num_examples = 0
    for idx, ex in enumerate(ds):
        cat = int(ex["instance"]["category_id"].numpy())
        image_id = int(ex["metadata"]["image_id"].numpy())
        category_to_indices.setdefault(cat, []).append(int(idx))
        category_to_image_indices.setdefault(cat, {}).setdefault(image_id, []).append(int(idx))
        num_examples += 1

    payload = {
        "dataset_name": dataset_name,
        "split": split,
        "num_examples_scanned": num_examples,
        "num_categories": len(category_to_indices),
        "category_to_indices": {str(k): v for k, v in category_to_indices.items()},
        "category_to_image_indices": {
            str(cat): {str(img_id): idxs for img_id, idxs in by_img.items()}
            for cat, by_img in category_to_image_indices.items()
        },
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return category_to_indices


def _load_category_index_payload(
    split: str = "train",
    data_dir: str | None = None,
    dataset_name: str = "lvis_instance",
    cache_dir: str | None = None,
) -> dict:
    cache_path = _index_cache_path(
        split=split,
        data_dir=data_dir,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    if not os.path.isfile(cache_path):
        build_category_index_cache(
            split=split,
            data_dir=data_dir,
            dataset_name=dataset_name,
            force_rebuild=False,
            cache_dir=cache_dir,
        )
    with open(cache_path, encoding="utf-8") as f:
        raw = json.load(f)

    # Backward compatibility: old cache files may miss category_to_image_indices.
    if "category_to_image_indices" not in raw:
        build_category_index_cache(
            split=split,
            data_dir=data_dir,
            dataset_name=dataset_name,
            force_rebuild=True,
            cache_dir=cache_dir,
        )
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
    return raw


def _choose_random_category_id_from_index(
    category_to_indices: dict[int, list[int]],
    *,
    seed: int | None = None,
) -> int:
    valid_cats = sorted(int(k) for k, v in category_to_indices.items() if v)
    if not valid_cats:
        raise ValueError("No category IDs found in category index cache.")
    return random.Random(seed).choice(valid_cats)


def load_random_class_data(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
    ignore_errors: bool = True,
    category_id: int | None = None,
    seed: int | None = None,
    max_scan_examples: int | None = None,
    cache_dir: str | None = None,
    force_rebuild_index: bool = False,
):
    """Return dataset filtered to one class; picks random class when ``category_id`` is None.
    This is designed for class-conditional instance segmentation batches where
    the annotation target is ``instance/category_id``.
    """
    category_to_indices = build_category_index_cache(
        split=split,
        data_dir=data_dir,
        force_rebuild=force_rebuild_index,
        cache_dir=cache_dir,
    )
    chosen_category_id = int(category_id) if category_id is not None else _choose_random_category_id_from_index(
        category_to_indices, seed=seed
    )

    kw: dict = {"split": split}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    base_ds = tfds.load("lvis_instance", **kw)
    if ignore_errors:
        base_ds = base_ds.ignore_errors()

    ds = base_ds.filter(
        lambda ex: tf.equal(
            tf.cast(ex["instance"]["category_id"], tf.int64),
            tf.cast(chosen_category_id, tf.int64),
        )
    )
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, chosen_category_id


def sample_random_class_batch(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 4,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
    ignore_errors: bool = True,
    category_id: int | None = None,
    seed: int | None = None,
    max_scan_examples: int | None = None,
    cache_dir: str | None = None,
    force_rebuild_index: bool = False,
):
    """Sample random-class batch where sampled images include all class instances."""
    payload = _load_category_index_payload(
        split=split,
        data_dir=data_dir,
        cache_dir=cache_dir,
    )
    category_to_indices = {
        int(cat_id): [int(i) for i in idxs]
        for cat_id, idxs in payload["category_to_indices"].items()
    }
    chosen_category_id = (
        int(category_id)
        if category_id is not None
        else _choose_random_category_id_from_index(category_to_indices, seed=seed)
    )
    by_img_raw = payload["category_to_image_indices"].get(str(chosen_category_id), {})
    by_img = {
        int(img_id): [int(i) for i in idxs]
        for img_id, idxs in by_img_raw.items()
    }
    if not by_img:
        raise ValueError(
            f"No cached image->indices found for category_id={chosen_category_id} in split={split!r}."
        )

    rng = random.Random(seed)
    image_ids = list(by_img.keys())
    if len(image_ids) >= batch_size:
        picked_images = rng.sample(image_ids, k=batch_size)
    else:
        picked_images = image_ids
    selected_indices = sorted({i for img_id in picked_images for i in by_img[img_id]})

    kw: dict = {"split": split, "shuffle_files": False}
    if data_dir is not None:
        kw["data_dir"] = data_dir
    ds = tfds.load("lvis_instance", **kw)
    if ignore_errors:
        ds = ds.ignore_errors()

    wanted = set(selected_indices)
    collected = []
    for idx, ex in enumerate(ds):
        if idx in wanted:
            collected.append(preprocess_func(ex))
            if len(collected) == len(selected_indices):
                break
    if not collected:
        raise ValueError(
            f"No examples found for category_id={chosen_category_id} in split={split!r}."
        )

    # Optional shuffle of the collected annotation pairs.
    if shuffle:
        rng.shuffle(collected)

    batch = {
        "image": tf.stack([ex["image"] for ex in collected], axis=0),
        "instance": {
            "bbox": tf.stack([ex["instance"]["bbox"] for ex in collected], axis=0),
            "area": tf.stack([ex["instance"]["area"] for ex in collected], axis=0),
            "category_id": tf.stack([ex["instance"]["category_id"] for ex in collected], axis=0),
            "class_idx": tf.stack([ex["instance"]["class_idx"] for ex in collected], axis=0),
            "category_name": tf.stack([ex["instance"]["category_name"] for ex in collected], axis=0),
            "iscrowd": tf.stack([ex["instance"]["iscrowd"] for ex in collected], axis=0),
            "mask_raster": tf.stack([ex["instance"]["mask_raster"] for ex in collected], axis=0),
        },
        "metadata": {
            "image_id": tf.stack([ex["metadata"]["image_id"] for ex in collected], axis=0),
            "annotation_id": tf.stack([ex["metadata"]["annotation_id"] for ex in collected], axis=0),
            "file_name": tf.stack([ex["metadata"]["file_name"] for ex in collected], axis=0),
            "width": tf.stack([ex["metadata"]["width"] for ex in collected], axis=0),
            "height": tf.stack([ex["metadata"]["height"] for ex in collected], axis=0),
            "rel_image_path": tf.stack([ex["metadata"]["rel_image_path"] for ex in collected], axis=0),
            "json_path": tf.stack([ex["metadata"]["json_path"] for ex in collected], axis=0),
        },
    }
    return batch, chosen_category_id
    raise ValueError(
        f"No examples found for category_id={chosen_category_id} in split={split!r}."
    )


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
    num_batches=5,
    batch_size=4,
    shuffle=False,
    max_show=4,
    dataset_name: str = "lvis_instance",
):
    """Visualize random class batches via ``sample_random_class_batch``.

    Each iteration picks one random category_id and returns a random batch of
    instance image-annotation pairs from that class.
    """
    for bi in range(num_batches):
        batch, chosen_cat = sample_random_class_batch(
            split=split,
            data_dir=ds_dir,
            batch_size=batch_size,
            shuffle_buffer=256,
            ignore_errors=True,
            category_id=None,
            seed=None,
            max_scan_examples=None,
        )

        actual_bs = int(batch["image"].shape[0])
        if actual_bs == 0:
            continue

        n = min(max_show, actual_bs)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for j in range(n):
            img = np.clip(batch["image"][j].numpy(), 0, 255).astype(np.uint8)
            h = int(batch["metadata"]["height"][j].numpy())
            w = int(batch["metadata"]["width"][j].numpy())
            image_id = int(batch["metadata"]["image_id"][j].numpy())
            ann_id = int(batch["metadata"]["annotation_id"][j].numpy())
            mk = batch["instance"]["mask_raster"][j].numpy()
            cat = int(batch["instance"]["category_id"][j].numpy())
            nm = batch["instance"]["category_name"][j].numpy()
            if isinstance(nm, (bytes, np.bytes_)):
                nm = nm.decode("utf-8", errors="replace")

            axes[j, 0].imshow(img)
            axes[j, 0].set_title(f"image  img_id={image_id} ann_id={ann_id}  {h}×{w}")
            axes[j, 0].axis("off")

            axes[j, 1].imshow(mk[..., 0], cmap="gray", vmin=0, vmax=1)
            axes[j, 1].set_title(f"mask_raster  cat={cat}")
            axes[j, 1].axis("off")

            colored = _overlay_mask_on_rgb(img, mk)
            axes[j, 2].imshow(colored)
            ax_ov = axes[j, 2]
            x, y, bw, bh = [float(t) for t in batch["instance"]["bbox"][j].numpy()]
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
            ax_ov.text(
                x,
                max(y - 2, 0),
                nm[:28],
                color="yellow",
                fontsize=6,
                clip_on=True,
            )
            axes[j, 2].set_title("overlay + bbox")
            axes[j, 2].axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = (
            f"{dataset_name}_sanity_{date_str}_{split}_batch_{bi + 1}_cat_{chosen_cat}.png"
        )
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    pass
