"""
TFDS builder for EntitySeg **semantic segmentation**: one RGB + **one dense label map** per image.

Lists come from ``entityseg_semseg_train.txt`` / ``entityseg_semseg_val.txt`` (image filename, mask
PNG filename per line). Masks are grayscale PNGs under ``ENTITYSEG_SEMSEG_MAPS_{TRAIN,VAL}_DIR``.
Rows whose RGB path does not exist or cannot be opened are **skipped**.

For instance segmentation (RLE, one row per object), use ``entity_seg_instance.py``.

  tfds build entityseg/entity_seg_semantic.py --data_dir=/path/to/tfds_output
  tfds.load("entity_seg_semantic", data_dir=..., split="train")
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


DATA_ROOT = os.environ.get("ENTITYSEG_DATA_ROOT", "/mnt/klum/entityseg/data")
IMAGES_ROOT = os.environ.get(
    "ENTITYSEG_IMAGES_ROOT",
    os.path.join(DATA_ROOT, "entityseg_test_images_dl"),
)

SEMSEG_TRAIN_TXT = os.environ.get(
    "ENTITYSEG_SEMSEG_TRAIN_TXT",
    os.path.join(DATA_ROOT, "entityseg_semseg_train.txt"),
)
SEMSEG_VAL_TXT = os.environ.get(
    "ENTITYSEG_SEMSEG_VAL_TXT",
    os.path.join(DATA_ROOT, "entityseg_semseg_val.txt"),
)

SEMSEG_TRAIN_DIR = os.environ.get(
    "ENTITYSEG_SEMSEG_MAPS_TRAIN_DIR",
    os.path.join(DATA_ROOT, "entityseg_semseg_maps_train", "semantic_maps_train"),
)
SEMSEG_VAL_DIR = os.environ.get(
    "ENTITYSEG_SEMSEG_MAPS_VAL_DIR",
    os.path.join(DATA_ROOT, "entityseg_semseg_maps_val", "semantic_maps_val"),
)

_SEMSEG_DIR = {"train": SEMSEG_TRAIN_DIR, "validation": SEMSEG_VAL_DIR}
_SEMSEG_TXT = {"train": SEMSEG_TRAIN_TXT, "validation": SEMSEG_VAL_TXT}

_MAX_EXAMPLES = os.environ.get("ENTITYSEG_MAX_EXAMPLES_PER_SPLIT", "").strip()
MAX_EXAMPLES_PER_SPLIT = int(_MAX_EXAMPLES) if _MAX_EXAMPLES else None

REQUIRE_SEMANTIC_MAP = os.environ.get("ENTITYSEG_REQUIRE_SEMANTIC_MAP", "").strip() not in (
    "0",
    "false",
    "no",
)


def _relpath(path: str, root: str) -> str:
    return os.path.relpath(path, start=root).replace(os.sep, "/")


def _parse_semseg_txt(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


@lru_cache(maxsize=1)
def _image_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not os.path.isdir(IMAGES_ROOT):
        return index
    for dirpath, _dirnames, filenames in os.walk(IMAGES_ROOT):
        for filename in filenames:
            index.setdefault(filename, os.path.join(dirpath, filename))
    return index


def _resolve_image_path(img_fn: str) -> str | None:
    direct = os.path.join(IMAGES_ROOT, img_fn)
    if os.path.isfile(direct):
        return direct
    return _image_index().get(os.path.basename(img_fn))


class EntitySegSemantic(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.1")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="EntitySeg semantic segmentation: RGB + one per-pixel class map (PNG) per image.",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "mask": tfds.features.Tensor(
                        shape=(None, None, 3),
                        dtype=np.uint8,
                        encoding="zlib",
                        doc="Class id in channel 0; channels 1–2 are zero.",
                    ),
                    "mask_valid": tfds.features.Tensor(shape=(), dtype=np.bool_),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "index": tfds.features.Tensor(shape=(), dtype=np.int64),
                            "file_name": tfds.features.Text(),
                            "mask_file": tfds.features.Text(),
                            "width": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "height": tfds.features.Tensor(shape=(), dtype=np.int32),
                            "rel_image_path": tfds.features.Text(),
                            "list_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "mask"),
            homepage="https://arxiv.org/abs/2211.05776",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples("train"),
            "validation": self._generate_examples("validation"),
        }

    def _generate_examples(self, split_name: str):
        txt_path = _SEMSEG_TXT[split_name]
        sem_dir = _SEMSEG_DIR[split_name]
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Semantic list not found: {txt_path}")
        pairs = _parse_semseg_txt(txt_path)
        n = 0
        for idx, (img_fn, mask_fn) in enumerate(pairs):
            if MAX_EXAMPLES_PER_SPLIT is not None and n >= MAX_EXAMPLES_PER_SPLIT:
                break
            rgb_path = _resolve_image_path(img_fn)
            sem_path = os.path.join(sem_dir, mask_fn)
            if rgb_path is None or not os.path.isfile(rgb_path):
                continue

            mask_arr = np.zeros((1, 1, 3), dtype=np.uint8)
            sem_valid = False
            h = w = 0
            try:
                with Image.open(rgb_path) as im:
                    w, h = im.size
            except OSError:
                continue

            if os.path.isfile(sem_path):
                sem = np.asarray(Image.open(sem_path).convert("L"), dtype=np.uint8)
                if sem.shape == (h, w):
                    mask_arr = np.zeros((h, w, 3), dtype=np.uint8)
                    mask_arr[:, :, 0] = sem
                    sem_valid = True

            if REQUIRE_SEMANTIC_MAP and not sem_valid:
                continue

            key = f"{split_name}_{idx}_{img_fn}"
            n += 1
            yield key, {
                "image": rgb_path,
                "mask": mask_arr,
                "mask_valid": np.bool_(sem_valid),
                "metadata": {
                    "index": np.int64(idx),
                    "file_name": img_fn,
                    "mask_file": mask_fn,
                    "width": np.int32(w),
                    "height": np.int32(h),
                    "rel_image_path": _relpath(rgb_path, IMAGES_ROOT),
                    "list_path": _relpath(os.path.abspath(txt_path), DATA_ROOT),
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
    ds = tfds.load("entity_seg_semantic", **kw)
    ds = ds.map(preprocess_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def preprocess_func(example):
    image = tf.cast(example["image"], tf.float32)
    mask = tf.cast(example["mask"], tf.float32)
    example["image"] = tf.cast(tf.image.resize(image, [256, 256], method="bilinear"), tf.uint8)
    example["mask"] = tf.cast(tf.image.resize(mask, [256, 256], method="nearest"), tf.uint8)
    return example


def sanity_check(ds_dir, split, num_batches=1, batch_size=8, shuffle=False):
    dataloader = load_data(
        data_dir=ds_dir,
        split=split,
        batch_size=batch_size,
        repeat=False,
        shuffle=shuffle,
    )

    for i, batch in enumerate(dataloader.take(num_batches)):
        images = batch["image"]
        masks = batch["mask"]
        metadata = batch["metadata"]
        print("reached sanity check")
        print("images.shape:", images.shape)
        print("masks.shape:", masks.shape)
        print("metadata:", metadata)

        plt.figure(figsize=(10, 12))
        n = min(3, images.shape[0])
        for j in range(n):
            plt.subplot(n, 2, 2 * j + 1)
            plt.imshow(np.clip(np.asarray(images[j]), 0, 255).astype(np.uint8))
            plt.axis("off")

            plt.subplot(n, 2, 2 * j + 2)
            plt.imshow(np.asarray(masks[j])[:, :, 0], cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = f"entity_seg_semantic_sanity_{date_str}_{split}_batch_{i + 1}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    pass
