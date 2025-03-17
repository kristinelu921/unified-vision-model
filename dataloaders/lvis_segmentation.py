# path is gs://kmh-gcp-asia-northeast1-b/kristine/lvm/lvis_giraffe_tfds/

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds

from ._tfds_common import apply_cache, resolve_builder_dir


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"validation", "valid", "val"}:
        return "validation"
    return normalized


def _normalize_crop_size(size: int | Sequence[int] | None) -> tuple[int, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        if size <= 0:
            raise ValueError(f"`crop_size` must be > 0, got {size}.")
        return size, size

    values = tuple(int(v) for v in size)
    if len(values) != 2:
        raise ValueError(f"`crop_size` must be an int or a 2-item sequence, got {size!r}.")
    target_h, target_w = values
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"`crop_size` entries must be > 0, got {size!r}.")
    return target_h, target_w


def _normalize_max_num_pair(max_num_pair: int) -> int:
    value = int(max_num_pair)
    if value <= 0:
        raise ValueError(f"`max_num_pair` must be > 0, got {max_num_pair}.")
    return value


def _prepare_example(example):
    image = tf.cast(example["image"], tf.uint8)
    mask = tf.cast(example["instance"]["mask_raster"], tf.uint8)
    mask = tf.where(mask > 0, 255, 0)
    mask = tf.repeat(mask, repeats=3, axis=-1)

    metadata = dict(example["metadata"])
    metadata["filename"] = tf.cast(metadata.get("file_name", ""), tf.string)
    metadata["image_path"] = tf.cast(metadata.get("rel_image_path", ""), tf.string)
    metadata["image_size"] = tf.cast([metadata["width"], metadata["height"]], tf.int32)
    instance = dict(example["instance"])
    metadata["category_id"] = tf.cast(instance["category_id"], tf.int32)
    metadata["category_name"] = tf.cast(instance.get("category_name", ""), tf.string)
    metadata["bbox"] = tf.cast(instance.get("bbox", tf.zeros((4,), dtype=tf.float32)), tf.float32)
    metadata["area"] = tf.cast(instance.get("area", 0.0), tf.float32)
    metadata["iscrowd"] = tf.cast(instance.get("iscrowd", 0), tf.int32)
    metadata["class_idx"] = tf.cast(instance.get("class_idx", -1), tf.int32)

    return {
        "image": image,
        "mask": mask,
        "metadata": metadata,
    }


def _extract_example(example):
    return (
        example["image"],
        example["mask"],
        example["metadata"],
    )


def random_recolor(example):
    """maps semantic label ids to deterministic random RGB colors."""
    label_ids = tf.cast(example["mask"][..., 0], tf.int32)
    red = tf.math.floormod(label_ids * 37 + 17, 256)
    green = tf.math.floormod(label_ids * 67 + 59, 256)
    blue = tf.math.floormod(label_ids * 97 + 101, 256)
    recolored = tf.stack([red, green, blue], axis=-1)

    # keep background label 0 black.
    recolored = tf.where(label_ids[..., tf.newaxis] == 0, 0, recolored)

    recolored_example = dict(example)
    recolored_example["mask"] = tf.cast(recolored, tf.uint8)
    return recolored_example


def resize_random_crop(
    example,
    size: int | Sequence[int] = (256, 256),
    *,
    long_side: int = 512,
):
    target_h, target_w = _normalize_crop_size(size)
    if long_side <= 0:
        raise ValueError(f"`long_side` must be > 0, got {long_side}.")

    image = tf.cast(example["image"], tf.uint8)
    mask = tf.cast(example["mask"], tf.uint8)

    height = tf.shape(image)[0]
    width = tf.shape(image)[1]
    scale = tf.cast(long_side, tf.float32) / tf.cast(tf.maximum(height, width), tf.float32)
    new_h = tf.maximum(target_h, tf.cast(tf.round(tf.cast(height, tf.float32) * scale), tf.int32))
    new_w = tf.maximum(target_w, tf.cast(tf.round(tf.cast(width, tf.float32) * scale), tf.int32))

    image_resized = tf.image.resize(
        tf.cast(image, tf.float32),
        [new_h, new_w],
        method="bilinear",
        antialias=True,
    )
    mask_resized = tf.image.resize(
        tf.cast(mask, tf.float32),
        [new_h, new_w],
        method="nearest",
    )

    merged = tf.concat([image_resized, mask_resized], axis=-1)
    cropped = tf.image.random_crop(merged, size=[target_h, target_w, 6])

    metadata = dict(example["metadata"])
    metadata["image_size"] = tf.cast([target_w, target_h], tf.int32)

    resized_example = dict(example)
    resized_example["image"] = tf.cast(tf.clip_by_value(cropped[..., :3], 0.0, 255.0), tf.uint8)
    resized_example["mask"] = tf.cast(tf.clip_by_value(cropped[..., 3:6], 0.0, 255.0), tf.uint8)
    resized_example["metadata"] = metadata
    return resized_example


def load_data(
    split: str = "train",
    data_dir: str | Path | None = None,
    batch_size: int = 64,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer_size: int = 1000,
    cache: bool | str | Path | None = False,
    crop_size: int | Sequence[int] | None = None,
    resize_long_side: int = 512,
    max_num_pair: int = 1,
    random_recolor_masks: bool = False,
):
    if data_dir is None:
        raise ValueError("`data_dir` is required for entityseg_segmentation.load_data.")
    normalized_max_num_pair = _normalize_max_num_pair(max_num_pair)

    builder_dir = resolve_builder_dir(data_dir)
    builder = tfds.builder_from_directory(builder_dir)

    resolved_split = _normalize_split(split)
    if resolved_split not in builder.info.splits:
        available = ", ".join(sorted(builder.info.splits.keys()))
        raise ValueError(f"Unknown split {split!r}. Available splits: {available}")

    ds = builder.as_dataset(split=resolved_split, shuffle_files=shuffle)
    ds = ds.map(_prepare_example, num_parallel_calls=tf.data.AUTOTUNE)
    if crop_size is not None:
        normalized_crop_size = _normalize_crop_size(crop_size)
        ds = ds.map(
            lambda ex: resize_random_crop(ex, size=normalized_crop_size, long_side=resize_long_side),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if random_recolor_masks:
        ds = ds.map(random_recolor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_extract_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    ds = apply_cache(ds, cache)
    if repeat:
        ds = ds.repeat()
    if normalized_max_num_pair > 1:
        ds = ds.batch(normalized_max_num_pair, drop_remainder=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def inspect_one_batch(
    split: str = "train",
    data_dir: str | Path | None = None,
    batch_size: int = 2,
    max_num_pair: int = 1,
    crop_size: int | Sequence[int] | None = 256,
    resize_long_side: int = 512,
    random_recolor_masks: bool = False,
):
    ds = load_data(
        split=split,
        data_dir=data_dir,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        max_num_pair=max_num_pair,
        crop_size=crop_size,
        resize_long_side=resize_long_side,
        random_recolor_masks=random_recolor_masks,
    )
    for images, masks, metadata in ds.take(1):
        return images, masks, metadata
    raise ValueError(f"Split {split!r} is empty.")
