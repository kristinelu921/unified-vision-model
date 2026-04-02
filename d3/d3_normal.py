"""
TFDS builder for D3-Predictor **surface normal** training data (Hypersim).

  dataset_dir/
    rgb/           # Any file per sample: usual images (``.png`` / ``.jpg`` / …) or ``.h5`` with an RGB tensor (see code for keys)
    normal/        # <stem>.h5  dataset key ``normal`` (or ``normals``)  float (H,W,3) — encoded to uint8 RGB for TFDS (PNG-style)

Load::

  tfds.load("d3_normal", data_dir="/path/to/tfds_output", split="train")
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage


_DATASET_ROOT = os.environ.get(
    "D3_NORMAL_DATASET_ROOT",
    #"/mnt/klu/d3_normal/Hypersim_normal_30_k/"
    "gs://kmh-gcp-us-central2/kristine/lvm/d3_normal/Hypersim_normal_30_k/",
)


def _dataset_type_from_basename(name: str) -> str | None:
    """Classify sample; anything that is not VKITTI/COCO is labeled hypersim."""
    if not name or name.endswith("/") or name.startswith("."):
        return None
    lower = name.lower()
    if lower.endswith((".jpg", ".jpeg")) and name.startswith("Scene"):
        return "vkitti"
    if lower.endswith((".jpg", ".jpeg")) and name.startswith("coco"):
        return "coco"
    return "hypersim"


def _iter_rgb_filenames(rgb_dir: str) -> list[tuple[str, str]]:
    """Returns list of (filename, dataset_type) for regular files under rgb/."""
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(rgb_dir)):
        full = os.path.join(rgb_dir, name)
        if not os.path.isfile(full):
            continue
        dt = _dataset_type_from_basename(name)
        if dt is not None:
            out.append((name, dt))
    return out


def _stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    """Return ``(bucket, object_prefix)`` where ``object_prefix`` ends with ``/`` or is empty."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {uri!r}")
    rest = uri[5:].lstrip("/")
    slash = rest.find("/")
    if slash == -1:
        return rest, ""
    bucket = rest[:slash]
    prefix = rest[slash + 1 :]
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    return bucket, prefix


def _read_normal_array(h5: h5py.File) -> np.ndarray:
    return np.asarray(h5["depth"][:], dtype=np.float32)


def _normals_float_to_png_rgb(n: np.ndarray) -> np.ndarray:
    """
    Encode float surface normals (H,W,3) as uint8 RGB for storage as an image / PNG.

    If values look like signed normals in ~[-1, 1], uses (n * 0.5 + 0.5) * 255.
    Otherwise min-max normalizes per array to [0, 255].
    """
    x = np.asarray(n, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected HxWx3 normals, got {x.shape}")
    max_v, min_v = np.nanmax(x), np.nanmin(x)
    if max_v <= 1.0 + 1e-3 and min_v >= -1.0 - 1e-3:
        vis = np.clip(x * 0.5 + 0.5, 0.0, 1.0)
    else:
        vis = (x - min_v) / (max_v - min_v + 1e-8)
    return np.clip(vis * 255.0, 0, 255).astype(np.uint8)


def _to_uint8_hw3(arr: np.ndarray) -> np.ndarray:
    """(H,W,3) or (H,W) float or uint → uint8 RGB."""
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3:
        if x.shape[2] > 3:
            x = x[..., :3]
    else:
        raise ValueError(f"expected HxWx3 image array, got shape {x.shape}")
    if np.nanmax(x) <= 1.0 + 1e-3:
        x = np.clip(x * 255.0, 0.0, 255.0)
    else:
        x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _read_rgb_image_from_h5(h5: h5py.File) -> np.ndarray:
    """Load a condition RGB image from an HDF5 (Hypersim-style packs often store rgb in h5)."""
    for key in ("rgb", "image", "color", "img", "condition", "albedo"):
        if key in h5:
            return _to_uint8_hw3(np.asarray(h5[key][:]))
    for key in h5.keys():
        ds = h5[key]
        if not hasattr(ds, "shape"):
            continue
        arr = np.asarray(ds[:])
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return _to_uint8_hw3(arr[..., :3])
    raise KeyError("no RGB-like dataset found in HDF5 (tried rgb, image, color, …)")


def _load_rgb_array_from_bytes(data: bytes, name_hint: str = "") -> tuple[np.ndarray, tuple[int, int]]:
    if name_hint.lower().endswith((".h5", ".hdf5")):
        with h5py.File(io.BytesIO(data), "r") as f:
            rgb_arr = _read_rgb_image_from_h5(f)
        h, w = rgb_arr.shape[0], rgb_arr.shape[1]
        return rgb_arr, (w, h)
    rgb_arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    h, w = rgb_arr.shape[0], rgb_arr.shape[1]
    return rgb_arr, (w, h)


class D3Normal(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "D3-Predictor surface-normal training pairs: RGB condition image and normal targets "
                "stored as uint8 RGB (PNG-style encoding from float normals in normal/*.h5)."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3), doc="RGB condition image."),
                    "mask": tfds.features.Image(
                        shape=(None, None, 3),
                        doc=(
                            "Normals as uint8 RGB (decoded from HDF5 float normals: "
                            "signed [-1,1] mapped with (n+1)/2 then ×255)."
                        ),
                    ),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "stem": tfds.features.Text(),
                            "dataset_type": tfds.features.Text(),
                            "rgb_path": tfds.features.Text(),
                            "normal_path": tfds.features.Text(),
                            "image_size": tfds.features.Tensor(shape=(2,), dtype=np.int32),
                            "index": tfds.features.Tensor(shape=(1,), dtype=np.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "normal"),
            homepage="https://huggingface.co/datasets/X-GenGroup/D3-Predictor-Data-Normal",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        root = _DATASET_ROOT
        return {
            "train": self._generate_examples(root, split="train"),
            "validation": self._generate_examples(root, split="validation"),
        }

    def _generate_examples(self, dataset_dir: str, split: str):
        if dataset_dir.startswith("gs://"):
            yield from self._generate_examples_gcs(dataset_dir, split)
            return

        if os.path.isfile(dataset_dir):
            raise ValueError(
                "Dataset root must be a gs:// URI or a directory containing rgb/ and normal/ "
                f"(got a file path): {dataset_dir!r}"
            )

        rgb_dir = os.path.join(dataset_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")

        yielded = 0
        for index, (rgb_file, dataset_type) in enumerate(_iter_rgb_filenames(rgb_dir)):
            if split == "validation":
                if index % 10 != 0:
                    continue
            elif split == "train":
                if index % 10 == 0:
                    continue
            else:
                raise ValueError(f"Invalid split: {split}")

            stem = _stem(rgb_file)
            rgb_path = os.path.join(rgb_dir, rgb_file)
            normal_path = os.path.join(dataset_dir, "normal", stem + ".h5")

            if not os.path.isfile(normal_path):
                continue

            try:
                with h5py.File(normal_path, "r") as f:
                    normal_arr = _read_normal_array(f)
            except (OSError, KeyError):
                continue

            if normal_arr.ndim != 3 or normal_arr.shape[2] != 3:
                continue

            try:
                normal_png = _normals_float_to_png_rgb(normal_arr)
            except ValueError:
                continue

            key = f"{dataset_type}_{stem}"
            if rgb_path.lower().endswith((".h5", ".hdf5")):
                try:
                    with h5py.File(rgb_path, "r") as rf:
                        rgb_arr = _read_rgb_image_from_h5(rf)
                except (OSError, KeyError, ValueError):
                    continue
                w, h = rgb_arr.shape[1], rgb_arr.shape[0]
                image_field: np.ndarray | str = rgb_arr
            else:
                try:
                    with Image.open(rgb_path) as im:
                        w, h = im.size
                except OSError:
                    continue
                image_field = rgb_path

            yielded += 1
            yield key, {
                "image": image_field,
                "mask": normal_png,
                "metadata": {
                    "stem": stem,
                    "dataset_type": dataset_type,
                    "rgb_path": rgb_path,
                    "normal_path": normal_path,
                    "image_size": np.array([w, h], dtype=np.int32),
                    "index": np.array([index], dtype=np.int32),
                },
            }

    def _generate_examples_gcs(self, gs_uri: str, split: str):
        bucket_name, prefix = _parse_gs_uri(gs_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        rgb_prefix = prefix + "rgb/"

        entries: list[tuple[str, str, str]] = []
        for blob in bucket.list_blobs(prefix=rgb_prefix):
            if blob.name.endswith("/"):
                continue
            base = os.path.basename(blob.name)
            dt = _dataset_type_from_basename(base)
            if dt is None:
                continue
            entries.append((blob.name, base, dt))
        entries.sort(key=lambda x: x[1])

        yielded = 0
        for index, (rgb_blob_name, rgb_file, dataset_type) in enumerate(entries):
            if index % 10 == 0:
                print(f"loaded {index} entries")
            if max_yields is not None and yielded >= max_yields:
                break

            if split == "validation":
                if index % 10 != 0:
                    continue
            elif split == "train":
                if index % 10 == 0:
                    continue
            else:
                raise ValueError(f"Invalid split: {split}")

            stem = _stem(rgb_file)
            normal_name = prefix + "normal/" + stem + ".h5"

            try:
                normal_blob = bucket.blob(normal_name)
                normal_bytes = normal_blob.download_as_bytes()
            except gcp_exceptions.NotFound:
                raise ValueError(f"Normal file not found: {normal_name}")

            try:
                with h5py.File(io.BytesIO(normal_bytes), "r") as f:
                    normal_arr = _read_normal_array(f)
            except (KeyError, OSError):
                raise ValueError(f"Error reading normal file: {normal_name}")

            if normal_arr.ndim != 3 or normal_arr.shape[2] != 3:
                raise ValueError(f"Invalid normal array shape: {normal_arr.shape}")

            try:
                normal_png = _normals_float_to_png_rgb(normal_arr)
            except ValueError:
                raise ValueError(f"Error encoding normal array to uint8 PNG: {normal_arr.shape}")

            try:
                rgb_blob = bucket.blob(rgb_blob_name)
                rgb_bytes = rgb_blob.download_as_bytes()
            except gcp_exceptions.NotFound:
                raise ValueError(f"RGB file not found: {rgb_blob_name}")

            try:
                rgb_arr, _ = _load_rgb_array_from_bytes(rgb_bytes, rgb_file)
            except (OSError, KeyError, ValueError):
                raise ValueError(f"Error loading RGB array: {rgb_blob_name}")
            key = f"{dataset_type}_{stem}"
            loc = f"gs://{bucket_name}/{rgb_blob_name}"
            yielded += 1
            yield key, {
                "image": rgb_arr,
                "mask": normal_png,
                "metadata": {
                    "stem": stem,
                    "dataset_type": dataset_type,
                    "rgb_path": loc,
                    "normal_path": f"gs://{bucket_name}/{normal_name}",
                    "image_size": np.array(
                        [rgb_arr.shape[1], rgb_arr.shape[0]],
                        dtype=np.int32,
                    ),
                    "index": np.array([index], dtype=np.int32),
                },
            }

def resize_random_crop(ex, size=(256, 256)):
    target_h, target_w = size
    long_side = 512.0

    image = tf.cast(ex["image"], tf.uint8)
    mask = tf.cast(ex["mask"], tf.float32)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    scale = long_side / tf.cast(tf.maximum(h, w), tf.float32)
    new_h = tf.maximum(target_h, tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32))
    new_w = tf.maximum(target_w, tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32))

    image_r = tf.image.resize(image, [new_h, new_w], method="bilinear", antialias=True)
    mask_r = tf.image.resize(mask, [new_h, new_w], method="bilinear", antialias=True)

    merged = tf.concat([image_r, mask_r], axis=-1)
    # 3 (RGB) + 3 (normal PNG-style), not 7 — d3_depth uses an extra depth channel.
    cropped = tf.image.random_crop(merged, size=[target_h, target_w, 6])

    ex["image"] = tf.cast(tf.clip_by_value(cropped[..., :3], 0.0, 255.0), tf.uint8)
    ex["mask"] = cropped[..., 3:6]
    ex["metadata"]["image_size"] = tf.cast([target_w, target_h], tf.int32)
    return ex


def load_data(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = False,
    shuffle_buffer: int = 256,
):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("d3_normal", **load_kw)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    #ds = ds.map(resize_random_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def sanity_check(
    tfds_data_dir: str,
    split: str = "train",
    num_batches: int = 1,
    batch_size: int = 1,
    max_show: int = 8,
):
    ds = tfds.load("d3_normal", split=split, data_dir=tfds_data_dir)
    ds = ds.map(resize_random_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        mask = ex["mask"].numpy()
        stems = ex["metadata"]["stem"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            n = mask[j]
            # Already uint8 PNG-style RGB from the builder.
            vis = np.clip(n.astype(np.float32) / 255.0, 0.0, 1.0)
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            sj = stems[j]
            if hasattr(sj, "numpy"):
                sj = sj.numpy()
            if isinstance(sj, (bytes, np.bytes_)):
                title = sj.decode("utf-8", errors="replace")
            else:
                title = str(sj)
            plt.title(title[:80])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(vis)
            plt.axis("off")
        plt.tight_layout()
        out = f"d3_normal_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    # See module docstring for build instructions.
    pass
