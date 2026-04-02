"""
TFDS builder for D3-Predictor depth training data (Hypersim / VKITTI / COCO mix).

Expected layout under ``dataset_dir`` (same as X-GenGroup D3-Predictor ``dataloader.py``):

  dataset_dir/
    rgb/           # condition images (.png ai*, .jpg Scene*, .jpg coco*)
    depth/         # <stem>.h5  dataset key 'depth'  float (H,W,3) in [0,1]
    depth_ori/     # <stem>.h5  dataset key 'depth'  float (H,W) metric depth
    <stem>.json    # {"caption": "..."}

Set env ``D3_DEPTH_DATASET_ROOT`` to a ``gs://bucket/prefix/`` URI with that layout in Cloud
Storage, or a local directory with the same folder structure.

Build TFDS from repo root::

  pip install tensorflow tensorflow-datasets h5py pillow matplotlib google-cloud-storage
  export D3_DEPTH_DATASET_ROOT=gs://kmh-gcp-us-central2/kristine/lvm/d3_depth/train_15K_plus_15K/
  # or a local directory: /path/to/train_15K_plus_15K
  # Optional small build: export D3_MAX_EXAMPLES_PER_SPLIT=50
  # (train + validation each get up to N examples; unset = full data)
  # GCS only: export D3_MAX_RGB_LIST=5000  # cap rgb listing during dev
  tfds build d3/d3_depth.py --data_dir=/path/to/tfds_output

Load::

  tfds.load("d3_depth", data_dir="/path/to/tfds_output", split="train")
"""

from __future__ import annotations

import io
import json
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
    "D3_DEPTH_DATASET_ROOT",
    "/mnt/klu/d3_depth_2/train_15K_plus_15K/",
    #"gs://kmh-gcp-us-central2/kristine/lvm/d3_depth_2/train_15K_plus_15K/",
)


def _max_examples_per_split() -> int | None:
    """If set (e.g. ``export D3_MAX_EXAMPLES_PER_SPLIT=100``), stop after this many *yields* per split."""
    raw = os.environ.get("D3_MAX_EXAMPLES_PER_SPLIT", "").strip()
    if not raw:
        return None
    return max(0, int(raw))


def _max_rgb_list() -> int | None:
    """Cap how many rgb blobs we list from GCS (saves listing huge buckets during dev)."""
    raw = os.environ.get("D3_MAX_RGB_LIST", "").strip()
    raw = 50;
    if not raw:
        return None
    return max(0, int(raw))


def _dataset_type_from_basename(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".png") and name.startswith("ai"):
        return "hypersim"
    if lower.endswith(".jpg") and name.startswith("Scene"):
        return "vkitti"
    if lower.endswith(".jpg") and name.startswith("coco"):
        return "coco"
    return None


def _iter_rgb_filenames(rgb_dir: str) -> list[tuple[str, str]]:
    """Returns list of (filename, dataset_type)."""
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(rgb_dir)):
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


class D3Depth(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.5")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "D3-Predictor depth training pairs: RGB condition, HDF5 depth targets, "
                "caption JSON (Marigold-style layout)."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3), doc="RGB condition image."),
                    "mask": tfds.features.Tensor(
                        shape=(None, None, 3),
                        dtype=np.float32,
                        encoding="zlib",
                        doc="3-channel depth in [0, 1] (from depth/*.h5).",
                    ),
                    "depth_metric": tfds.features.Tensor(
                        shape=(None, None, 1),
                        dtype=np.float32,
                        encoding="zlib",
                        doc="Single-channel depth from depth_ori/*.h5.",
                    ),
                    "caption": tfds.features.Text(),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "stem": tfds.features.Text(),
                            "dataset_type": tfds.features.Text(),
                            "rgb_path": tfds.features.Text(),
                            "depth_path": tfds.features.Text(),
                            "depth_ori_path": tfds.features.Text(),
                            "json_path": tfds.features.Text(),
                            "image_size": tfds.features.Tensor(shape=(2,), dtype=tf.int32),
                            "index": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "mask"),
            homepage="https://huggingface.co/datasets/X-GenGroup/D3-Predictor-Data-Depth",
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
                "Dataset root must be a gs:// URI or a directory containing rgb/, depth/, depth_ori/, "
                f"and *.json files (got a file path): {dataset_dir!r}"
            )

        rgb_dir = os.path.join(dataset_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")

        max_yields = _max_examples_per_split()
        yielded = 0
        for index, (rgb_file, dataset_type) in enumerate(_iter_rgb_filenames(rgb_dir)):
            if max_yields is not None and yielded >= max_yields:
                break
            if split == "validation":
                if index % 50 != 0:
                    continue
            elif split == "train":
                if index % 50 == 0:
                    continue
            else:
                raise ValueError(f"Invalid split: {split}")

            stem = _stem(rgb_file)
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(dataset_dir, "depth", stem + ".h5")
            depth_unnorm_path = os.path.join(dataset_dir, "depth_ori", stem + ".h5")
            json_path = os.path.join(dataset_dir, stem + ".json")

            if not (
                os.path.isfile(json_path)
                and os.path.isfile(depth_path)
                and os.path.isfile(depth_unnorm_path)
            ):
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
            if "caption" not in meta:
                continue
            caption = meta["caption"]

            try:
                with h5py.File(depth_path, "r") as f:
                    depth_vis = np.asarray(f["depth"][:], dtype=np.float32)
                with h5py.File(depth_unnorm_path, "r") as f:
                    depth_metric = np.asarray(f["depth"][:], dtype=np.float32)
            except OSError:
                continue

            if depth_vis.ndim != 3 or depth_vis.shape[2] != 3:
                continue
            if depth_metric.ndim == 2:
                depth_metric = depth_metric[..., np.newaxis]
            elif depth_metric.ndim != 3 or depth_metric.shape[2] != 1:
                continue

            w, h = Image.open(rgb_path).size
            key = f"{dataset_type}_{stem}"
            yielded += 1
            yield key, {
                "image": rgb_path,
                "mask": depth_vis,
                "depth_metric": depth_metric,
                "caption": caption,
                "metadata": {
                    "stem": stem,
                    "dataset_type": dataset_type,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "depth_ori_path": depth_unnorm_path,
                    "json_path": json_path,
                    "image_size": np.array([w, h], dtype=np.int32),
                    "index": np.array([index], dtype=np.int32),
                },
            }

    def _generate_examples_gcs(self, gs_uri: str, split: str):


        bucket_name, prefix = _parse_gs_uri(gs_uri)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        rgb_prefix = prefix + "rgb/"

        cap = _max_rgb_list()
        entries: list[tuple[str, str, str]] = []
        for blob in bucket.list_blobs(prefix=rgb_prefix):
            if blob.name.endswith("/"):
                continue
            base = os.path.basename(blob.name)
            dt = _dataset_type_from_basename(base)
            if dt is None:
                continue
            entries.append((blob.name, base, dt))
            if cap is not None and len(entries) >= cap:
                break
        entries.sort(key=lambda x: x[1])

        max_yields = _max_examples_per_split()
        yielded = 0
        for index, (rgb_blob_name, rgb_file, dataset_type) in enumerate(entries):
            if max_yields is not None and yielded >= max_yields:
                break

            #get split
            if split == "validation":
                if index % 10 != 0:
                    continue
            elif split == "train":
                if index % 10 == 0:
                    continue
            else:
                raise ValueError(f"Invalid split: {split}")

            stem = _stem(rgb_file)
            depth_name = prefix + "depth/" + stem + ".h5"
            depth_ori_name = prefix + "depth_ori/" + stem + ".h5"
            json_name = prefix + stem + ".json"

            json_blob = bucket.blob(json_name)
            meta_raw = json_blob.download_as_bytes()

            meta = json.loads(meta_raw.decode("utf-8"))
            if "caption" not in meta:
                continue
            caption = meta["caption"]

            depth_blob = bucket.blob(depth_name)
            depth_ori_blob = bucket.blob(depth_ori_name)
            depth_bytes = depth_blob.download_as_bytes()
            depth_ori_bytes = depth_ori_blob.download_as_bytes()

            try:
                with h5py.File(io.BytesIO(depth_bytes), "r") as f:
                    depth_vis = np.asarray(f["depth"][:], dtype=np.float32)
                with h5py.File(io.BytesIO(depth_ori_bytes), "r") as f:
                    depth_metric = np.asarray(f["depth"][:], dtype=np.float32)
            except (KeyError, OSError):
                continue

            if depth_vis.ndim != 3 or depth_vis.shape[2] != 3:
                continue
            if depth_metric.ndim == 2:
                depth_metric = depth_metric[..., np.newaxis]
            elif depth_metric.ndim != 3 or depth_metric.shape[2] != 1:
                continue

            try:
                rgb_blob = bucket.blob(rgb_blob_name)
                rgb_bytes = rgb_blob.download_as_bytes()
            except gcp_exceptions.NotFound:
                continue

            rgb_arr = np.asarray(
                Image.open(io.BytesIO(rgb_bytes)).convert("RGB"),
                dtype=np.uint8,
            )
            key = f"{dataset_type}_{stem}"
            loc = f"gs://{bucket_name}/{rgb_blob_name}"
            yielded += 1
            yield key, {
                "image": rgb_arr,
                "mask": depth_vis,
                "depth_metric": depth_metric,
                "caption": caption,
                "metadata": {
                    "stem": stem,
                    "dataset_type": dataset_type,
                    "rgb_path": loc,
                    "depth_path": f"gs://{bucket_name}/{depth_name}",
                    "depth_ori_path": f"gs://{bucket_name}/{depth_ori_name}",
                    "json_path": f"gs://{bucket_name}/{json_name}",
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
    depth_metric = tf.cast(ex["depth_metric"], tf.float32)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    scale = long_side / tf.cast(tf.maximum(h, w), tf.float32)
    new_h = tf.maximum(target_h, tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32))
    new_w = tf.maximum(target_w, tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32))

    image_r = tf.image.resize(image, [new_h, new_w], method="bilinear", antialias=True)
    mask_r = tf.image.resize(mask, [new_h, new_w], method="bilinear", antialias=True)
    depth_r = tf.image.resize(depth_metric, [new_h, new_w], method="bilinear", antialias=True)

    merged = tf.concat([image_r, mask_r, depth_r], axis=-1)
    cropped = tf.image.random_crop(merged, size=[target_h, target_w, 7])

    ex["image"] = tf.cast(tf.clip_by_value(cropped[..., :3], 0.0, 255.0), tf.uint8)
    ex["mask"] = cropped[..., 3:6]
    ex["depth_metric"] = cropped[..., 6:7]
    ex["metadata"]["image_size"] = tf.cast([target_w, target_h], tf.int32)
    return ex


def load_data(
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("d3_depth", **load_kw)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    #ds = ds.map(resize_random_crop)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def sanity_check(
    tfds_data_dir: str,
    split: str = "train",
    num_batches: int = 1,
    batch_size: int = 4,
    max_show: int = 8,
):
    ds = tfds.load("d3_depth", split=split, data_dir=tfds_data_dir)
    ds = ds.map(resize_random_crop, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        # Match ``supervised_keys``: ``depth_vis`` is the 3-ch [0,1] visualization depth from ``depth/*.h5``.
        depth_vis = ex["mask"].numpy()
        cap = ex["caption"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            dvis = np.clip(depth_vis[j], 0.0, 1.0)
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            cj = cap[j]
            if hasattr(cj, "numpy"):
                cj = cj.numpy()
            if isinstance(cj, (bytes, np.bytes_)):
                title = cj.decode("utf-8", errors="replace")
            else:
                title = str(cj)
            plt.title(title[:60])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(dvis)
            plt.axis("off")
        plt.tight_layout()
        out = f"d3_depth_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    # See module docstring for build instructions.
    pass
