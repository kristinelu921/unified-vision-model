"""
TFDS builder for GenPercept DIode evaluation data.

Default on-disk layout (from `eval_diode_genpercept.tar.gz` extraction):

  root/
    indoors/scene_*/scan_*/*.png
    indoors/scene_*/scan_*/*_depth.npy
    indoors/scene_*/scan_*/*_depth_mask.npy
    outdoors/...

Set env `GENPERCEPT_EVAL_DIODE_ROOT` to point at the `diode/` folder.

Build:
  tfds build d3/eval_diode.py --data_dir=/path/to/tfds_output
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .eval_common import (
    decode_tf_text,
    depth_to_vis_2d,
    finite_mask,
    relpath_posix,
    walk_files,
)


_ROOT = os.environ.get("GENPERCEPT_EVAL_DIODE_ROOT", "/mnt/klum/lvm/genpercept_eval/eval_diode/diode")


class EvalDiode(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="GenPercept evaluation dataset: DIode (RGB + depth .npy + valid mask .npy).",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32, encoding="zlib"),
                    "valid_mask": tfds.features.Tensor(shape=(None, None, 1), dtype=np.bool_, encoding="zlib"),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "rel_image_path": tfds.features.Text(),
                            "rel_depth_path": tfds.features.Text(),
                            "rel_mask_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "depth"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {"test": self._generate_examples(_ROOT)}

    def _generate_examples(self, root: str):
        root = os.path.abspath(root)
        for img_path in sorted(walk_files(root, exts=(".png", ".jpg", ".jpeg"))):
            base = os.path.basename(img_path)
            if base.endswith("_depth.png") or base.endswith("_mask.png"):
                continue
            stem, _ = os.path.splitext(img_path)
            depth_path = stem + "_depth.npy"
            mask_path = stem + "_depth_mask.npy"
            if not (os.path.isfile(depth_path) and os.path.isfile(mask_path)):
                raise ValueError(f"Depth or mask file not found: {depth_path} or {mask_path}")

            try:
                depth = np.load(depth_path)
                m = np.load(mask_path)
            except Exception:
                raise ValueError(f"Error loading depth or mask: {depth_path} or {mask_path}")
            valid = (m != 0) & finite_mask(depth[..., 0])
            valid = valid[..., np.newaxis]

            key = relpath_posix(img_path, root)
            yield key, {
                "image": img_path,
                "depth": depth,
                "valid_mask": valid,
                "metadata": {
                    "rel_image_path": relpath_posix(img_path, root),
                    "rel_depth_path": relpath_posix(depth_path, root),
                    "rel_mask_path": relpath_posix(mask_path, root),
                },
            }


def load_data(
    split: str = "test",
    data_dir: str | None = None,
    batch_size: int = 8,
    repeat: bool = True,
    shuffle: bool = True,
    shuffle_buffer: int = 256,
):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("eval_diode", **load_kw)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def sanity_check(
    tfds_data_dir: str,
    split: str = "test",
    num_batches: int = 1,
    batch_size: int = 4,
    max_show: int = 8,
):
    """Load TFDS and save a grid of RGB + depth visualizations."""
    ds = tfds.load("eval_diode", split=split, data_dir=tfds_data_dir)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        depth = ex["depth"].numpy()
        vm = ex["valid_mask"].numpy()
        rel = ex["metadata"]["rel_image_path"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            dvis = depth_to_vis_2d(depth[j], vm[j])
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            plt.title(decode_tf_text(rel[j])[:80])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(dvis, cmap="viridis", vmin=0.0, vmax=1.0)
            plt.axis("off")
        plt.tight_layout()
        out = f"eval_diode_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    pass

