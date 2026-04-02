"""
TFDS builder for GenPercept ScanNet evaluation data.

Default on-disk layout (from `eval_scannet_genpercept.tar.gz` extraction):

  root/
    sceneXXXX_YY/
      color/*.jpg
      depth/*.png   # uint16 depth (typically millimeters)

Set env `GENPERCEPT_EVAL_SCANNET_ROOT` to point at the `scannet/` folder.

Build:
  tfds build d3/eval_scannet.py --data_dir=/path/to/tfds_output
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from .eval_common import decode_tf_text, depth_to_vis_2d, depth_u16_to_m, relpath_posix


_ROOT = os.environ.get("GENPERCEPT_EVAL_SCANNET_ROOT", "/mnt/klum/lvm/genpercept_eval/eval_scannet/scannet")


class EvalScanNet(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="GenPercept evaluation dataset: ScanNet (RGB JPG + uint16 depth in mm).",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32, encoding="zlib"),
                    "valid_mask": tfds.features.Tensor(shape=(None, None, 1), dtype=np.bool_, encoding="zlib"),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "scene": tfds.features.Text(),
                            "frame": tfds.features.Text(),
                            "rel_image_path": tfds.features.Text(),
                            "rel_depth_path": tfds.features.Text(),
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
        for scene in sorted(os.listdir(root)):
            scene_dir = os.path.join(root, scene)
            if not os.path.isdir(scene_dir):
                continue
            color_dir = os.path.join(scene_dir, "color")
            depth_dir = os.path.join(scene_dir, "depth")
            if not (os.path.isdir(color_dir) and os.path.isdir(depth_dir)):
                continue

            for fn in sorted(os.listdir(color_dir)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                frame = os.path.splitext(fn)[0]
                rgb_path = os.path.join(color_dir, fn)
                depth_path = os.path.join(depth_dir, frame + ".png")
                if not os.path.isfile(depth_path):
                    continue

                try:
                    d_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
                except Exception:
                    continue
                if d_u16.ndim != 2:
                    continue

                depth = depth_u16_to_m(d_u16, scale=1000.0)[..., np.newaxis]
                valid = (d_u16 > 0)[..., np.newaxis]

                key = f"{scene}/{frame}"
                yield key, {
                    "image": rgb_path,
                    "depth": depth,
                    "valid_mask": valid.astype(np.bool_, copy=False),
                    "metadata": {
                        "scene": scene,
                        "frame": frame,
                        "rel_image_path": relpath_posix(rgb_path, root),
                        "rel_depth_path": relpath_posix(depth_path, root),
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
    """Load prepared ``eval_scan_net`` TFDS (split is ``test`` only)."""
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("eval_scan_net", **load_kw)
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
    ds = tfds.load("eval_scan_net", split=split, data_dir=tfds_data_dir)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        depth = ex["depth"].numpy()
        vm = ex["valid_mask"].numpy()
        scenes = ex["metadata"]["scene"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            dvis = depth_to_vis_2d(depth[j], vm[j])
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            plt.title(decode_tf_text(scenes[j])[:80])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(dvis, cmap="viridis", vmin=0.0, vmax=1.0)
            plt.axis("off")
        plt.tight_layout()
        out = f"eval_scan_net_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    pass

