"""
TFDS builder for GenPercept NYUv2 evaluation data.

Default on-disk layout (from `eval_nyu_genpercept.tar.gz` extraction):

  root/
    train/<scene>/{rgb_XXXX.png, depth_XXXX.png, filled_XXXX.png}
    test/<scene>/{rgb_XXXX.png, depth_XXXX.png, filled_XXXX.png}

Depth PNGs are uint16, typically millimeters (convert to meters by /1000).

Set env `GENPERCEPT_EVAL_NYU_ROOT` to point at the `nyuv2/` folder.

Build:
  tfds build d3/eval_nyu.py --data_dir=/path/to/tfds_output
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


_ROOT = os.environ.get("GENPERCEPT_EVAL_NYU_ROOT", "/mnt/klum/lvm/genpercept_eval/eval_nyu/nyuv2")


class EvalNyu(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="GenPercept evaluation dataset: NYUv2 (RGB + uint16 depth in mm).",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32, encoding="zlib"),
                    "filled_depth": tfds.features.Tensor(
                        shape=(None, None, 1), dtype=np.float32, encoding="zlib"
                    ),
                    "valid_mask": tfds.features.Tensor(shape=(None, None, 1), dtype=np.bool_, encoding="zlib"),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "split": tfds.features.Text(),
                            "scene": tfds.features.Text(),
                            "frame": tfds.features.Text(),
                            "rel_image_path": tfds.features.Text(),
                            "rel_depth_path": tfds.features.Text(),
                            "rel_filled_depth_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=("image", "depth"),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            "train": self._generate_examples(_ROOT, split="train"),
            "test": self._generate_examples(_ROOT, split="test"),
        }

    def _generate_examples(self, root: str, split: str):
        root = os.path.abspath(root)
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        for scene in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for fn in sorted(os.listdir(scene_dir)):
                if not (fn.startswith("rgb_") and fn.lower().endswith(".png")):
                    continue
                frame = os.path.splitext(fn)[0].split("_", 1)[-1]
                rgb_path = os.path.join(scene_dir, fn)
                depth_path = os.path.join(scene_dir, f"depth_{frame}.png")
                filled_path = os.path.join(scene_dir, f"filled_{frame}.png")
                if not (os.path.isfile(depth_path) and os.path.isfile(filled_path)):
                    continue

                try:
                    d_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
                    f_u16 = np.array(Image.open(filled_path), dtype=np.uint16)
                except Exception:
                    continue
                if d_u16.ndim != 2 or f_u16.ndim != 2:
                    continue
                if d_u16.shape != f_u16.shape:
                    continue

                depth = depth_u16_to_m(d_u16, scale=1000.0)[..., np.newaxis]
                filled = depth_u16_to_m(f_u16, scale=1000.0)[..., np.newaxis]
                valid = (d_u16 > 0)[..., np.newaxis]

                key = f"{split}/{scene}/{frame}"
                yield key, {
                    "image": rgb_path,
                    "depth": depth,
                    "filled_depth": filled,
                    "valid_mask": valid.astype(np.bool_, copy=False),
                    "metadata": {
                        "split": split,
                        "scene": scene,
                        "frame": frame,
                        "rel_image_path": relpath_posix(rgb_path, root),
                        "rel_depth_path": relpath_posix(depth_path, root),
                        "rel_filled_depth_path": relpath_posix(filled_path, root),
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
    ds = tfds.load("eval_nyu", **load_kw)
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
    """Load TFDS and save RGB + sparse depth + filled depth per row."""
    ds = tfds.load("eval_nyu", split=split, data_dir=tfds_data_dir)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        depth = ex["depth"].numpy()
        filled = ex["filled_depth"].numpy()
        vm = ex["valid_mask"].numpy()
        scenes = ex["metadata"]["scene"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(12, 2.5 * b))
        for j in range(b):
            dvis = depth_to_vis_2d(depth[j], vm[j])
            fvis = depth_to_vis_2d(filled[j])
            plt.subplot(b, 3, 3 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            plt.title(decode_tf_text(scenes[j])[:40])
            plt.subplot(b, 3, 3 * j + 2)
            plt.imshow(dvis, cmap="viridis", vmin=0.0, vmax=1.0)
            plt.axis("off")
            plt.subplot(b, 3, 3 * j + 3)
            plt.imshow(fvis, cmap="viridis", vmin=0.0, vmax=1.0)
            plt.axis("off")
        plt.tight_layout()
        out = f"eval_nyu_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    pass

