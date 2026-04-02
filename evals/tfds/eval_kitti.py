"""
TFDS builder for GenPercept KITTI (Eigen split style) evaluation data.

Default on-disk layout (from `eval_kitti_genpercept.tar.gz` extraction):

  root/
    <drive>_sync/proj_depth/groundtruth/image_02/<frame>.png          # uint16 depth, scaled by 256
    <date>/<drive>_sync/image_02/data/<frame>.png                     # RGB

Set env `GENPERCEPT_EVAL_KITTI_ROOT` to point at the `kitti/` folder.

Build:
  tfds build d3/eval_kitti.py --data_dir=/path/to/tfds_output
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


_ROOT = os.environ.get("GENPERCEPT_EVAL_KITTI_ROOT", "/mnt/klum/lvm/genpercept_eval/eval_kitti/kitti")


def _iter_drives(root: str) -> list[str]:
    out: list[str] = []
    for name in sorted(os.listdir(root)):
        if name.endswith("_sync") and os.path.isdir(os.path.join(root, name)):
            out.append(name)
    return out


class EvalKitti(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="GenPercept evaluation dataset: KITTI (RGB PNG + uint16 depth/256).",
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "depth": tfds.features.Tensor(shape=(None, None, 1), dtype=np.float32, encoding="zlib"),
                    "valid_mask": tfds.features.Tensor(shape=(None, None, 1), dtype=np.bool_, encoding="zlib"),
                    "metadata": tfds.features.FeaturesDict(
                        {
                            "drive": tfds.features.Text(),
                            "date": tfds.features.Text(),
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
        for drive in _iter_drives(root):
            date = drive[:10]
            depth_dir = os.path.join(
                root,
                drive,
                "proj_depth",
                "groundtruth",
                "image_02",
            )
            if not os.path.isdir(depth_dir):
                continue
            rgb_dir = os.path.join(root, date, drive, "image_02", "data")
            if not os.path.isdir(rgb_dir):
                continue

            for fn in sorted(os.listdir(depth_dir)):
                if not fn.lower().endswith(".png"):
                    continue
                depth_path = os.path.join(depth_dir, fn)
                rgb_path = os.path.join(rgb_dir, fn)
                if not os.path.isfile(rgb_path):
                    continue

                try:
                    d_u16 = np.array(Image.open(depth_path), dtype=np.uint16)
                except Exception:
                    continue
                if d_u16.ndim != 2:
                    continue
                depth_m = depth_u16_to_m(d_u16, scale=256.0)
                valid = (d_u16 > 0)[..., np.newaxis]
                depth = depth_m[..., np.newaxis]

                frame = os.path.splitext(fn)[0]
                key = f"{drive}/{frame}"
                yield key, {
                    "image": rgb_path,
                    "depth": depth.astype(np.float32, copy=False),
                    "valid_mask": valid.astype(np.bool_, copy=False),
                    "metadata": {
                        "drive": drive,
                        "date": date,
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
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("eval_kitti", **load_kw)
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
    ds = tfds.load("eval_kitti", split=split, data_dir=tfds_data_dir)
    ds = ds.batch(batch_size).take(num_batches)

    for i, ex in enumerate(ds):
        rgb = ex["image"].numpy()
        depth = ex["depth"].numpy()
        vm = ex["valid_mask"].numpy()
        frames = ex["metadata"]["frame"]
        b = min(max_show, rgb.shape[0])
        plt.figure(figsize=(10, 2.5 * b))
        for j in range(b):
            dvis = depth_to_vis_2d(depth[j], vm[j])
            plt.subplot(b, 2, 2 * j + 1)
            plt.imshow(np.clip(rgb[j], 0, 255).astype(np.uint8))
            plt.axis("off")
            plt.title(decode_tf_text(frames[j])[:80])
            plt.subplot(b, 2, 2 * j + 2)
            plt.imshow(dvis, cmap="viridis", vmin=0.0, vmax=1.0)
            plt.axis("off")
        plt.tight_layout()
        out = f"eval_kitti_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    pass

