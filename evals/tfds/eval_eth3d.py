"""
TFDS builder for GenPercept ETH3D evaluation data.

Default on-disk layout (from `eval_eth3d_genpercept.tar.gz` extraction):

  root/
    rgb/<scene>/images/dslr_images/*.JPG
    rgb/<scene>/dslr_calibration_jpg/cameras.txt   # provides WIDTH/HEIGHT
    depth/<scene>_dslr_depth/<scene>/ground_truth_depth/dslr_images/<frame>.JPG

Note: ETH3D "depth" files are *raw little-endian float32 arrays* (H*W entries)
stored with a `.JPG` extension; they are not JPEG images.

Set env `GENPERCEPT_EVAL_ETH3D_ROOT` to point at the `eth3d/` folder.

Build:
  tfds build d3/eval_eth3d.py --data_dir=/path/to/tfds_output
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


_ROOT = os.environ.get("GENPERCEPT_EVAL_ETH3D_ROOT", "/mnt/klum/lvm/genpercept_eval/eval_eth3d/eth3d")


def _read_any_camera_size(cameras_txt: str) -> tuple[int, int]:
    with open(cameras_txt, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            # CAMERA_ID MODEL WIDTH HEIGHT ...
            if len(parts) >= 4:
                w = int(parts[2])
                h = int(parts[3])
                return w, h
    raise ValueError(f"Could not parse WIDTH/HEIGHT from {cameras_txt}")


def _read_raw_f32_depth(path: str, *, w: int, h: int) -> np.ndarray:
    expected = int(w) * int(h)
    x = np.fromfile(path, dtype="<f4", count=expected)
    if x.size != expected:
        raise ValueError(f"Wrong depth file size for {path}: got {x.size}, expected {expected}")
    return x.reshape((h, w))


class EvalEth3D(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="GenPercept evaluation dataset: ETH3D (RGB JPG + raw float32 depth).",
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
        rgb_root = os.path.join(root, "rgb")
        depth_root = os.path.join(root, "depth")
        if not (os.path.isdir(rgb_root) and os.path.isdir(depth_root)):
            raise FileNotFoundError(f"Expected {rgb_root} and {depth_root} under {root}")

        # Walk RGB frames and match to raw depth frames by filename.
        for rgb_path in sorted(walk_files(rgb_root, exts=(".jpg", ".jpeg", ".JPG", ".JPEG"))):
            rel = os.path.relpath(rgb_path, start=rgb_root).replace(os.sep, "/")
            parts = rel.split("/")
            if len(parts) < 4:
                continue
            scene = parts[0]
            frame = parts[-1]

            cameras_txt = os.path.join(rgb_root, scene, "dslr_calibration_jpg", "cameras.txt")
            if not os.path.isfile(cameras_txt):
                continue
            try:
                w, h = _read_any_camera_size(cameras_txt)
            except Exception:
                continue

            depth_scene = f"{scene}_dslr_depth"
            depth_path = os.path.join(
                depth_root,
                depth_scene,
                scene,
                "ground_truth_depth",
                "dslr_images",
                frame,
            )
            if not os.path.isfile(depth_path):
                continue

            try:
                depth2d = _read_raw_f32_depth(depth_path, w=w, h=h)
            except Exception:
                continue
            v = finite_mask(depth2d)
            depth = np.where(v, depth2d, 0.0).astype(np.float32, copy=False)[..., np.newaxis]
            valid = v[..., np.newaxis]

            key = f"{scene}/{frame}"
            yield key, {
                "image": rgb_path,
                "depth": depth,
                "valid_mask": valid,
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
    """Load prepared ``eval_eth3_d`` TFDS (split is ``test`` only)."""
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    ds = tfds.load("eval_eth3_d", **load_kw)
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
    ds = tfds.load("eval_eth3_d", split=split, data_dir=tfds_data_dir)
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
        out = f"eval_eth3_d_sanity_{datetime.now():%Y-%m-%d_%H-%M-%S}_b{i}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    pass

