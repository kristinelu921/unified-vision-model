"""Single-pass VAE encoding for Mass13k TFDS images.

Mimics `kristine/one_pass.py` behavior:
- load data from TFDS dataloader
- iterate through the dataset exactly once (no repeat)
- encode image batches with FLUX2 VAE
- save each latent with numpy to output directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure jflux2 and port_vae_safetensor_jax are importable when run from lvm
_JFLUX2_DIR = "/kmh-nfs-ssd-us-mount/code/xtiange/jflux2"
if _JFLUX2_DIR not in sys.path:
    sys.path.insert(0, _JFLUX2_DIR)

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from jflux2.modules.autoencoder import AutoEncoder

# Paths for VAE weights
_ROOT = "/kmh-nfs-ssd-us-mount"
_DATA_ROOT = f"{_ROOT}/data/xtiange"
_JAX_WEIGHTS = f"{_DATA_ROOT}/jax_weights"
_HF_CACHE = f"{_DATA_ROOT}/hf_cache"


def _default_paths(model_variant: str) -> dict[str, str]:
    variant = model_variant.upper()
    return {
        "vae_weights": f"{_JAX_WEIGHTS}/{variant}/jflux2_vae.safetensors",
        "vae_config": f"{_HF_CACHE}/vae/config.json",
    }


from port_vae_safetensor_jax import (
    autoencoder_params_from_diffusers_config,
    load_converted_jax_safetensors as load_vae_weights,
    load_converted_weights_into_autoencoder,
)


def _get_device(device_name: str) -> jax.Device:
    if device_name == "cpu":
        cpus = jax.devices("cpu")
        if not cpus:
            raise RuntimeError("No JAX CPU device available.")
        return cpus[0]
    if device_name == "tpu":
        tpus = jax.devices("tpu")
        if not tpus:
            raise RuntimeError("No JAX TPU device available.")
        return tpus[0]
    raise ValueError(f"Unsupported device={device_name!r}")


def _to_filename(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
    if hasattr(value, "decode"):
        return value.decode("utf-8")
    return str(value)


def _preprocess_example(image: tf.Tensor, mask: tf.Tensor, metadata: dict[str, tf.Tensor]):
    shape = tf.shape(image)
    small_dim = tf.reduce_min([shape[0], shape[1]])
    cropped_img = tf.image.resize_with_crop_or_pad(image, small_dim, small_dim)
    cropped_mask = tf.image.resize_with_crop_or_pad(mask, small_dim, small_dim)
    image = tf.image.resize(cropped_img, [256, 256], method="bilinear")
    mask = tf.image.resize(cropped_mask, [256, 256], method="nearest")
    return image, mask, metadata


def _load_data_one_pass(dataset_name: str, split: str, data_dir: str, batch_size: int):
    ds = tfds.load(dataset_name, split=split, data_dir=data_dir)
    ds = ds.map(lambda ex: (ex["image"], ex["mask"], ex["metadata"]), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _to_vae_input(images_hwc: np.ndarray) -> jnp.ndarray:
    if images_hwc.ndim != 4 or images_hwc.shape[-1] != 3:
        raise ValueError(f"Expected image batch shape [B,H,W,3], got {images_hwc.shape}.")
    x = images_hwc.astype(np.float32)
    x = x / 127.5 - 1.0
    x = np.transpose(x, (0, 3, 1, 2))  # BHWC -> BCHW
    return jnp.asarray(x, dtype=jnp.float32)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or f"{_ROOT}/code/kristine/lvm/uco3d_car_{args.split}_latents"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    defaults = _default_paths("4B")
    vae_weights = args.vae_weights or defaults["vae_weights"]
    vae_config = args.vae_config or defaults["vae_config"]

    device = _get_device(args.device)
    print(f"[setup] device={device}")
    print(f"[setup] vae_weights={vae_weights}")
    print(f"[setup] vae_config={vae_config}")

    with jax.default_device(device):
        ae_params = autoencoder_params_from_diffusers_config(
            vae_config,
            rng_seed=args.seed,
            param_dtype=jnp.float32,
        )
        ae = AutoEncoder(ae_params)
        ae = load_converted_weights_into_autoencoder(ae, load_vae_weights(vae_weights))

    ds = _load_data_one_pass(
        dataset_name=args.dataset_name,
        split=args.split,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(
        f"[data] dataset={args.dataset_name} split={args.split} "
        f"data_dir={args.data_dir} batch_size={args.batch_size}"
    )

    written = 0
    total = 0

    # Exact single pass over dataloader.
    for images, masks, metadata in ds:
        images_np = images.numpy() if hasattr(images, "numpy") else np.asarray(images)
        masks_np = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        filenames_raw = metadata["filename"]
        filenames = filenames_raw.numpy() if hasattr(filenames_raw, "numpy") else np.asarray(filenames_raw)

        x = _to_vae_input(images_np)
        masks_np = _to_vae_input(masks_np)
        with jax.default_device(device):
            z = ae.encode(jax.device_put(x, device))
            z = np.asarray(jax.device_get(z), dtype=np.float32)
            masks = ae.encode(jax.device_put(masks_np, device))
            masks = np.asarray(jax.device_get(masks), dtype=np.float32)

        batch_size = int(z.shape[0])
        total += batch_size

        for i in range(batch_size):
            filename = _to_filename(filenames[i])
            out_path = output_dir / f"{filename}_image.npy"
            out_path_mask = output_dir / f"{filename}_mask.npy"
            if out_path.exists() and not args.overwrite:
                continue
            np.save(out_path, z[i], allow_pickle=False)
            np.save(out_path_mask, masks[i], allow_pickle=False)
            written += 1

        if total % args.log_every == 0:
            print(f"[encode] seen={total} written={written}")

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "vae_weights": str(vae_weights),
        "vae_config": str(vae_config),
        "device": args.device,
        "batch_size": args.batch_size,
        "count_total": total,
        "count_written": written,
        "count_skipped_existing": total - written,
    }
    with (output_dir / "_index.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[done] "
        f"total={total} written={written} skipped={total - written} output_dir={output_dir}"
    )


def parse_args() -> argparse.Namespace:
    _lvm = f"{_ROOT}/code/kristine/lvm"
    parser = argparse.ArgumentParser(description="Single-pass TFDS VAE encoding")
    parser.add_argument("--dataset_name", type=str, default="UCO3DCar")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"{_lvm}/uco3d_car_tfds",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to {lvm}/uco3d_car_{split}_latents",
    )
    parser.add_argument("--vae_weights", type=str, default=None)
    parser.add_argument("--vae_config", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "tpu"], default="tpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_every", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
