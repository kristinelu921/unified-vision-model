from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

try:
    from .lvis_detection_tfds import load_data as load_detection_data
    from .lvis_segmentation import load_data as load_segmentation_data
except ImportError:  # pragma: no cover
    from jflux2.dataloaders.lvis_detection_tfds import load_data as load_detection_data
    from jflux2.dataloaders.lvis_segmentation import load_data as load_segmentation_data


def _pixels_changed_by_mask(rgb: np.ndarray, mask_rgb: np.ndarray, *, atol: int = 12) -> np.ndarray:
    delta = np.max(np.abs(mask_rgb.astype(np.int16) - rgb.astype(np.int16)), axis=-1)
    return delta > atol


def run_visualization(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    split: str = "train",
    num_samples: int = 4,
    crop_size: int | Sequence[int] | None = 256,
    resize_long_side: int = 512,
) -> None:
    """Save per-sample PNGs: ``*_rgb.png``, ``*_mask.png``, ``*_overlay.png``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bs = max(int(num_samples), 1)
    ds = load_detection_data(
        split=split,
        data_dir=data_dir,
        batch_size=bs,
        repeat=False,
        shuffle=False,
        crop_size=crop_size,
        resize_long_side=resize_long_side,
    )
    images, masks, mask_valids, _metadata = next(iter(ds.take(1)))
    imgs = images.numpy()
    msks = masks.numpy()
    valids = mask_valids.numpy()
    n = min(int(num_samples), int(imgs.shape[0]))

    for i in range(n):
        stem = f"sample_{i:02d}"
        rgb = imgs[i]
        m = msks[i]
        valid = bool(valids[i])
        Image.fromarray(rgb).save(out / f"{stem}_rgb.png")
        sel = _pixels_changed_by_mask(rgb, m)
        Image.fromarray((sel * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
        img_f = rgb.astype(np.float32)
        ov = img_f.copy()
        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        ov[sel] = 0.55 * ov[sel] + 0.45 * red
        Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).save(out / f"{stem}_overlay.png")
        (out / f"{stem}_mask_valid.txt").write_text(f"mask_valid={valid}\n", encoding="utf-8")

    print(f"Wrote {n} samples (rgb, mask, overlay) under {out}")


def run_segmentation_visualization(
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    split: str = "train",
    num_samples: int = 4,
    crop_size: int | Sequence[int] | None = 256,
    resize_long_side: int = 512,
) -> None:
    """
    Save per-sample PNGs for ``lvis_segmentation``:
    - ``*_rgb.png``: source image
    - ``*_mask_rgb.png``: 3-channel mask as returned by loader
    - ``*_mask_bw.png``: single-channel B/W view from mask channel 0
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bs = max(int(num_samples), 1)
    ds = load_segmentation_data(
        split=split,
        data_dir=data_dir,
        batch_size=bs,
        repeat=False,
        shuffle=False,
        crop_size=crop_size,
        resize_long_side=resize_long_side,
    )
    images, masks, metadata = next(iter(ds.take(1)))
    imgs = images.numpy()
    msks = masks.numpy()
    n = min(int(num_samples), int(imgs.shape[0]))

    for i in range(n):
        stem = f"sample_{i:02d}"
        rgb = imgs[i]
        mask = msks[i]

        Image.fromarray(rgb).save(out / f"{stem}_rgb.png")
        Image.fromarray(mask).save(out / f"{stem}_mask_rgb.png")
        Image.fromarray(mask[..., 0]).save(out / f"{stem}_mask_bw.png")

        c0 = mask[..., 0]
        c1 = mask[..., 1]
        c2 = mask[..., 2]
        channels_equal = bool(np.array_equal(c0, c1) and np.array_equal(c0, c2))
        unique_values = np.unique(mask).tolist()
        (out / f"{stem}_mask_stats.txt").write_text(
            (
                f"category_id={int(np.asarray(metadata['category_id'][i]))}\n"
                f"category_name={np.asarray(metadata['category_name'][i]).item()}\n"
                f"class_idx={int(np.asarray(metadata['class_idx'][i]))}\n"
                f"mask_shape={tuple(mask.shape)}\n"
                f"channels_equal={channels_equal}\n"
                f"unique_values={unique_values}\n"
            ),
            encoding="utf-8",
        )

    print(f"Wrote {n} samples (rgb, mask_rgb, mask_bw, stats) under {out}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default_out = root / "recolor_example" / "lvis_bbox_mask_viz"

    parser = argparse.ArgumentParser(
        description="Visualize LVIS masks from detection or segmentation loaders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example (from jflux/ with PYTHONPATH=.):\n"
            "  python3 -m jflux2.dataloaders.visualize_lvis_detection "
            "--loader segmentation --data_dir gs://.../lvis_giraffe_tfds --split train --num 4"
        ),
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="detection",
        choices=("detection", "segmentation"),
        help="Select dataloader backend to visualize.",
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Built TFDS path (local or gs://).")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=Path, default=default_out, help=f"Default: {default_out}")
    parser.add_argument("--num", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--resize_long_side", type=int, default=512)
    args = parser.parse_args()

    if args.loader == "segmentation":
        run_segmentation_visualization(
            args.data_dir,
            args.out_dir,
            split=args.split,
            num_samples=args.num,
            crop_size=args.crop_size,
            resize_long_side=args.resize_long_side,
        )
    else:
        run_visualization(
            args.data_dir,
            args.out_dir,
            split=args.split,
            num_samples=args.num,
            crop_size=args.crop_size,
            resize_long_side=args.resize_long_side,
        )


if __name__ == "__main__":
    main()
