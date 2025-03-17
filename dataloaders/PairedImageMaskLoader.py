from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import jax.numpy as jnp
import numpy as np
from PIL import Image

_PALETTE = [
    [0, 0, 0],
    [255, 0, 0],
    [64, 34, 64],
    [0, 255, 0],
    [93, 64, 0],
    [0, 0, 255],
    [0, 255, 255],
]


@dataclass(frozen=True)
class TrainExample:
    image_path: Path
    mask_path: Path


class PairedImageMaskLoader:
    """Non-TFDS image/mask loader that yields batched tensors + metadata."""

    def __init__(
        self,
        dataset_root: str,
        batch_size: int,
        seed: int,
        image_size: int = 256,
        images_dir: str | None = None,
        masks_dir: str | None = None,
        drop_last: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}.")

        self.images_dir = Path(images_dir) if images_dir else self.dataset_root / "images"
        self.masks_dir = Path(masks_dir) if masks_dir else self.dataset_root / "masks"
        self._rng = np.random.default_rng(seed)

        self.examples = self._build_examples()
        if not self.examples:
            raise ValueError("No valid training examples found.")
        self.image_shape, self.mask_shape = self._validate_shapes()

    def _build_examples(self) -> list[TrainExample]:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks_dir not found: {self.masks_dir}")

        rows: list[TrainExample] = []
        missing: list[str] = []
        image_paths = sorted(p for p in self.images_dir.iterdir() if p.is_file())
        if not image_paths:
            raise ValueError(f"No files found in images_dir: {self.images_dir}")

        for image_path in image_paths:
            mask_path = self.masks_dir / image_path.name
            if not mask_path.exists():
                missing.append(str(mask_path))
                continue
            rows.append(TrainExample(image_path=image_path, mask_path=mask_path))

        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                "Dataset files missing. First missing paths:\n"
                f"{preview}\n"
                f"(total missing: {len(missing)})"
            )
        return rows

    def _load_rgb_tensor(self, path: Path) -> jnp.ndarray:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            arr = np.asarray(img, dtype=np.float32)
        arr = arr / 127.5 - 1.0
        arr = np.transpose(arr, (2, 0, 1))
        return jnp.asarray(arr, dtype=jnp.float32)

    def _load_mask_tensor(self, path: Path) -> jnp.ndarray:
        with Image.open(path) as img:
            indexed_mask = img.convert("L")
            indexed_mask = indexed_mask.resize(
                (self.image_size, self.image_size),
                Image.Resampling.NEAREST,
            )
            color_mask = _process_annotation(indexed_mask)
            arr = np.asarray(color_mask, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr / 127.5 - 1.0
        return jnp.asarray(arr, dtype=jnp.float32)

    def _validate_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        first = self.examples[0]
        image_shape = tuple(self._load_rgb_tensor(first.image_path).shape)
        mask_shape = tuple(self._load_mask_tensor(first.mask_path).shape)
        for ex in self.examples[1:]:
            if tuple(self._load_rgb_tensor(ex.image_path).shape) != image_shape:
                raise ValueError(f"All images must share one shape. Mismatch at {ex.image_path}.")
            if tuple(self._load_mask_tensor(ex.mask_path).shape) != mask_shape:
                raise ValueError(f"All masks must share one shape. Mismatch at {ex.mask_path}.")
        return image_shape, mask_shape

    def __len__(self) -> int:
        return len(self.examples)

    def iter_batches_from_examples(
        self,
        examples: list[TrainExample],
        shuffle: bool = True,
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, dict[str, np.ndarray]]]:
        idxs = np.arange(len(examples))
        if shuffle:
            self._rng.shuffle(idxs)

        for start in range(0, len(idxs), self.batch_size):
            end = start + self.batch_size
            if end > len(idxs) and self.drop_last:
                break
            batch_indices = idxs[start:end]

            imgs: list[jnp.ndarray] = []
            masks: list[jnp.ndarray] = []
            filenames: list[str] = []
            image_paths: list[str] = []
            mask_paths: list[str] = []
            for i in batch_indices:
                ex = examples[int(i)]
                imgs.append(self._load_rgb_tensor(ex.image_path))
                masks.append(self._load_mask_tensor(ex.mask_path))
                filenames.append(ex.image_path.name)
                image_paths.append(str(ex.image_path))
                mask_paths.append(str(ex.mask_path))

            metadata = {
                "filename": np.asarray(filenames, dtype=np.str_),
                "image_path": np.asarray(image_paths, dtype=np.str_),
                "mask_path": np.asarray(mask_paths, dtype=np.str_),
            }
            yield jnp.stack(imgs, axis=0), jnp.stack(masks, axis=0), metadata

    def iter_batches(
        self,
        shuffle: bool = True,
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, dict[str, np.ndarray]]]:
        yield from self.iter_batches_from_examples(self.examples, shuffle=shuffle)


def _resolve_split_root(data_dir: str, split: str) -> Path:
    root = Path(data_dir)
    split_root = root / split
    if (split_root / "images").exists() and (split_root / "masks").exists():
        return split_root
    if (root / "images").exists() and (root / "masks").exists():
        return root
    raise FileNotFoundError(
        "Could not find dataset layout under data_dir. Expected either "
        f"{data_dir}/{split}/images+masks or {data_dir}/images+masks. "
        f"data_dir={data_dir}, split={split}"
    )


def _process_annotation(anno_image: Image.Image) -> Image.Image:
    new_anno_image = Image.new("RGB", anno_image.size)
    pixels = anno_image.load()
    new_pixels = new_anno_image.load()
    for y in range(anno_image.size[1]):
        for x in range(anno_image.size[0]):
            pixel_value = int(pixels[x, y])
            if pixel_value < 0 or pixel_value >= len(_PALETTE):
                pixel_value = 0
            new_pixels[x, y] = tuple(_PALETTE[pixel_value])
    return new_anno_image


def load_data(
    dataset_name: str = "PairedImageMask",
    split: str = "train",
    data_dir: str | None = None,
    batch_size: int = 64,
    repeat: bool = True,
    shuffle: bool = True,
    *,
    seed: int = 42,
    image_size: int = 256,
    drop_last: bool = True,
):
    """Mass13kLatents-like API, but for non-TFDS image/mask folders."""
    del dataset_name
    if data_dir is None:
        raise ValueError("data_dir is required for PairedImageMaskLoader.")

    split_root = _resolve_split_root(data_dir=data_dir, split=split)
    loader = PairedImageMaskLoader(
        dataset_root=str(split_root),
        batch_size=batch_size,
        seed=seed,
        image_size=image_size,
        drop_last=drop_last,
    )

    def _iter():
        while True:
            yielded = False
            for batch in loader.iter_batches(shuffle=shuffle):
                yielded = True
                yield batch
            if not repeat:
                break
            if not yielded:
                break

    return _iter()
