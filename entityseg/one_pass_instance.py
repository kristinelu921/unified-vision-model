from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib
if os.environ.get("DISPLAY") and matplotlib.get_backend().lower() == "agg":
    # Prefer an interactive backend when a display is available.
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import tensorflow_datasets as tfds

DEFAULT_TFDS_DIR = "/mnt/klum/entityseg/entityseg_instance_tfds/"
DEFAULT_DATASET_NAME = "entity_seg_instance"
DEFAULT_HEADLESS_OUT = "entityseg_instance_browser"


def _as_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _as_str(value: Any) -> str:
    v = np.asarray(value).item()
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    return str(v)


def _normalize_split(split: str) -> str:
    s = split.strip().lower()
    if s in ("val", "valid", "validation"):
        return "validation"
    return "train"


def _build_group(first_ex: dict[str, Any], remaining_iter, pending_ex: dict[str, Any] | None):
    current_image_id = _as_int(first_ex["metadata"]["image_id"])
    image = np.asarray(first_ex["image"], dtype=np.uint8)
    group = {
        "image_id": current_image_id,
        "file_name": _as_str(first_ex["metadata"]["file_name"]),
        "image": image,
        "instances": [],
    }

    def add_instance(example: dict[str, Any]) -> None:
        group["instances"].append(
            {
                "bbox": np.asarray(example["instance"]["bbox"], dtype=np.float32),
                "category_id": _as_int(example["instance"]["category_id"]),
                "mask": np.asarray(example["instance"]["mask_raster"], dtype=np.uint8),
                "annotation_id": _as_int(example["metadata"]["annotation_id"]),
            }
        )

    add_instance(first_ex)
    pending = pending_ex

    while True:
        if pending is not None:
            ex = pending
            pending = None
        else:
            try:
                ex = next(remaining_iter)
            except StopIteration:
                return group, None

        image_id = _as_int(ex["metadata"]["image_id"])
        if image_id != current_image_id:
            return group, ex
        add_instance(ex)


def browse_tfds(
    data_dir: str,
    dataset_name: str,
    split: str = "train",
    start_index: int = 0,
    step: int = 1,
    save_limit: int = 1000,
    out_dir: str = DEFAULT_HEADLESS_OUT,
) -> None:
    split = _normalize_split(split)
    raw_ds = tfds.load(dataset_name, data_dir=data_dir, split=split, shuffle_files=False)
    ex_iter = iter(tfds.as_numpy(raw_ds))
    groups: list[dict[str, Any]] = []
    pending_example = None
    stream_exhausted = False
    idx = max(0, int(start_index))
    step = max(1, int(step))

    def ensure_group(target_index: int) -> bool:
        nonlocal pending_example, stream_exhausted
        while len(groups) <= target_index and not stream_exhausted:
            first = pending_example
            if first is None:
                try:
                    first = next(ex_iter)
                except StopIteration:
                    stream_exhausted = True
                    break
            group, pending_example = _build_group(first, ex_iter, None)
            groups.append(group)
        return len(groups) > target_index

    def _group_masks(group: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
        instances = group["instances"]
        first_mask = np.asarray(instances[0]["mask"], dtype=np.uint8) if instances else np.zeros((1, 1), dtype=np.uint8)
        if first_mask.ndim == 3:
            first_mask = first_mask[..., 0]
        union_mask = np.zeros(first_mask.shape, dtype=np.uint8)
        num_shape_mismatch = 0
        for inst in instances:
            mask = np.asarray(inst["mask"], dtype=np.uint8)
            if mask.ndim == 3:
                mask = mask[..., 0]
            if mask.shape != union_mask.shape:
                num_shape_mismatch += 1
                continue
            union_mask = np.maximum(union_mask, (mask > 0).astype(np.uint8) * 255)
        return first_mask, union_mask, num_shape_mismatch

    def _resize_mask_to_image(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (h, w):
            mask = np.asarray(
                Image.fromarray(mask.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST),
                dtype=np.uint8,
            )
        return mask

    def _render_triptych_fast(group: dict[str, Any]) -> np.ndarray:
        image = group["image"]
        h, w = image.shape[:2]
        overlay = image.astype(np.float32).copy()
        union_mask = np.zeros((h, w), dtype=np.uint8)
        instances = group["instances"]

        for i, inst in enumerate(instances):
            mask = _resize_mask_to_image(np.asarray(inst["mask"], dtype=np.uint8), h, w) > 0
            color = np.array(plt.cm.hsv((i * 0.61803398875) % 1.0)[:3], dtype=np.float32) * 255.0
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
            union_mask = np.maximum(union_mask, mask.astype(np.uint8) * 255)

        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_u8, mode="RGB")
        draw = ImageDraw.Draw(overlay_pil)
        for inst in instances:
            x, y, bw, bh = [float(v) for v in np.asarray(inst["bbox"], dtype=np.float32)]
            draw.rectangle((x, y, x + bw, y + bh), outline=(0, 255, 0), width=1)
            draw.text((x, max(0.0, y - 10.0)), f"cat={int(inst['category_id'])}", fill=(255, 255, 0))
        overlay_boxed = np.asarray(overlay_pil, dtype=np.uint8)

        union_mask_rgb = np.repeat(union_mask[..., None], 3, axis=2)
        return np.concatenate([image, overlay_boxed, union_mask_rgb], axis=1)

    backend = matplotlib.get_backend().lower()
    headless = backend.endswith("agg") or not os.environ.get("DISPLAY")
    if headless:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        print(
            f"Headless mode detected (backend={matplotlib.get_backend()}, DISPLAY={os.environ.get('DISPLAY')!r})."
        )
        print(f"Saving frames under: {out_dir_path.resolve()}")
        num_saved = 0
        idx_to_save = idx
        max_to_save = max(0, int(save_limit))
        while num_saved < max_to_save:
            if not ensure_group(idx_to_save):
                print("Reached end of dataset.")
                break
            group = groups[idx_to_save]
            preview = _render_triptych_fast(group)
            out_path = out_dir_path / f"{split}_idx_{idx_to_save:06d}_img_{group['image_id']}.png"
            Image.fromarray(preview, mode="RGB").save(out_path)
            num_saved += 1
            if num_saved % 100 == 0 or num_saved == 1:
                print(f"Saved {num_saved}/{max_to_save}: {out_path}")
            idx_to_save += step
        print(f"Done. Saved {num_saved} frame(s) to {out_dir_path.resolve()}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    def draw(index: int) -> None:
        for ax in axes:
            ax.clear()

        if not ensure_group(index):
            for ax in axes:
                ax.axis("off")
            axes[1].set_title("Reached end of dataset.")
            fig.suptitle(f"Loaded {len(groups)} image groups from split={split}", fontsize=11)
            fig.canvas.draw_idle()
            return

        group = groups[index]
        image = group["image"]
        instances = group["instances"]
        image_id = group["image_id"]
        file_name = group["file_name"]
        first_mask, union_mask, num_shape_mismatch = _group_masks(group)

        axes[0].imshow(image)
        axes[0].set_title("RGB image")
        axes[0].axis("off")

        axes[1].imshow(first_mask, cmap="gray", vmin=0, vmax=1 if first_mask.max() <= 1 else 255)
        axes[1].set_title(f"First instance mask (all instances exist separately) (H,W={first_mask.shape})")
        axes[1].axis("off")

        axes[2].imshow(union_mask, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Union of masks (same-shape only)")
        axes[2].axis("off")

        missing_ratio = float((union_mask == 0).mean()) if union_mask.size else 1.0
        image_hw = tuple(image.shape[:2])
        mask_hw = tuple(first_mask.shape[:2])
        mismatch_note = ""
        if image_hw != mask_hw:
            mismatch_note = f" | image_hw={image_hw} vs mask_hw={mask_hw}"
        if num_shape_mismatch:
            mismatch_note += f" | dropped_masks={num_shape_mismatch}"
        fig.suptitle(
            (
                f"{index + 1}/{max(len(groups), index + 1)} | image_id={image_id} | anns={len(instances)} | "
                f"uncovered_px={missing_ratio:.3f}{mismatch_note}\n{file_name}"
            ),
            fontsize=11,
        )
        fig.canvas.draw_idle()

    def save_current_frame(index: int, out_dir: Path) -> Path | None:
        if not ensure_group(index):
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        group = groups[index]
        image_id = group["image_id"]
        out_path = out_dir / f"{split}_idx_{index:06d}_img_{image_id}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        return out_path

    def on_key(event) -> None:
        nonlocal idx
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key in ("n", "right", " "):
            idx = idx + step
            draw(idx)
            return
        if event.key in ("p", "left", "backspace"):
            idx = max(idx - step, 0)
            draw(idx)
            return
        if event.key == "home":
            idx = 0
            draw(idx)
            return
        if event.key == "end":
            while ensure_group(len(groups)):
                pass
            idx = max(0, len(groups) - 1)
            draw(idx)

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(idx)
    plt.tight_layout()

    print("Controls: n/right/space=next, p/left/backspace=prev, home/end, q=quit")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive EntitySeg instance browser from TFDS.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_TFDS_DIR)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", choices=("train", "validation", "val"), default="train")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--step", type=int, default=1, help="How many images to jump per next/prev.")
    parser.add_argument("--save_limit", type=int, default=1000, help="Headless mode: max frames to save.")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_HEADLESS_OUT, help="Headless mode output folder.")
    args = parser.parse_args()

    browse_tfds(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        start_index=args.start_index,
        step=args.step,
        save_limit=args.save_limit,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
