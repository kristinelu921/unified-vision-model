#!/usr/bin/env python3
"""Create a contact sheet for one vkitti2_combo_grouped9 TFDS sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds


def decode_text(value) -> str:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/klum/vkitti2_combo_grouped9_tfds")
    parser.add_argument("--split", default="car_tree__task_depth_primary_full")
    parser.add_argument("--out-prefix", default="one_pass_vkitti2_grouped9")
    args = parser.parse_args()

    ds = tfds.load("vkitti2_combo_grouped9", split=args.split, data_dir=args.data_dir)
    ex = next(iter(ds))
    base = ex["base_images"].numpy()
    transformed = ex["transformed_images"].numpy()
    primary_masks = ex["primary_masks"].numpy()
    secondary_masks = ex["secondary_masks"].numpy()
    group_meta = ex["group_metadata"]
    example_meta = ex["example_metadata"]

    fig, axes = plt.subplots(3, 9, figsize=(27, 9))
    for i in range(9):
        axes[0, i].imshow(base[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"{decode_text(example_meta['scene'][i])}\n"
            f"{decode_text(example_meta['variation'][i])} "
            f"{decode_text(example_meta['camera'][i])}",
            fontsize=8,
        )

        axes[1, i].imshow(transformed[i])
        axes[1, i].axis("off")
        axes[1, i].set_title(f"transformed {i + 1}", fontsize=9)

        primary = primary_masks[i].squeeze() > 0
        secondary = secondary_masks[i].squeeze() > 0
        overlay = np.zeros((*primary.shape, 3), dtype=np.uint8)
        overlay[..., 0] = primary * 255
        overlay[..., 1] = secondary * 255
        axes[2, i].imshow(overlay)
        axes[2, i].axis("off")
        axes[2, i].set_title("mask A/B", fontsize=9)

    fig.suptitle(
        (
            f"{decode_text(group_meta['task_split_name'])}\n"
            f"{decode_text(group_meta['primary_category'])} + "
            f"{decode_text(group_meta['secondary_category'])}, "
            f"group_size={int(group_meta['group_size'].numpy())}, "
            f"official_scenes={int(group_meta['official_scene_count'].numpy())}, "
            f"contexts={int(group_meta['scene_context_count'].numpy())}"
        ),
        fontsize=14,
    )
    plt.tight_layout()
    image_out = f"{args.out_prefix}.png"
    plt.savefig(image_out, dpi=140, bbox_inches="tight")
    plt.close()

    payload = {
        "source": "tfds",
        "dataset": "vkitti2_combo_grouped9",
        "data_dir": args.data_dir,
        "split": args.split,
        "group_metadata": {
            key: decode_text(value)
            for key, value in group_meta.items()
            if key not in {"group_size", "official_scene_count", "scene_context_count"}
        },
        "group_size": int(group_meta["group_size"].numpy()),
        "official_scene_count": int(group_meta["official_scene_count"].numpy()),
        "scene_context_count": int(group_meta["scene_context_count"].numpy()),
        "examples": [],
    }
    for i in range(9):
        payload["examples"].append(
            {
                "index": i,
                "record_id": decode_text(example_meta["record_id"][i]),
                "scene": decode_text(example_meta["scene"][i]),
                "variation": decode_text(example_meta["variation"][i]),
                "camera": decode_text(example_meta["camera"][i]),
                "frame": decode_text(example_meta["frame"][i]),
                "scene_context": decode_text(example_meta["scene_context"][i]),
                "rel_base_path": decode_text(example_meta["rel_base_path"][i]),
                "rel_transformed_path": decode_text(example_meta["rel_transformed_path"][i]),
                "annotation": json.loads(decode_text(example_meta["annotation_json"][i])),
            }
        )
    meta_out = f"{args.out_prefix}.json"
    Path(meta_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {image_out}")
    print(f"Saved {meta_out}")


if __name__ == "__main__":
    main()
