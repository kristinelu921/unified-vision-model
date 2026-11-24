import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from evals.tfds.hypersim_combo_pairs import load_data, sanity_check


def _decode_text(x) -> str:
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _batch_value(batch_value):
    if hasattr(batch_value, "numpy"):
        return batch_value.numpy()
    return batch_value


def _decode_sequence_dict(seq_dict):
    normalized = {}
    for key, value in seq_dict.items():
        if len(value.shape) >= 1:
            normalized[key] = value[0]
        else:
            normalized[key] = value
    out = []
    n = int(normalized["bbox_xywh"].shape[0])
    for i in range(n):
        out.append(
            {
                "instance_id": _decode_text(normalized["instance_id"][i]),
                "category": _decode_text(normalized["category"][i]),
                "area": int(_batch_value(normalized["area"][i])),
                "bbox_xywh": _batch_value(normalized["bbox_xywh"][i]).tolist(),
                "rel_mask_path": _decode_text(normalized["rel_mask_path"][i]),
            }
        )
    return out


def _draw_boxes(ax, items, color, prefix):
    for idx, item in enumerate(items):
        x, y, w, h = item["bbox_xywh"]
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
        )
        ax.text(
            x,
            max(0, y - 4),
            f"{prefix}{idx}",
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
        )


def one_pass(
    ds_dir: str,
    split: str,
    record_id: str | None,
    task_id: str | None,
    out_prefix: str,
):
    sanity_check(ds_dir, split, num_batches=1, batch_size=1)

    ds = load_data(
        data_dir=ds_dir,
        split=split,
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    for i, batch in enumerate(ds):
        metadata = batch["metadata"]
        batch_record_id = _decode_text(metadata["record_id"][0])
        batch_task_id = _decode_text(metadata["task_id"][0])

        if record_id and batch_record_id != record_id:
            continue
        if task_id and batch_task_id != task_id:
            continue

        base = batch["base_image"].numpy()[0]
        transformed = batch["transformed_image"].numpy()[0]
        all_primary_instances = _decode_sequence_dict(batch["all_primary_instances"])
        all_secondary_instances = _decode_sequence_dict(batch["all_secondary_instances"])

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(np.clip(base, 0, 255).astype(np.uint8))
        ax[0].axis("off")
        ax[0].set_title("base")
        _draw_boxes(ax[0], all_primary_instances, "lime", "P")
        _draw_boxes(ax[0], all_secondary_instances, "cyan", "S")
        ax[1].imshow(np.clip(transformed, 0, 255).astype(np.uint8))
        ax[1].axis("off")
        ax[1].set_title("transformed")
        plt.tight_layout()
        image_out = f"{out_prefix}.png"
        plt.savefig(image_out, dpi=150, bbox_inches="tight")
        plt.close()

        meta_out = f"{out_prefix}.json"
        payload = {
            "split": split,
            "record_id": batch_record_id,
            "task_id": batch_task_id,
            "metadata": {k: _decode_text(v[0]) for k, v in metadata.items()},
            "roles": {
                "primary": {
                    "instance_id": _decode_text(batch["roles"]["primary"]["instance_id"][0]),
                    "category": _decode_text(batch["roles"]["primary"]["category"][0]),
                    "bbox_xywh": _batch_value(batch["roles"]["primary"]["bbox_xywh"][0]).tolist(),
                },
                "secondary": {
                "instance_id": _decode_text(batch["roles"]["secondary"]["instance_id"][0]),
                "category": _decode_text(batch["roles"]["secondary"]["category"][0]),
                "bbox_xywh": _batch_value(batch["roles"]["secondary"]["bbox_xywh"][0]).tolist(),
                },
            },
            "all_primary_instances": all_primary_instances,
            "all_secondary_instances": all_secondary_instances,
        }
        with open(meta_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved {image_out}")
        print(f"Saved {meta_out}")
        return

    raise ValueError(
        f"No sample matched split={split!r}, record_id={record_id!r}, task_id={task_id!r}."
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/klum/combo_hypersim_tfds")
    parser.add_argument("--split", default="dining_room_chair_table")
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--out-prefix", default="one_pass_combo")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    one_pass(
        ds_dir=args.data_dir,
        split=args.split,
        record_id=args.record_id,
        task_id=args.task_id,
        out_prefix=args.out_prefix,
    )
