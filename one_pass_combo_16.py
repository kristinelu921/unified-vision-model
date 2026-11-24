import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evals.combo.generate_combo_pairs import (
    apply_layer,
    choose_roles,
    read_array,
    read_image,
    read_json,
    visualize_depth,
    visualize_normal,
)


DEFAULT_TASKS_PATH = Path("evals/combo/selected_16_tasks.json")
DEFAULT_TFDS_DIR = "/mnt/klum/combo_hypersim_tfds_roles_v12"


def _decode_text(x) -> str:
    if hasattr(x, "numpy"):
        x = x.numpy()
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def _task_order(tasks_path: Path) -> dict[str, int]:
    tasks = read_json(tasks_path)
    return {task["id"]: i for i, task in enumerate(tasks)}


def _find_record(records, record_id: str | None):
    if record_id is None:
        return records[0]
    for record in records:
        if record["id"] == record_id:
            return record
    raise ValueError(f"record_id {record_id!r} not found")


def _decode_sequence_dict(seq_dict):
    out = []
    n = int(_to_numpy(seq_dict["bbox_xywh"]).shape[0])
    for i in range(n):
        out.append(
            {
                "instance_id": _decode_text(seq_dict["instance_id"][i]),
                "category": _decode_text(seq_dict["category"][i]),
                "area": int(_to_numpy(seq_dict["area"][i])),
                "bbox_xywh": _to_numpy(seq_dict["bbox_xywh"][i]).tolist(),
                "rel_mask_path": _decode_text(seq_dict["rel_mask_path"][i]),
            }
        )
    return out


def build_all_transforms_from_manifest(manifest_path: Path, tasks_path: Path, record_id: str | None):
    manifest = read_json(manifest_path)
    tasks = read_json(tasks_path)
    records = manifest["records"] if isinstance(manifest, dict) else manifest
    record = _find_record(records, record_id)

    rgb = read_image(record["image"])
    size_hw = rgb.shape[:2]
    depth_vis = visualize_depth(read_array(record["depth"], record.get("depth_key")), size_hw)
    normal_vis = visualize_normal(read_array(record["normal"], record.get("normal_key")), size_hw)
    roles = choose_roles(record, size_hw)

    outputs = []
    for task in tasks:
        transformed = rgb.copy()
        for layer in task["layers"]:
            transformed = apply_layer(transformed, layer, roles, depth_vis, normal_vis)
        outputs.append(
            {
                "task_id": task["id"],
                "description": task.get("description", ""),
                "image": transformed,
            }
        )

    payload = {
        "source": "manifest",
        "record_id": record["id"],
        "scene": record.get("scene", ""),
        "camera": record.get("camera", ""),
        "frame": record.get("frame", ""),
        "roles": {
            "primary": {
                "id": roles["primary"]["id"],
                "category": roles["primary"]["category"],
                "bbox": roles["primary"]["_bbox"],
            },
            "secondary": {
                "id": roles["secondary"]["id"],
                "category": roles["secondary"]["category"],
                "bbox": roles["secondary"]["_bbox"],
            },
        },
        "role_groups": {
            "primary_group": [
                {"id": inst["id"], "category": inst["category"], "bbox": inst["_bbox"]}
                for inst in roles["primary_group"]
            ],
            "secondary_group": [
                {"id": inst["id"], "category": inst["category"], "bbox": inst["_bbox"]}
                for inst in roles["secondary_group"]
            ],
        },
        "task_ids": [output["task_id"] for output in outputs],
    }
    return record["id"], rgb, outputs, payload


def build_all_transforms_from_tfds(
    data_dir: str,
    split: str,
    tasks_path: Path,
    record_id: str | None,
):
    import tensorflow_datasets as tfds

    ds = tfds.load("hypersim_combo_pairs", split=split, data_dir=data_dir)
    order = _task_order(tasks_path)

    chosen_record_id = record_id
    collected = {}
    payload = None

    for ex in ds:
        ex_record_id = _decode_text(ex["metadata"]["record_id"])
        if chosen_record_id is None:
            chosen_record_id = ex_record_id
        if ex_record_id != chosen_record_id:
            continue

        task_id = _decode_text(ex["metadata"]["task_id"])
        collected[task_id] = {
            "task_id": task_id,
            "description": _decode_text(ex["metadata"]["task_description"]),
            "image": _to_numpy(ex["transformed_image"]),
        }

        if payload is None:
            payload = {
                "source": "tfds",
                "data_dir": data_dir,
                "split": split,
                "record_id": ex_record_id,
                "metadata": {k: _decode_text(v) for k, v in ex["metadata"].items()},
                "roles": {
                    "primary": {
                        "instance_id": _decode_text(ex["roles"]["primary"]["instance_id"]),
                        "category": _decode_text(ex["roles"]["primary"]["category"]),
                        "bbox_xywh": _to_numpy(ex["roles"]["primary"]["bbox_xywh"]).tolist(),
                    },
                    "secondary": {
                        "instance_id": _decode_text(ex["roles"]["secondary"]["instance_id"]),
                        "category": _decode_text(ex["roles"]["secondary"]["category"]),
                        "bbox_xywh": _to_numpy(ex["roles"]["secondary"]["bbox_xywh"]).tolist(),
                    },
                },
                "all_primary_instances": _decode_sequence_dict(ex["all_primary_instances"]),
                "all_secondary_instances": _decode_sequence_dict(ex["all_secondary_instances"]),
            }

        if len(collected) == len(order):
            break

    if not collected:
        raise ValueError(f"No TFDS samples found for split={split!r}, record_id={record_id!r}.")

    outputs = [collected[task_id] for task_id in sorted(collected, key=lambda t: order[t])]
    missing = [task_id for task_id in order if task_id not in collected]
    payload["task_ids"] = [output["task_id"] for output in outputs]
    payload["missing_task_ids"] = missing
    base = _to_numpy(ex["base_image"])
    return chosen_record_id, base, outputs, payload


def save_contact_sheet(record_id: str, base: np.ndarray, outputs, out_prefix: str):
    cols = 4
    rows = 1 + int(np.ceil(len(outputs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(np.clip(base, 0, 255).astype(np.uint8))
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"base\n{record_id}", fontsize=10)

    for j in range(1, cols):
        axes[0, j].axis("off")

    for idx, output in enumerate(outputs):
        r = 1 + idx // cols
        c = idx % cols
        axes[r, c].imshow(np.clip(output["image"], 0, 255).astype(np.uint8))
        axes[r, c].axis("off")
        axes[r, c].set_title(output["task_id"], fontsize=9)

    total_slots = rows * cols
    used_slots = cols + len(outputs)
    for idx in range(used_slots, total_slots):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    image_out = f"{out_prefix}.png"
    plt.savefig(image_out, dpi=150, bbox_inches="tight")
    plt.close()
    return image_out


def save_metadata(payload, out_prefix: str):
    meta_out = f"{out_prefix}.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return meta_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data-dir", default=DEFAULT_TFDS_DIR)
    parser.add_argument("--split", default="dining_room_chair_table")
    parser.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--out-prefix", default="one_pass_combo_16")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.manifest is not None:
        record_id, base, outputs, payload = build_all_transforms_from_manifest(
            args.manifest, args.tasks, args.record_id
        )
    else:
        record_id, base, outputs, payload = build_all_transforms_from_tfds(
            args.data_dir, args.split, args.tasks, args.record_id
        )
    image_out = save_contact_sheet(record_id, base, outputs, args.out_prefix)
    meta_out = save_metadata(payload, args.out_prefix)
    print(f"Saved {image_out}")
    print(f"Saved {meta_out}")


if __name__ == "__main__":
    main()
