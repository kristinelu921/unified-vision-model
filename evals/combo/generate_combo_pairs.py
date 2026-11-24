#!/usr/bin/env python3
"""Generate combined dense-task image pairs from a normalized manifest.

Input manifest JSON:
{
  "records": [
    {
      "id": "scene_frame",
      "image": "/path/rgb.png",
      "depth": "/path/depth.npy|png|hdf5",
      "normal": "/path/normal.npy|png|hdf5",
      "instances": [
        {
          "id": "chair_1",
          "category": "chair",
          "mask": "/path/mask.png|npy|hdf5",
          "bbox": [x, y, w, h]
        }
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SEGMENT_COLOR = np.array([255, 0, 0], dtype=np.uint8)
BOX_COLOR = (57, 255, 20)
NAMED_COLORS = {
    "red": np.array([255, 0, 0], dtype=np.uint8),
    "green": np.array([0, 255, 0], dtype=np.uint8),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_array(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        return data[key or data.files[0]]
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return np.asarray(Image.open(path))
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            if key is None:
                key = next(iter(f.keys()))
            return np.asarray(f[key])
    raise ValueError(f"Unsupported array file: {path}")


def resize_to(arr: np.ndarray, size_hw: tuple[int, int], nearest: bool = False) -> np.ndarray:
    h, w = size_hw
    if arr.shape[:2] == (h, w):
        return arr
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr).resize((w, h), mode))
    return np.asarray(Image.fromarray(arr).resize((w, h), mode))


def visualize_depth(depth: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    depth = np.squeeze(depth).astype(np.float32)
    depth = resize_to(depth, size_hw)
    valid = np.isfinite(depth) & (depth > 0)
    out = np.zeros(depth.shape, dtype=np.float32)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (out * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def visualize_normal(normal: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    normal_is_uint8 = normal.dtype == np.uint8
    normal = normal.astype(np.float32)
    if normal.ndim == 2:
        normal = np.stack([normal, normal, normal], axis=-1)
    if normal.shape[-1] > 3:
        normal = normal[..., :3]
    normal = resize_to(normal, size_hw)
    if not normal_is_uint8:
        finite = np.isfinite(normal)
        normal = np.where(finite, normal, 0.0)
        if normal.min() < -0.05:
            normal = (normal + 1.0) * 0.5
        normal = np.clip(normal, 0.0, 1.0) * 255.0
    return normal.astype(np.uint8)


def read_mask(instance: dict[str, Any], size_hw: tuple[int, int]) -> np.ndarray:
    mask = read_array(instance["mask"])
    mask = np.squeeze(mask)
    mask = resize_to(mask, size_hw, nearest=True)
    return mask > 0


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def box_mask(bbox: list[float], size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    x, y, bw, bh = [int(round(v)) for v in bbox]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def choose_roles(record: dict[str, Any], size_hw: tuple[int, int]) -> dict[str, dict[str, Any]]:
    instances = []
    for inst in record.get("instances", []):
        mask = read_mask(inst, size_hw)
        if mask.sum() == 0:
            continue
        item = dict(inst)
        item["_mask"] = mask
        item["_bbox"] = inst.get("bbox") or bbox_from_mask(mask)
        instances.append(item)

    if not instances:
        raise ValueError(f"{record.get('id')} needs at least one visible instance")

    primary = instances[0]
    secondary = next(
        (inst for inst in instances[1:] if inst.get("category") != primary.get("category")),
        instances[1] if len(instances) > 1 else primary,
    )

    primary_group = [inst for inst in instances if inst.get("category") == primary.get("category")]
    secondary_group = [inst for inst in instances if inst.get("category") == secondary.get("category")]

    return {
        "primary": primary,
        "secondary": secondary,
        "primary_group": primary_group,
        "secondary_group": secondary_group,
    }


def combine_region_masks(
    region: str,
    instances: list[dict[str, Any]],
    size_hw: tuple[int, int],
) -> np.ndarray:
    if not instances:
        return np.zeros(size_hw, dtype=bool)

    mask = np.zeros(size_hw, dtype=bool)
    bmask = np.zeros(size_hw, dtype=bool)
    for instance in instances:
        mask |= instance["_mask"]
        bmask |= box_mask(instance["_bbox"], size_hw)
    if region == "mask":
        return mask
    if region == "outside_mask":
        return ~mask
    if region == "full_image":
        return np.ones(size_hw, dtype=bool)
    if region == "box":
        return bmask
    if region == "outside_box":
        return ~bmask
    if region == "box_minus_mask":
        return bmask & ~mask
    raise ValueError(f"Unsupported region: {region}")


def apply_layer(
    canvas: np.ndarray,
    layer: dict[str, Any],
    roles: dict[str, Any],
    depth_vis: np.ndarray,
    normal_vis: np.ndarray,
) -> np.ndarray:
    out = canvas.copy()
    role_name = layer["role"]
    inst = roles[role_name]
    group = roles[f"{role_name}_group"]
    mask = combine_region_masks(layer["region"], group, canvas.shape[:2])
    layer_type = layer["type"]

    if layer_type == "depth":
        out[mask] = depth_vis[mask]
    elif layer_type == "normal":
        out[mask] = normal_vis[mask]
    elif layer_type == "segment":
        blended = (0.55 * out[mask] + 0.45 * SEGMENT_COLOR).astype(np.uint8)
        out[mask] = blended
    elif layer_type == "color":
        color = layer["color"]
        if isinstance(color, str):
            color = NAMED_COLORS[color]
        color = np.asarray(color, dtype=np.float32)
        alpha = layer.get("alpha")
        if alpha is None:
            out[mask] = color.astype(np.uint8)
        else:
            alpha = float(alpha)
            blended = (1.0 - alpha) * out[mask].astype(np.float32) + alpha * color
            out[mask] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    elif layer_type == "box":
        image = Image.fromarray(out)
        draw = ImageDraw.Draw(image)
        for group_inst in group:
            x, y, w, h = [int(round(v)) for v in group_inst["_bbox"]]
            draw.rectangle([x, y, x + w, y + h], outline=BOX_COLOR, width=4)
        out = np.asarray(image)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
    return out


def generate_record(record: dict[str, Any], tasks: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    rgb = read_image(record["image"])
    size_hw = rgb.shape[:2]
    depth_vis = visualize_depth(read_array(record["depth"], record.get("depth_key")), size_hw)
    normal_vis = visualize_normal(read_array(record["normal"], record.get("normal_key")), size_hw)
    roles = choose_roles(record, size_hw)

    record_dir = out_dir / str(record["id"])
    record_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        transformed = rgb.copy()
        for layer in task["layers"]:
            transformed = apply_layer(transformed, layer, roles, depth_vis, normal_vis)

        base_path = record_dir / f"{task['id']}_base.png"
        transformed_path = record_dir / f"{task['id']}_transformed.png"
        meta_path = record_dir / f"{task['id']}_meta.json"
        Image.fromarray(rgb).save(base_path)
        Image.fromarray(transformed).save(transformed_path)

        meta = {
            "record_id": record["id"],
            "task_id": task["id"],
            "description": task.get("description", ""),
            "roles": {
                name: {
                    "id": inst.get("id"),
                    "category": inst.get("category"),
                    "bbox": inst.get("_bbox"),
                }
                for name, inst in roles.items()
                if not name.endswith("_group")
            },
            "role_groups": {
                name: [
                    {
                        "id": inst.get("id"),
                        "category": inst.get("category"),
                        "bbox": inst.get("_bbox"),
                    }
                    for inst in group
                ]
                for name, group in roles.items()
                if name.endswith("_group")
            },
            "base": str(base_path),
            "transformed": str(transformed_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        rows.append(meta)
    return rows


def generate_record_for_pool(payload: tuple[dict[str, Any], list[dict[str, Any]], str, bool]) -> tuple[list[dict[str, Any]], str | None]:
    record, tasks, out_dir, skip_errors = payload
    try:
        return generate_record(record, tasks, Path(out_dir)), None
    except Exception as exc:
        if not skip_errors:
            raise
        return [], f"Skipping {record.get('id')}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tasks", default=Path(__file__).with_name("selected_9_tasks.json"), type=Path)
    parser.add_argument("--out", default=Path(__file__).with_name("samples"), type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    tasks = read_json(args.tasks)
    records = manifest["records"] if isinstance(manifest, dict) else manifest
    if args.limit:
        records = records[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    out_manifest = args.out / "manifest.jsonl"
    n = 0
    with out_manifest.open("w", encoding="utf-8") as f:
        if args.jobs > 1:
            payloads = ((record, tasks, str(args.out), args.skip_errors) for record in records)
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                iterator = executor.map(generate_record_for_pool, payloads)
                for rows, error in iterator:
                    if error:
                        print(error, file=sys.stderr)
                    for row in rows:
                        f.write(json.dumps(row) + "\n")
                        n += 1
            print(f"Wrote {n} pairs to {args.out}")
            return

        for record in records:
            rows, error = generate_record_for_pool((record, tasks, str(args.out), args.skip_errors))
            if error:
                print(error, file=sys.stderr)
            for row in rows:
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"Wrote {n} pairs to {args.out}")


if __name__ == "__main__":
    main()
