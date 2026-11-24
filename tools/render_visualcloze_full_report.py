#!/usr/bin/env python3
"""Render representative VisualCloze samples and tabular score exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CONDITIONS = ("real", "scrambled", "grey", "oracle")
DISPLAY_CONDITIONS = ("Real", "Scrambled", "Grey", "Oracle")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def _pretty(task: str) -> str:
    return task.replace("_", " ").title()


def _json_dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _representative_queries(scores: dict, task_order: list[str]) -> dict[str, dict]:
    rows = scores["results"]
    selected = {}
    for task in task_order:
        candidates = [
            row
            for row in rows
            if row["task"] == task
            and row["condition"] == "real"
            and math.isfinite(float(row["primary_value"]))
        ]
        if not candidates:
            raise RuntimeError(f"no finite real-condition score for {task}")
        values = np.asarray([row["primary_value"] for row in candidates], np.float64)
        median = float(np.median(values))
        chosen = min(candidates, key=lambda row: (abs(float(row["primary_value"]) - median), int(row["q"])))
        selected[task] = {
            "q": int(chosen["q"]),
            "episode_id": chosen["episode_id"],
            "primary_metric": chosen["primary_metric"],
            "primary_value": float(chosen["primary_value"]),
            "finite_real_records": len(candidates),
            "selection": "finite real-condition score closest to the task median",
        }
    return selected


def _render_sheets(
    run_dir: Path,
    out_dir: Path,
    task_order: list[str],
    selected: dict[str, dict],
    episodes: dict[tuple[str, int, str], dict],
    inference: dict[str, dict],
    *,
    tasks_per_sheet: int,
) -> list[Path]:
    tile = 144
    cell_w = 160
    row_h = 190
    label_w = 218
    header_h = 54
    columns = ("Query input", "Target", *DISPLAY_CONDITIONS)
    width = label_w + len(columns) * cell_w
    sheets = []
    for sheet_index, start in enumerate(range(0, len(task_order), tasks_per_sheet), start=1):
        tasks = task_order[start : start + tasks_per_sheet]
        canvas = Image.new("RGB", (width, header_h + row_h * len(tasks)), (247, 248, 250))
        draw = ImageDraw.Draw(canvas)
        header_font = _font(17, bold=True)
        task_font = _font(17, bold=True)
        detail_font = _font(13)
        for col, title in enumerate(columns):
            x = label_w + col * cell_w + cell_w // 2
            box = draw.textbbox((0, 0), title, font=header_font)
            draw.text((x - (box[2] - box[0]) / 2, 16), title, fill=(27, 31, 38), font=header_font)
        draw.line((0, header_h - 1, width, header_h - 1), fill=(190, 195, 204), width=1)
        for row_index, task in enumerate(tasks):
            info = selected[task]
            q = info["q"]
            y0 = header_h + row_index * row_h
            draw.rectangle((0, y0, width, y0 + row_h), fill=(255, 255, 255))
            if row_index:
                draw.line((0, y0, width, y0), fill=(218, 222, 229), width=1)
            draw.text((14, y0 + 48), _pretty(task), fill=(24, 28, 35), font=task_font)
            draw.text((14, y0 + 76), f"query {q:02d}", fill=(82, 89, 101), font=detail_font)
            draw.text(
                (14, y0 + 98),
                f"{info['primary_metric']} = {info['primary_value']:.4f}",
                fill=(82, 89, 101),
                font=detail_font,
            )
            real_ep = episodes[(task, q, "real")]
            with np.load(run_dir / real_ep["file"]) as packed:
                query_input = Image.fromarray(np.asarray(packed["query_input"], np.uint8), "RGB")
                query_target = Image.fromarray(np.asarray(packed["query_target"], np.uint8), "RGB")
            images = [query_input, query_target]
            for condition in CONDITIONS:
                ep = episodes[(task, q, condition)]
                generated = inference[ep["episode_id"]]
                with Image.open(run_dir / generated["prediction"]) as pred:
                    images.append(pred.convert("RGB").copy())
            for col, image in enumerate(images):
                x = label_w + col * cell_w + (cell_w - tile) // 2
                y = y0 + 16
                resized = image.resize((tile, tile), Image.Resampling.LANCZOS)
                canvas.paste(resized, (x, y))
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 190, 199), width=1)
        path = out_dir / f"samples-{sheet_index:02d}.png"
        canvas.save(path, optimize=True)
        sheets.append(path)
    return sheets


def _write_tables(out_dir: Path, scores: dict, task_order: list[str], *, title: str) -> None:
    fields = [
        "task",
        "primary_metric",
        "direction",
        "condition",
        "mean",
        "ci95_low",
        "ci95_high",
        "n",
        "n_nan",
        "copy_input_mean",
        "perfect_target_mean",
    ]
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in task_order:
            for condition in CONDITIONS:
                cell = scores["cells"][f"{task}@{condition}"]
                writer.writerow(
                    {
                        "task": task,
                        "primary_metric": cell["primary_metric"],
                        "direction": "lower" if cell["primary_lower_is_better"] else "higher",
                        "condition": condition,
                        "mean": f"{cell['mean']:.8f}",
                        "ci95_low": f"{cell['ci95'][0]:.8f}",
                        "ci95_high": f"{cell['ci95'][1]:.8f}",
                        "n": cell["n"],
                        "n_nan": cell["n_nan"],
                        "copy_input_mean": f"{cell['copy_input_mean']:.8f}",
                        "perfect_target_mean": f"{cell['perfect_target_mean']:.8f}",
                    }
                )

    lines = [
        f"# {title}",
        "",
        "Each task uses the primary metric declared by its evaluation contract.",
        "Each value is mean [95% bootstrap CI].",
        "",
        "| Task | Metric | Real | Scrambled | Grey | Oracle | Direction |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task in task_order:
        cells = [scores["cells"][f"{task}@{condition}"] for condition in CONDITIONS]
        values = [f"{cell['mean']:.4f} [{cell['ci95'][0]:.4f}, {cell['ci95'][1]:.4f}]" for cell in cells]
        direction = "lower" if cells[0]["primary_lower_is_better"] else "higher"
        lines.append(
            f"| {_pretty(task)} | `{cells[0]['primary_metric']}` | "
            + " | ".join(values)
            + f" | {direction} |"
        )
    lines.append("")
    finite_counts = {
        (task, condition): scores["cells"][f"{task}@{condition}"]["n"]
        for task in task_order
        for condition in CONDITIONS
    }
    if len(set(finite_counts.values())) == 1:
        lines.append(f"Every cell has n={next(iter(finite_counts.values()))} finite records.")
    else:
        lines.append("Finite record counts vary by task/condition; see `results.csv` for each cell.")
    lines.append("")
    (out_dir / "results.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores-file", default="scores_perceptual.json")
    parser.add_argument("--expected-tasks", type=int, default=15)
    parser.add_argument("--tasks-per-sheet", type=int, default=3)
    parser.add_argument("--title", default="VisualCloze full-pack results")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    inference_doc = json.loads((run_dir / "inference.json").read_text())
    scores = json.loads((run_dir / args.scores_file).read_text())
    task_order = list(dict.fromkeys(ep["task"] for ep in manifest["episodes"]))
    if len(task_order) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} tasks, found {len(task_order)}")
    if args.tasks_per_sheet < 1:
        raise RuntimeError("--tasks-per-sheet must be positive")
    episodes = {(ep["task"], int(ep["q"]), ep["condition"]): ep for ep in manifest["episodes"]}
    inference = {row["episode_id"]: row for row in inference_doc["results"]}
    selected = _representative_queries(scores, task_order)
    sheets = _render_sheets(
        run_dir,
        out_dir,
        task_order,
        selected,
        episodes,
        inference,
        tasks_per_sheet=args.tasks_per_sheet,
    )
    _write_tables(out_dir, scores, task_order, title=args.title)
    _json_dump(
        out_dir / "samples.json",
        {
            "selection_method": "finite real-condition primary score closest to each task median",
            "score_report": str((run_dir / args.scores_file).relative_to(run_dir)),
            "tasks": selected,
            "sheets": [path.name for path in sheets],
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "sheets": len(sheets), "tasks": len(task_order)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
