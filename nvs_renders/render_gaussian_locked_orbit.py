#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import dataclasses
from PIL import Image


def fibonacci_sphere_points(n: int, y_min: float = -1.0, y_max: float = 1.0) -> np.ndarray:
    phi = math.pi * (3.0 - math.sqrt(5.0))
    pts = np.zeros((n, 3), dtype=np.float32)
    y_min = float(np.clip(y_min, -1.0, 1.0))
    y_max = float(np.clip(y_max, -1.0, 1.0))
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    for i in range(n):
        # Uniform area sampling on a spherical band by sampling y uniformly.
        t = (i + 0.5) / n
        y = y_min + (y_max - y_min) * t
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts[i] = [math.cos(theta) * r, y, math.sin(theta) * r]
    return pts


def world_to_viewmat(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    # OpenCV camera convention.
    z = target - cam_pos
    z = z / (np.linalg.norm(z) + 1e-12)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(np.dot(z, up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(x, z)
    R = np.stack([x, y, z], axis=0)
    t = -(R @ cam_pos)
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--num_views", type=int, default=1000)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--focal_ndc", type=float, default=2.0)
    ap.add_argument("--fit_margin", type=float, default=1.15)
    ap.add_argument("--radius_percentile", type=float, default=99.5)
    ap.add_argument(
        "--min_camera_y",
        type=float,
        default=-1.0,
        help="Minimum camera y on the orbit unit sphere. Set > 0 to keep views above table.",
    )
    ap.add_argument("--max_camera_y", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--background",
        choices=["white", "black", "transparent"],
        default="white",
        help="Background for saved PNGs.",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="Render at Nx resolution and downsample to output size.",
    )
    ap.add_argument(
        "--prune_opacity_q",
        type=float,
        default=0.0,
        help="Drop splats with sigmoid(opacity) below this quantile in [0,1].",
    )
    ap.add_argument(
        "--prune_max_scale_q",
        type=float,
        default=1.0,
        help="Drop splats whose max(exp(scale)) is above this quantile in [0,1].",
    )
    args = ap.parse_args()

    # Make local uco3d dataset_utils importable.
    repo_uco3d = "/kmh-nfs-ssd-us-mount/code/kristine/lvm/uco3d"
    if os.path.isdir(repo_uco3d):
        sys.path.insert(0, repo_uco3d)

    from dataset_utils.gauss3d_rendering import render_splats_opencv
    from dataset_utils.gauss3d_utils import load_compressed_gaussians

    seq_dir = os.path.abspath(args.seq_dir)
    gs_dir = os.path.join(seq_dir, "gaussian_splats")
    out_dir = args.out_dir or os.path.join(seq_dir, f"gaussian_views_{args.num_views}_locked")
    os.makedirs(out_dir, exist_ok=True)

    splats = load_compressed_gaussians(gs_dir)
    if getattr(splats, "fg_mask", None) is not None:
        mask = splats.fg_mask.reshape(-1).bool()
        # Keep only foreground splats for rendering (not just for camera fitting).
        splats_dict = dataclasses.asdict(splats)
        for k, v in splats_dict.items():
            if torch.is_tensor(v) and v.shape[0] == mask.shape[0]:
                splats_dict[k] = v[mask]
        splats = type(splats)(**splats_dict)
    else:
        mask = torch.ones(splats.means.shape[0], dtype=torch.bool)

    if args.prune_opacity_q > 0.0 or args.prune_max_scale_q < 1.0:
        keep = torch.ones(splats.means.shape[0], dtype=torch.bool)
        if args.prune_opacity_q > 0.0:
            alphas = torch.sigmoid(splats.opacities.reshape(-1))
            a_thr = torch.quantile(alphas, float(args.prune_opacity_q))
            keep &= alphas >= a_thr
        if args.prune_max_scale_q < 1.0:
            max_scale = torch.exp(splats.scales).max(dim=1).values
            s_thr = torch.quantile(max_scale, float(args.prune_max_scale_q))
            keep &= max_scale <= s_thr
        splats_dict = dataclasses.asdict(splats)
        for k, v in splats_dict.items():
            if torch.is_tensor(v) and v.shape[0] == keep.shape[0]:
                splats_dict[k] = v[keep]
        splats = type(splats)(**splats_dict)

    means = splats.means.cpu().numpy()
    center = np.median(means, axis=0)
    d = np.linalg.norm(means - center, axis=1)
    obj_radius = float(np.percentile(d, args.radius_percentile))
    cam_radius = obj_radius * args.fit_margin * 3.0

    origin = np.zeros(3, dtype=np.float32)
    viewmats = []
    cam_positions = []
    distances = []
    for p in fibonacci_sphere_points(args.num_views, y_min=args.min_camera_y, y_max=args.max_camera_y):
        cam_pos = p * cam_radius
        cam_positions.append(cam_pos)
        distances.append(float(np.linalg.norm(cam_pos - origin)))
        viewmats.append(world_to_viewmat(cam_pos, origin))
    viewmats = torch.from_numpy(np.stack(viewmats, axis=0))

    K = torch.tensor(
        [[args.focal_ndc, 0.0, 0.0], [0.0, args.focal_ndc, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )[None].repeat(args.num_views, 1, 1)

    # Recenter splats to origin (camera target) while preserving texture/shading.
    splats.means = splats.means - torch.from_numpy(center).float()

    render_h = args.height * max(1, args.supersample)
    render_w = args.width * max(1, args.supersample)

    for start in range(0, args.num_views, max(1, args.batch_size)):
        end = min(args.num_views, start + max(1, args.batch_size))
        try:
            colors, alphas, _ = render_splats_opencv(
                viewmats=viewmats[start:end],
                camera_matrix=K[start:end],
                splats=splats,
                render_size=(render_h, render_w),
                device=args.device,
                near_plane=0.01,
                camera_matrix_in_ndc=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Gaussian rendering failed. This script requires CUDA-enabled gsplat "
                "and a working GPU device (e.g. --device cuda:0)."
            ) from e

        colors_np = colors.clamp(0, 1).cpu().numpy()
        alpha_np = alphas.clamp(0, 1).cpu().numpy()
        if alpha_np.ndim == 3:
            alpha_np = alpha_np[..., None]
        elif alpha_np.ndim == 4 and alpha_np.shape[-1] == 1:
            pass
        else:
            raise RuntimeError(f"Unexpected alpha shape: {alpha_np.shape}")
        if args.background == "transparent":
            imgs = np.concatenate([colors_np, alpha_np], axis=-1)
            imgs = (imgs * 255.0).astype(np.uint8)
        else:
            bg = 1.0 if args.background == "white" else 0.0
            colors_np = colors_np * alpha_np + bg * (1.0 - alpha_np)
            imgs = (np.clip(colors_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        for i in range(end - start):
            img = imgs[i]
            if args.supersample > 1:
                img = np.asarray(
                    Image.fromarray(img).resize((args.width, args.height), Image.Resampling.LANCZOS)
                )
            imageio.imwrite(os.path.join(out_dir, f"view_{start + i:04d}.png"), img)

    cameras_json = {
        "num_views": args.num_views,
        "image_width": args.width,
        "image_height": args.height,
        "camera_matrix_convention": "ndc",
        "views": [],
    }
    K_list = K[0].cpu().numpy().tolist()
    for i in range(args.num_views):
        cameras_json["views"].append(
            {
                "view_index": i,
                "image_file": f"view_{i:04d}.png",
                "camera_position_world": np.asarray(cam_positions[i], dtype=np.float32).tolist(),
                "extrinsics_world_to_camera": viewmats[i].cpu().numpy().tolist(),
                "intrinsics_ndc": K_list,
            }
        )
    with open(os.path.join(out_dir, "camera_params.json"), "w") as f:
        json.dump(cameras_json, f)

    report = {
        "seq_dir": seq_dir,
        "gaussian_splats_dir": gs_dir,
        "num_views": args.num_views,
        "origin": [0.0, 0.0, 0.0],
        "source_center_shift_applied": center.tolist(),
        "radius_percentile": args.radius_percentile,
        "object_radius": obj_radius,
        "camera_radius": cam_radius,
        "distance_min": float(np.min(distances)),
        "distance_max": float(np.max(distances)),
        "distance_std": float(np.std(distances)),
        "camera_y_min": float(args.min_camera_y),
        "camera_y_max": float(args.max_camera_y),
        "background": args.background,
        "supersample": int(args.supersample),
        "prune_opacity_q": float(args.prune_opacity_q),
        "prune_max_scale_q": float(args.prune_max_scale_q),
        "device": args.device,
    }
    with open(os.path.join(out_dir, "camera_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("done", out_dir)


if __name__ == "__main__":
    main()
