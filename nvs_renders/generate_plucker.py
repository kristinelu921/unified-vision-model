import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import json

def get_direction(intrinsic, extrinsic, pixels):
    """
    Ray direction in world space for each pixel.
    Assumes extrinsic is world-to-camera (w2c): p_cam = R @ p_world + t.
    Ray in camera space: d_cam = K^-1 @ [u, v, 1]^T.
    Ray in world space: d_world = R^T @ d_cam (R^T rotates camera->world).
    """
    pixel_grid = np.zeros((256, 256, 3), dtype=np.float64)
    pixel_grid[:, :, :2] = pixels
    pixel_grid[:, :, 2] = 1

    intrinsic = np.array(intrinsic, dtype=np.float64)
    extrinsic = np.array(extrinsic, dtype=np.float64)

    K = intrinsic
    R = extrinsic[:3, :3]  # w2c rotation

    # pixel_grid (256,256,3) -> (3, 256*256) for matmul
    uv_h = pixel_grid.reshape(-1, 3).T  # (3, N)
    d_cam = np.linalg.inv(K) @ uv_h  # (3, N) ray dirs in camera space
    d_world = R.T @ d_cam  # (3, N) rotate to world space
    d_world = d_world / np.linalg.norm(d_world, axis=0, keepdims=True)

    return d_world.T.reshape(256, 256, 3)


def direction_to_rgb(d):
    """convert direction to RGB code"""
    return (d + 1) / 2 * 255

def plot_image(intrinsic, extrinsic):
    """Pixel grid: u (column) and v (row) coordinates."""
    u = np.arange(256, dtype=np.float64)
    v = np.arange(256, dtype=np.float64)
    uu, vv = np.meshgrid(u, v, indexing="xy")  # uu[i,j]=j, vv[i,j]=i
    coords = np.stack([uu, vv], axis=-1)  # (256, 256, 2)
    directions = get_direction(intrinsic, extrinsic, coords)
    print("directions.shape:", directions.shape)
    rgb = direction_to_rgb(directions)
    return rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.camera_json_path, "r") as f:
        camera_data = json.load(f) #works specifically with the nvs_renders ones
    
    views = camera_data["views"]
    
    for view in views:
        intrinsic = view["intrinsics_ndc"]
        extrinsic = view["extrinsics_world_to_camera"]
        image_file = view["image_file"]

        rgb = plot_image(intrinsic, extrinsic)
        rgb = rgb.clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(os.path.join(args.output_dir, image_file.replace(".png", "_plucker.png")))