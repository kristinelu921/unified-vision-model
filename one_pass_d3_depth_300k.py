import os

import matplotlib.pyplot as plt
import numpy as np

from d3.d3_depth_300k import load_data, sanity_check


def one_pass():
    ds_dir = os.environ.get("D3_DEPTH_300K_TFDS_DIR", "/tmp/d3_depth_300k_smoke")
    split = os.environ.get("D3_DEPTH_300K_SPLIT", "train")

    ds = load_data(data_dir=ds_dir, split=split, batch_size=1, repeat=False, shuffle=False)
    batch = next(iter(ds))

    rgb = batch["image"].numpy()[0]
    depth = batch["mask"].numpy()[0]
    depth_metric = batch["depth_metric"].numpy()[0]
    rgb_path = batch["metadata"]["rgb_path"][0].numpy().decode("utf-8")
    depth_path = batch["metadata"]["depth_path"][0].numpy().decode("utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].imshow(np.clip(rgb, 0, 255).astype(np.uint8))
    ax[0].set_title("RGB")
    ax[0].axis("off")
    depth_vis = np.clip(depth.astype(np.float32) / 255.0, 0.0, 1.0)
    ax[1].imshow(depth_vis[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
    ax[1].set_title("depth mask")
    ax[1].axis("off")
    metric = depth_metric[..., 0].astype(np.float32)
    metric_vis = np.ma.masked_where(metric <= 0, metric)
    valid_metric = metric[metric > 0]
    if valid_metric.size:
        lo, hi = np.percentile(valid_metric, [1.0, 99.0])
    else:
        lo, hi = 0.0, 1.0
    ax[2].imshow(metric_vis, cmap="gray", vmin=lo, vmax=hi)
    ax[2].set_title("depth metric")
    ax[2].axis("off")
    plt.tight_layout()
    out = "one_pass_d3_depth_300k.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")
    print(f"rgb_path={rgb_path}")
    print(f"depth_path={depth_path}")
    sanity_check(ds_dir, split, num_batches=1, batch_size=4)


if __name__ == "__main__":
    one_pass()
