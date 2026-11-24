import os

import matplotlib.pyplot as plt
import numpy as np

from d3.d3_normal_300k import load_data, sanity_check


def one_pass():
    ds_dir = os.environ.get("D3_NORMAL_300K_TFDS_DIR", "/tmp/d3_normal_300k_smoke")
    split = os.environ.get("D3_NORMAL_300K_SPLIT", "train")

    ds = load_data(data_dir=ds_dir, split=split, batch_size=1, repeat=False, shuffle=False)
    batch = next(iter(ds))

    rgb = batch["image"].numpy()[0]
    normal = batch["mask"].numpy()[0]
    rgb_path = batch["metadata"]["rgb_path"][0].numpy().decode("utf-8")
    normal_path = batch["metadata"]["normal_path"][0].numpy().decode("utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(np.clip(rgb, 0, 255).astype(np.uint8))
    ax[0].set_title("RGB")
    ax[0].axis("off")
    ax[1].imshow(np.clip(normal.astype(np.float32) / 255.0, 0.0, 1.0))
    ax[1].set_title("normal")
    ax[1].axis("off")
    plt.tight_layout()
    out = "one_pass_d3_normal_300k.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")
    print(f"rgb_path={rgb_path}")
    print(f"normal_path={normal_path}")
    sanity_check(ds_dir, split, num_batches=1, batch_size=4)


if __name__ == "__main__":
    one_pass()
