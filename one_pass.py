from d3.d3_normal import load_data, sanity_check
import matplotlib.pyplot as plt
import numpy as np


def one_pass():
    ds_dir = "gs://kmh-gcp-us-central2/kristine/lvm/d3_normal_tfds"

    # d3_normal: train skips sorted index % 10 == 0; validation is only those rows. Your file may exist only in validation.
    sanity_check(ds_dir, "train", num_batches=1, batch_size=8)

    target_substring = "train_ai_007_010_cam_00_fr0084.png"

    for split in ("train", "validation"):
        ds = load_data(data_dir=ds_dir, split=split, batch_size=1, repeat=False, shuffle=False)
        for i, batch in enumerate(ds):
            if i > 500_000:
                break
            p = batch["metadata"]["rgb_path"][0]
            if hasattr(p, "numpy"):
                p = p.numpy()
            if isinstance(p, (bytes, np.bytes_)):
                p = p.decode("utf-8", errors="replace")
            else:
                p = str(p)
            s = batch["metadata"]["stem"][0]
            if hasattr(s, "numpy"):
                s = s.numpy()
            if isinstance(s, (bytes, np.bytes_)):
                s = s.decode("utf-8", errors="replace")
            else:
                s = str(s)
            if target_substring not in p and target_substring not in s:
                continue
            print(f"Found in split={split!r} path={p!r} stem={s!r}")
            rgb = batch["image"].numpy()[0]
            mask = batch["mask"].numpy()[0]
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].imshow(np.clip(rgb, 0, 255).astype(np.uint8))
            ax[0].axis("off")
            ax[1].imshow(np.clip(mask.astype(np.float32) / 255.0, 0, 1))
            ax[1].axis("off")
            plt.tight_layout()
            plt.savefig("one_pass_normal.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved one_pass_normal.png")
            sanity_check(ds_dir, split, num_batches=1, batch_size=8)
            return

    print("Not found in train or validation. Confirm the file is under rgb/ in the source tree used to build this TFDS, and that normal/<stem>.h5 exists.")
    sanity_check(ds_dir, "train", num_batches=1, batch_size=8)


if __name__ == "__main__":
    one_pass()
