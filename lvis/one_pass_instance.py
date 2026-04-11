"""Stream ``lvis_instance`` TFDS (one row per LVIS instance); log + optional mask PNGs.

Run from the ``lvis`` directory::

  python3 -m one_pass_instance
"""

from lvis_giraffe import load_data, sanity_check

import numpy as np
from PIL import Image


def one_pass():
    ds_dir = "/mnt/klum/lvis/lvis_giraffe_tfds/"
    ds = load_data(
        data_dir=ds_dir,
        split="train",
        batch_size=1,
        repeat=False,
        shuffle=False,
        ignore_errors=True,
    )
    #which = {45, 127, 217, 12, 827, 943, 793, 735, 1099, 154}
    categories: dict[int, dict[str, any]] = {}
    sanity_check(ds_dir, "train", num_batches=5)
    with open("single_pass_thru_instance.txt", "w") as f:
        for idx, batch in enumerate(ds):
            image = batch["image"]
            instance = batch["instance"]
            category_id = instance["category_id"]
            metadata = batch["metadata"]
            f.write(f"{image.shape}, category_id={category_id}, {metadata}\n")

            cat = int(np.asarray(category_id))
        
            mask = instance["mask_raster"].numpy()
            mask = np.squeeze(mask, axis=0)
            mask = np.squeeze(mask, axis=-1)
            area_frac = float((mask > 0).mean())
            if area_frac > 0.01:
                if cat not in categories:
                    categories[cat] = {"count": 0, "name": str(np.asarray(instance["category_name"]))}
                categories[cat]["count"] += 1

            
    with open("categories.txt", "w") as out:
        print(categories)
        for key, value in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
            count = value["count"]
            name = value["name"]
            out.write(f"{key}: {name} {count}\n")


if __name__ == "__main__":
    one_pass()
