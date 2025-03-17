from entity_seg_instance import load_data, sanity_check

import numpy as np
from PIL import Image

def one_pass():
    ds_dir = '/mnt/klum/entityseg/entityseg_instance_tfds/'
    ds = load_data(
        data_dir=ds_dir,
        split='train',
        batch_size=1,
        repeat=False,
        shuffle=False,
        ignore_errors=True,
    )
    categories = {}
    #TODO: add back later
    #sanity_check(ds_dir, 'train', num_batches=5, ignore_errors=True)
    with open('single_pass_thru_instance.txt', 'w') as f:
        for idx, batch in enumerate(ds):
            image = batch["image"]
            instance = batch["instance"]
            category_id = instance["category_id"]
            metadata = batch["metadata"]
            f.write(f"{image.shape}, category_id={category_id}, {metadata}\n")

            cat = int(np.asarray(category_id))

            mask = instance["mask_raster"].numpy()
            mask = np.squeeze(mask, axis=0)  # drop batch dim when batch_size==1
            mask = np.squeeze(mask, axis=-1)  # H×W
            area_frac = float((mask > 0).mean())
            if cat == 17 and area_frac > 0.01:
                mask_u8 = (mask.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(mask_u8, mode="L").save(f"instance_mask_{idx:06d}.png")

    with open("categories.txt", "w") as f:
        for category_id, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{category_id}: {count}\n")


if __name__ == "__main__":
    one_pass()
