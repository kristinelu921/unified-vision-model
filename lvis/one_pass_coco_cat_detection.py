"""One pass through COCO cat detection TFDS; log bbox stats + sanity PNGs.

Run from the ``lvis`` directory::

  python3 -m one_pass_coco_cat_detection
"""

from coco_cat_detection import load_data, sanity_check

import numpy as np


def one_pass():
    ds_dir = "/mnt/klum/lvm/coco_cat_detection_tfds/"
    ds = load_data(
        data_dir=ds_dir,
        split="train",
        batch_size=1,
        repeat=False,
        shuffle=False,
    )

    sanity_check(ds_dir, "train", num_batches=5, batch_size=4, max_show=4)

    total_boxes = 0
    total_images = 0
    box_counts: list[int] = []

    with open("single_pass_thru_coco_cat_detection.txt", "w") as f:
        for batch in ds:
            image = batch["image"]
            annotations = batch["annotations"]
            metadata = batch["metadata"]
            bboxes = annotations["bbox"].numpy()
            nb = int(bboxes.shape[1]) if len(bboxes.shape) > 2 else int(bboxes.shape[0])
            f.write(f"{image.shape}, boxes={nb}, {metadata}\n")

            total_images += 1
            total_boxes += nb
            box_counts.append(nb)

    print(f"Total images: {total_images}")
    print(f"Total boxes: {total_boxes}")
    if box_counts:
        print(f"Avg boxes/image: {np.mean(box_counts):.1f}")
        print(f"Max boxes in one image: {max(box_counts)}")
        print(f"Images with 1 box: {sum(1 for c in box_counts if c == 1)}")

    with open("coco_cat_detection_stats.txt", "w") as out:
        out.write(f"total_images={total_images}\n")
        out.write(f"total_boxes={total_boxes}\n")
        out.write(f"avg_boxes_per_image={np.mean(box_counts):.2f}\n")
        out.write(f"max_boxes_in_one_image={max(box_counts)}\n")
        out.write(f"images_with_1_box={sum(1 for c in box_counts if c == 1)}\n")
    print("Saved coco_cat_detection_stats.txt")


if __name__ == "__main__":
    one_pass()
