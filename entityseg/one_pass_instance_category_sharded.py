"""Sample a few random batches from ``entity_seg_instance_category_sharded`` TFDS.

Run from the ``entityseg`` directory::

  python3 -m one_pass_instance_category_sharded --data_dir gs://.../entityseg_category_tfds

"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from entity_seg_instance_category_sharded import load_random_batch


def _save_batch_visualization(
    batch,
    chosen_category_id: int,
    batch_idx: int,
) -> str:
    n = int(batch["image"].shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        img_f = np.asarray(batch["image"][i].numpy(), dtype=np.float32)
        img = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        # preprocess_func mask is 3ch RGB with red channel only (outline).
        outline_r = np.asarray(batch["mask"][i].numpy(), dtype=np.uint8)[..., 0] > 0
        overlay = img.copy().astype(np.float32)
        overlay[outline_r] = overlay[outline_r] * 0.6 + np.array(
            [255, 0, 0], dtype=np.float32
        ) * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        category_name = batch["metadata"]["category_name"][i].numpy()
        if isinstance(category_name, (bytes, np.bytes_)):
            category_name = category_name.decode("utf-8", errors="replace")
        num_instances = int(batch["metadata"]["num_instances"][i].numpy())

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(
            f"batch={batch_idx} ex={i} cat={chosen_category_id} ({category_name}) n={num_instances}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(outline_r.astype(np.float32), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("red mask outline (R channel)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = (
        f"entity_seg_instance_category_sharded_batch_{batch_idx:02d}_cat_{chosen_category_id}.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample random-batches from entity_seg_instance_category_sharded and (optionally) save PNGs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="TFDS root (local or gs://) containing built entity_seg_instance_category_sharded.",
    )
    parser.add_argument(
        "--split_prefix",
        type=str,
        default="train",
        help="When split is auto-chosen, pick a shard with this prefix (train|validation).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Optional explicit split name, e.g. train_cat_17. If empty, a random train_*_cat shard is used.",
    )
    parser.add_argument(
        "--category_id",
        type=int,
        default=None,
        help="If set, use split_prefix_cat_{id} (via load_random_batch).",
    )
    parser.add_argument("--num_batches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=1024)
    parser.add_argument(
        "--log_path",
        type=str,
        default="single_pass_thru_entityseg_category_sharded.txt",
    )
    parser.add_argument(
        "--save_viz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a PNG for each batch. Use --no-save_viz to skip.",
    )
    args = parser.parse_args()

    split_prefix = str(args.split_prefix).strip()
    split = str(args.split).strip() or None
    num_batches = max(1, int(args.num_batches))
    batch_size = max(1, int(args.batch_size))

    with open(args.log_path, "w", encoding="utf-8") as out:
        out.write("batch_idx\tsplit\tcategory_id\tbatch_category_ids\tviz_path\n")
        for batch_idx in range(num_batches):
            chosen_category_id, batch = load_random_batch(
                ds_dir=args.data_dir,
                split=split,
                split_prefix=split_prefix,
                category_id=(args.category_id if split is None else None),
                batch_size=batch_size,
                shuffle_buffer=args.shuffle_buffer,
            )
            eff_split = split or f"{split_prefix}_cat_{chosen_category_id}"
            batch_category_ids = [int(x) for x in batch["metadata"]["category_id"].numpy().tolist()]

            viz_path = ""
            if args.save_viz:
                viz_path = _save_batch_visualization(
                    batch, chosen_category_id=chosen_category_id, batch_idx=batch_idx
                )
                print(f"saved: {viz_path}")

            out.write(
                f"{batch_idx}\t{eff_split}\t{chosen_category_id}\t{batch_category_ids}\t{viz_path}\n"
            )
            print(
                f"[{batch_idx + 1}/{num_batches}] split={eff_split} "
                f"category_id={chosen_category_id}"
            )


if __name__ == "__main__":
    main()
