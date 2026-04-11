from lvis import sanity_check
from lvis_person import load_data

import numpy as np
import tensorflow as tf


def _ragged_values_as_numpy(x):
    """Ragged or dense tensor -> 1-D numpy array of scalars (e.g. class indices)."""
    if isinstance(x, tf.RaggedTensor):
        return x.flat_values.numpy()
    return tf.reshape(x, [-1]).numpy()


def _as_text(name):
    if isinstance(name, (bytes, np.bytes_)):
        return name.decode("utf-8", errors="replace")
    return str(name)


def one_pass():
    ds_dir = "/mnt/klum/lvis/lvis_person_tfds"
    # ds_dir = "gs://kmh-gcp-asia-northeast1-b/kristine/lvm/lvis_tfds"
    ds = load_data(data_dir=ds_dir, split="train", batch_size=1, repeat=False, shuffle=False)
    sanity_check(ds_dir, "train", num_batches=20, dataset_name="lvis_person")

    #watch = {44, 11, 126, 216, 792, 826, 942, 98, 153, 610, 124, 1177, 386, 946, 734, 1098, 231, 206, 117}

    with open("single_pass_thru.txt", "w") as f:
        for batch in ds:
            image = batch["image"]
            annotations = batch["annotations"]
            metadata = batch["metadata"]
            f.write(f"{image.shape}, {annotations}, {metadata}\n")

            class_idxs = _ragged_values_as_numpy(annotations["class_idx"])
            names = _ragged_values_as_numpy(annotations["category_name"])
            for class_idx, name in zip(class_idxs, names):
                cid = int(class_idx)
                if cid in watch:
                    print(f"{cid}: {_as_text(name)}")
                    watch.discard(cid)
            if not watch:
                break
if __name__ == "__main__":
    one_pass()
