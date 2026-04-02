from entity_seg_instance import load_data, sanity_check


def one_pass():
    ds_dir = '/mnt/klum/entityseg/entityseg_instance_tfds/'
    ds = load_data(data_dir=ds_dir, split='train', batch_size=64, repeat=False, shuffle=False)
    sanity_check(ds_dir, 'train', num_batches=5)
    with open('single_pass_thru.txt', 'w') as f:
        for batch in ds:
            image = batch["image"]
            instance = batch["instance"]
            metadata = batch["metadata"]
            f.write(f"{image.shape}, {instance}, {metadata}\n")


if __name__ == "__main__":
    one_pass()
