import importlib

DS_DIRS = {
    "DIODE": "/mnt/klum/eval_tfds/eval_diode_tfds",
    "KITTI": "/mnt/klum/eval_tfds/eval_kitti_tfds",
    "NYU": "/mnt/klum/eval_tfds/eval_nyu_tfds",
    "SCANNET": "/mnt/klum/eval_tfds/eval_scannet_tfds",
    "ETH3D": "/mnt/klum/eval_tfds/eval_eth3d_tfds",
}

def one_pass(ds_dir: str, load_data: callable, sanity_check: callable):
    sanity_check(ds_dir, "test", num_batches=1, batch_size=1)
    ds = load_data(data_dir=ds_dir, split="test", batch_size=1, repeat=False, shuffle=False)

    out_path = "/kmh-nfs-ssd-us-mount/code/kristine/lvm/single_pass_thru.txt"
    with open(out_path, "w", encoding="utf-8"):
        pass

    for i, batch in enumerate(ds):
        image = batch["image"]
        depth = batch["depth"]
        metadata = batch["metadata"]
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"{i}, {image.shape}, {depth.shape}, {metadata}\n")

    sanity_check(ds_dir, "test", num_batches=1, batch_size=1)

if __name__ == "__main__":
    for ds_name, ds_dir in DS_DIRS.items():
        module_name = f"evals.eval_{ds_name.lower()}"
        mod = importlib.import_module(module_name)
        one_pass(ds_dir, mod.load_data, mod.sanity_check)