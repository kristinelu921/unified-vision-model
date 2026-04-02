"""
Config for D3 TFDS one-pass evaluation.
"""

ONE_PASS_CFG = {
    "data_dir": "gs://kmh-gcp-us-central2/kristine/lvm/d3_depth_tfds",
    "split": "train",
    "batch_size": 1,
    "max_examples": 10001,
    "sigma_warn_threshold": 0.99,
    "single_pass_log_path": "/kmh-nfs-ssd-us-mount/code/kristine/lvm/single_pass_thru.txt",
    "accuracy_log_path": "accuracy.txt",
    "valid_depth_ranges": {
        "vkitti": (1e-5, 80.0),
        "hypersim": (1e-5, 65.0),
        "coco": (1e-5, 80.0),
    },
    # Mirrors training logic fallback for unknown dataset type.
    "default_all_valid": True,
}
