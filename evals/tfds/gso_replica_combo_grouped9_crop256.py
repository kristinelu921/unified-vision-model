"""TFDS builder for 256x256 cropped Omnidata Replica-GSO grouped-9 combos."""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow_datasets as tfds

from .gso_replica_combo_grouped9 import GsoReplicaComboGrouped9, _load_json, _sanitize_split_name


_DEFAULT_INDEX_ROOT = Path("/mnt/klum/lvm/gso_replica_combo_grouped9_crop256_index")


def _index_root() -> Path:
    return Path(os.environ.get("GSO_REPLICA_CROP256_GROUPED9_INDEX", str(_DEFAULT_INDEX_ROOT))).expanduser().resolve()


class GsoReplicaComboGrouped9Crop256(GsoReplicaComboGrouped9):
    VERSION = tfds.core.Version("1.0.0")

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        summary = _load_json(_index_root() / "summary.json")
        splits = {}
        for item in summary["splits"]:
            if int(item["groups"]) <= 0:
                continue
            split_name = _sanitize_split_name(item["split_name"])
            splits[split_name] = self._generate_examples(Path(item["path"]))
        return splits


def load_data(split: str, data_dir: str | None = None):
    load_kw: dict = {"split": split}
    if data_dir is not None:
        load_kw["data_dir"] = data_dir
    return tfds.load("gso_replica_combo_grouped9_crop256", **load_kw)

