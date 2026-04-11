"""Build/load LVIS instance category-index cache.

Usage:
  python lvis/build_lvis_instance_cache.py \
    --data_dir /mnt/klum/lvis/lvis_instance_tfds \
    --split train
"""

from __future__ import annotations

import argparse
import json
import os

from lvis_instance import build_category_index_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/klum/lvis/lvis_instance_tfds",
        help="TFDS root containing lvis_instance.",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional custom cache directory. Defaults to data_dir.",
    )
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    cache = build_category_index_cache(
        split=args.split,
        data_dir=args.data_dir,
        cache_dir=(args.cache_dir or None),
        force_rebuild=args.force_rebuild,
    )
    total_examples = sum(len(v) for v in cache.values())
    print(f"split={args.split}")
    print(f"num_categories={len(cache)}")
    print(f"num_examples_indexed={total_examples}")
    top = sorted(cache.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    print("top_categories_by_count=", json.dumps([(k, len(v)) for k, v in top]))

    # Print cache path for convenience.
    base = args.cache_dir or args.data_dir
    cache_path = os.path.join(base, f"lvis_instance_category_index_{args.split}.json")
    print(f"cache_path={cache_path}")


if __name__ == "__main__":
    main()
