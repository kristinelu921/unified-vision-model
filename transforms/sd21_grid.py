"""4×4 grid: columns = Original + three prompts; rows = four inputs."""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
ROWS = [
    ("dog", "dog.jpeg"),
    ("house", "house.jpeg"),
    ("mountain", "mountain.webp"),
    ("polar_bear", "polar_bear.jpeg"),
]
PROMPTS = ["gjgjfkjagkekgg", "apple", "ahhiodjfiogjdfogi", "Studio Ghibli style illustration"]
COLS = ["Original"] + PROMPTS


def _im(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    fig, axes = plt.subplots(4, 5, figsize=(12, 15))
    for r, (stem, src_name) in enumerate(ROWS):
        orig = _im(os.path.join(ROOT, "images", src_name))
        axes[r, 0].imshow(orig)
        axes[r, 0].axis("off")
        for c, p in enumerate(PROMPTS, start=1):
            out = _im(os.path.join(ROOT, "outputs_0.5", p, f"{stem}_{0.8:.2f}.png"))
            axes[r, c].imshow(out)
            axes[r, c].axis("off")

    for c, title in enumerate(COLS):
        axes[0, c].set_title(title, fontsize=9)
    for r, (stem, _) in enumerate(ROWS):
        axes[r, 0].set_ylabel(stem, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(ROOT, "sd21_grid_.8.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(out_path)


if __name__ == "__main__":
    main()
