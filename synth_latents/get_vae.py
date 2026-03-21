import argparse
from pathlib import Path
import json
from itertools import combinations

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModel, AutoModelForPreTraining

try:
    from .transforms import add_noise_to_latents, affine_transform_latents, interpolate_latents
except ImportError:
    from transforms import add_noise_to_latents, affine_transform_latents, interpolate_latents

DEFAULT_VAE_ID = "stabilityai/sdxl-vae"
DEFAULT_MAE_ID = "facebook/vit-mae-base"
DEFAULT_SIGLIP_ID = "google/siglip2-base-patch32-256"
TARGET_IMAGE_SIZE = (256, 256)
_VAE = None
_MAE_PROCESSOR = None
_MAE_MODEL = None
_SIGLIP_PROCESSOR = None
_SIGLIP_MODEL = None


def get_vae():
    global _VAE
    if _VAE is None:
        _VAE = AutoencoderKL.from_pretrained(DEFAULT_VAE_ID)
        _VAE.eval()
    return _VAE


def get_mae_processor():
    global _MAE_PROCESSOR
    if _MAE_PROCESSOR is None:
        _MAE_PROCESSOR = AutoImageProcessor.from_pretrained(DEFAULT_MAE_ID)
    return _MAE_PROCESSOR


def get_mae_model():
    global _MAE_MODEL
    if _MAE_MODEL is None:
        _MAE_MODEL = AutoModelForPreTraining.from_pretrained(DEFAULT_MAE_ID)
        _MAE_MODEL.eval()
    return _MAE_MODEL


def get_siglip_processor():
    global _SIGLIP_PROCESSOR
    if _SIGLIP_PROCESSOR is None:
        _SIGLIP_PROCESSOR = AutoImageProcessor.from_pretrained(DEFAULT_SIGLIP_ID)
    return _SIGLIP_PROCESSOR


def get_siglip_model():
    global _SIGLIP_MODEL
    if _SIGLIP_MODEL is None:
        try:
            _SIGLIP_MODEL = AutoModel.from_pretrained(DEFAULT_SIGLIP_ID, dtype="auto")
        except TypeError:
            _SIGLIP_MODEL = AutoModel.from_pretrained(
                DEFAULT_SIGLIP_ID, torch_dtype="auto"
            )
        _SIGLIP_MODEL.eval()
    return _SIGLIP_MODEL


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    image = image.resize(TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS)
    return image


def image_to_tensor(image):
    image_np = np.asarray(image, dtype=np.float32)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor / 127.5 - 1.0


@torch.no_grad()
def get_bottleneck_latents(image_path, model_type="vae"):
    image = load_image(image_path)
    if model_type == "vae":
        image_tensor = image_to_tensor(image)
        vae = get_vae()
        latents = vae.encode(image_tensor).latent_dist.sample()
        return latents * vae.config.scaling_factor
    if model_type == "mae":
        processor = get_mae_processor()
        model = get_mae_model()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state
    if model_type == "siglip":
        processor = get_siglip_processor()
        model = get_siglip_model()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        raise ValueError("SigLIP model output did not include last_hidden_state or pooler_output.")
    raise ValueError(f"Unsupported model_type: {model_type}")



@torch.no_grad()
def decode_latents(latents, model_type="vae"):
    if model_type != "vae":
        raise NotImplementedError(
            f"decode_latents only supports model_type='vae'. Received: {model_type}"
        )
    vae = get_vae()
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((decoded * 255).round().astype(np.uint8))

def run_imagenet(noise_level=0.1):
    raise NotImplementedError("run_imagenet is not implemented yet.")


def add_label(image, text, label_height=36):
    canvas = Image.new("RGB", (image.width, image.height + label_height), color="white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), text, fill="black")
    return canvas


def make_grid(images, rows, cols, background="white"):
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")

    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    grid = Image.new("RGB", (cols * cell_width, rows * cell_height), color=background)

    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        x_offset = col * cell_width + (cell_width - image.width) // 2
        y_offset = row * cell_height + (cell_height - image.height) // 2
        grid.paste(image, (x_offset, y_offset))

    return grid

def add_to_stats(all_stats, latents, stat_type, stat_value, seed=42):
    entry = {
        "stat_type": stat_type,
        "stat_value": stat_value,
        "seed": seed,
        "shape": list(latents.shape),
        "max_latents": latents.max().item(),
        "min_latents": latents.min().item(),
    }
    all_stats.append(entry)


def zero_latent_slice(latents, axis, index):
    zeroed_latents = latents.clone()
    if axis == "channel":
        zeroed_latents[:, index, :, :] = 0
    elif axis == "height":
        zeroed_latents[:, :, index, :] = 0
    elif axis == "width":
        zeroed_latents[:, :, :, index] = 0
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    return zeroed_latents


def add_zeroed_latent_views(image_entries, output_dir, all_stats):
    for entry in image_entries:
        latents = entry["latents"]
        stem = entry["stem"]
        transform_specs = (
            [("channel", index) for index in range(latents.shape[1])]
            + [("height", index) for index in range(latents.shape[2])]
            + [("width", index) for index in range(latents.shape[3])]
        )
        zeroed_cells = []

        for axis, index in transform_specs:
            zeroed_latents = zero_latent_slice(latents, axis, index)
            decoded_image = decode_latents(zeroed_latents)
            torch.save(
                zeroed_latents.cpu(),
                output_dir / f"{stem}_latents_zero_{axis}_{index:02d}.pt",
            )
            zeroed_cells.extend(
                [
                    add_label(entry["image"].copy(), f"{stem} original"),
                    add_label(decoded_image, f"zero {axis} {index}"),
                ]
            )
            add_to_stats(all_stats, zeroed_latents, f"zero_{axis}", index, seed=None)

        zeroed_grid = make_grid(
            zeroed_cells,
            rows=len(transform_specs),
            cols=2,
        )
        zeroed_grid.save(output_dir / f"{stem}_zeroed_latent_slices.png")


def add_pairwise_interpolations(image_entries, output_dir, all_stats, alpha=0.5):
    interpolation_cells = []
    pair_count = 0

    for left_entry, right_entry in combinations(image_entries, 2):
        interpolated_latents = interpolate_latents(
            left_entry["latents"],
            right_entry["latents"],
            alpha=alpha,
        )
        interpolated_image = decode_latents(interpolated_latents)
        pair_stem = f"{left_entry['stem']}_to_{right_entry['stem']}"
        torch.save(
            interpolated_latents.cpu(),
            output_dir / f"{pair_stem}_latents_interp_{alpha:.2f}.pt",
        )
        interpolation_cells.extend(
            [
                add_label(left_entry["image"].copy(), f"{left_entry['stem']} original"),
                add_label(interpolated_image, f"interp {alpha:.2f}"),
                add_label(right_entry["image"].copy(), f"{right_entry['stem']} original"),
            ]
        )
        add_to_stats(all_stats, interpolated_latents, "interpolation", alpha, seed=None)
        pair_count += 1

    if pair_count == 0:
        return

    interpolation_grid = make_grid(interpolation_cells, rows=pair_count, cols=3)
    interpolation_grid.save(output_dir / "toy_example_interpolations.png")


def toy_example(
    file_path,
    output_path,
    shift_level=1.0,
    scale_level=1.0,
    noise_level=0.0,
    seed=42,
    model_type="vae",
):
    input_path = Path(file_path)
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png")
            for path in input_path.glob(pattern)
        )
    if not image_paths:
        raise FileNotFoundError(f"No input images found at: {file_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type != "vae":
        raise NotImplementedError(
            "toy_example currently supports only model_type='vae' because the pipeline "
            "depends on decoding latents back into images for grids and ablations."
        )

    num_images = len(image_paths)
    transform_specs = [
        ("noise", 0.2),
        ("noise", 0.4),
        ("noise", 0.6),
        ("noise", 0.8),
        ("noise", 1.0),
    ]
    num_transforms = len(transform_specs)
    grid_rows = [[] for _ in range(num_transforms)]
    all_stats = []
    image_entries = []

    for image_path in image_paths:
        original_image = load_image(image_path)
        latents = get_bottleneck_latents(image_path, model_type=model_type)
        stem = image_path.stem
        torch.save(latents.cpu(), output_dir / f"{stem}_latents.pt")
        image_entries.append({"image": original_image, "latents": latents, "stem": stem})

        for row_index, (transform_type, value) in enumerate(transform_specs):
            if transform_type == "shift":
                transformed_latents = affine_transform_latents(latents, shift_level=value)
            elif transform_type == "scale":
                transformed_latents = affine_transform_latents(latents, scale_level=value)
            else:
                transformed_latents = add_noise_to_latents(latents, noise_level=value, seed=seed)
                torch.save(
                    transformed_latents.cpu(),
                    output_dir / f"{stem}_latents_noisy_{value:.2f}.pt",
                )

            decoded_image = decode_latents(transformed_latents)
            grid_rows[row_index].append(add_label(original_image.copy(), f"{stem} original"))
            grid_rows[row_index].append(add_label(decoded_image, f"{transform_type} {value}"))
            add_to_stats(
                all_stats,
                transformed_latents,
                transform_type,
                value,
                seed=seed if transform_type == "noise" else None,
            )

    grid_cells = [image for row in grid_rows for image in row]
    grid = make_grid(grid_cells, rows=num_transforms, cols=num_images * 2)
    grid.save(output_dir / "toy_example_grid.png")
    add_zeroed_latent_views(image_entries, output_dir, all_stats)
    add_pairwise_interpolations(image_entries, output_dir, all_stats)
    with open(output_dir / "toy_example_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file-path",
        default="/kmh-nfs-ssd-us-mount/code/kristine/lvm/synth_latents/images",
    )
    parser.add_argument(
        "--output-path",
        default="/kmh-nfs-ssd-us-mount/code/kristine/lvm/synth_latents/output",
    )
    parser.add_argument(
        "--model-type",
        choices=("vae", "mae", "siglip"),
        default="vae",
        help="Choose which encoder to load. The current visualization pipeline only decodes VAE latents.",
    )
    parser.add_argument("--noise-level", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    toy_example(
        file_path=args.file_path,
        output_path=args.output_path,
        noise_level=args.noise_level,
        seed=args.seed,
        model_type=args.model_type,
    )
    print(f"Saved toy example grid across all images in {args.file_path}")
