#!/usr/bin/env python3
"""Pinned, local-only component loader for the official Analogist release."""
from __future__ import annotations

import sys
from pathlib import Path

SOURCE_COMMIT = "3d7b485691ff50090af55caffb4bc7802990e5ca"
MODEL_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-inpainting"
MODEL_REVISION = "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
PROMPT = "an image, best quality, extremely detailed"
NEGATIVE_PROMPT = (
    "Messy,Disordered,Chaotic,Cluttered,Haphazard,Unkempt,Scattered,"
    "Disheveled,Tangled,Random"
)


def _source(source_dir: Path) -> None:
    path = str(source_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def load_runtime_components(source_dir: Path, model_dir: Path, dtype=None):
    """Load runtime weights from the pinned local snapshot only."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    _source(source_dir)
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from models.scheduling_ddim import DDIMScheduler

    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": False,
        "variant": "fp16",
        "use_safetensors": True,
    }
    unet = UNet2DConditionModel.from_pretrained(model_dir / "unet", **kwargs)
    vae = AutoencoderKL.from_pretrained(model_dir / "vae", **kwargs)
    scheduler = DDIMScheduler.from_pretrained(model_dir / "scheduler", local_files_only=True)
    return unet, vae, scheduler


def encode_fixed_prompts(source_dir: Path, model_dir: Path):
    """Encode the fixed task-neutral prompt and official negative prompt on CPU."""
    import torch

    _source(source_dir)
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(model_dir / "tokenizer", local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(
        model_dir / "text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
        low_cpu_mem_usage=False,
        variant="fp16",
        use_safetensors=True,
    ).eval()
    tokens = tokenizer(
        [PROMPT, NEGATIVE_PROMPT],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = encoder(tokens.input_ids)[0].cpu().contiguous()
    return embeddings[:1], embeddings[1:]
