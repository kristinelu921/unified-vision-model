from __future__ import annotations

import argparse

import gc

import hashlib

from io import BytesIO

import json

import math

import os

from pathlib import Path

import socket

import sys

import time

import numpy as np

from PIL import Image, ImageOps

from painter_compat import install as install_import_shims

JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")

TRUSTED_MANIFEST = JFLUX / "docs/bench_v2_audit/manifest_v1.json"

TRUSTED_MANIFEST_SHA256 = "89f2501aa722cd121da12e33a24ac9037d5cf50df7627911b7f6ce22fcd6c49d"

PAINTER_VALIDATION_PLAN_ID = "3a2ad324c375b72f641c9359a3dc62467f2cb668ee79fb67b47f59addf6a7bed"

VICT_COMMIT = "56c395244223ae464d754cae1ef2cad073cd48ac"

PAINTER_REVISION = "97200aa029bab2e61ea5b4ce627534e35f152b23"

CHECKPOINT_SHA256 = "1bdd60ce013e82253e2a031726f36d07587947438b0e561d4338d6f0c1113f44"

CHECKPOINT_BYTES = 1_483_011_937

ARCH = "painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"

P = 1

EPISODE_SEED = 42

STEPS = 60

LR = 1e-6

READOUT = "VICT@60step_encoder_noaug_R0"

CANDIDATE_ID = "R0"

READOUT_TAG = "vict60_encoder_noaug_R0"

ADAPTATION_TRANSFORM = "single_crop_full_image_resize_448_no_augmentation"

ACTIVATION_CHECKPOINTING = True

OPTIMIZER_IMPLEMENTATION = "torch_xla.amp.syncfree.AdamW"

AMP_DTYPE = "bfloat16"

AMP_INITIAL_SCALE = 65536.0

PLAN_SCHEMA = "vict_bench_v2_native_p1_noaug_plan_v1"

VIEW_SCHEMA = "vict_bench_v2_native_p1_noaug_generation_v1"

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)

IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)

SOURCE_SHA256 = {
    "models_painter_shared_tt.py": "f7f00ce3987612c8a1f68dee3b01454b27f141eaf020c36743c458413411bdc8",
    "engine_test_time.py": "6cc18600b1ff746288f2a2fc4bc73ca12e50f9fa6321e92248ee196604bd7b20",
    "data/pairdataset_tt.py": "8076e15ed638c9947765e1bb4e17c81187756625f9b83063b6a0901288ee45ec",
}

NO_AUGMENT_TASKS = {"instance_segmentation", "human_pose_2d", "face_landmarks"}

TARGET_BICUBIC_TASKS = {"relative_depth", "human_pose_2d", "face_landmarks"}

OUTPUT_NEAREST_TASKS = {"instance_segmentation", "human_pose_2d", "face_landmarks"}

OUTPUT_BILINEAR_TASKS = {
    "relative_depth", "semantic_segmentation", "edge_detection",
    "saliency_detection", "surface_normals",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _rrc_params(height: int, width: int, generator, scale=(0.3, 1.0), ratio=(0.75, 4.0 / 3.0)):
    import torch
    area = height * width
    log_ratio = torch.log(torch.tensor(ratio, dtype=torch.float32))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(scale[0], scale[1], generator=generator))
        aspect = math.exp(float(torch.empty(1).uniform_(float(log_ratio[0]), float(log_ratio[1]), generator=generator)))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            top = int(torch.randint(0, height - h + 1, (1,), generator=generator))
            left = int(torch.randint(0, width - w + 1, (1,), generator=generator))
            return top, left, h, w
    in_ratio = width / height
    if in_ratio < ratio[0]:
        w, h = width, int(round(width / ratio[0]))
    elif in_ratio > ratio[1]:
        h, w = height, int(round(height * ratio[1]))
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w

def transform_pair(image: np.ndarray, target: np.ndarray | None, task: str, generator) -> tuple[np.ndarray, np.ndarray | None]:
    import torch
    image_pil = Image.fromarray(image, "RGB")
    target_pil = None if target is None else Image.fromarray(target, "RGB")
    target_resample = Image.Resampling.BICUBIC if task in TARGET_BICUBIC_TASKS else Image.Resampling.NEAREST
    if task in NO_AUGMENT_TASKS:
        box = (0, 0, image_pil.width, image_pil.height)
        do_flip = False
    else:
        top, left, height, width = _rrc_params(image_pil.height, image_pil.width, generator)
        box = (left, top, left + width, top + height)
        do_flip = bool(float(torch.rand(1, generator=generator)) < 0.5)
    image_pil = image_pil.crop(box).resize((448, 448), Image.Resampling.BICUBIC)
    if target_pil is not None:
        target_pil = target_pil.crop(box).resize((448, 448), target_resample)
    if do_flip:
        image_pil = ImageOps.mirror(image_pil)
        if target_pil is not None:
            target_pil = ImageOps.mirror(target_pil)
    x = np.asarray(image_pil, np.float32) / np.float32(255.0)
    y = None if target_pil is None else np.asarray(target_pil, np.float32) / np.float32(255.0)
    return x, y

def make_canvas(tiles: dict, task: str, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    import torch
    if seed is None:
        query = np.asarray(Image.fromarray(tiles["query_input"], "RGB").resize((448, 448), Image.Resampling.BICUBIC), np.float32) / 255.0
        demo = np.asarray(Image.fromarray(tiles["demo_input"], "RGB").resize((448, 448), Image.Resampling.BICUBIC), np.float32) / 255.0
        target_mode = Image.Resampling.BICUBIC if task in TARGET_BICUBIC_TASKS else Image.Resampling.NEAREST
        demo_target = np.asarray(Image.fromarray(tiles["demo_target"], "RGB").resize((448, 448), target_mode), np.float32) / 255.0
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) & ((1 << 63) - 1))
        # Official PairDatasetTT transforms query first, then the prompt.
        query, _ = transform_pair(tiles["query_input"], None, task, generator)
        demo, demo_target = transform_pair(tiles["demo_input"], tiles["demo_target"], task, generator)
    samples = np.concatenate([demo, query], axis=0)
    # The true query target is never passed to the model. A finite masked filler
    # is used exactly as in the prior XLA port.
    targets = np.concatenate([demo_target, demo_target], axis=0)
    return ((samples - IMAGENET_MEAN) / IMAGENET_STD,
            (targets - IMAGENET_MEAN) / IMAGENET_STD)

def verify_sources(source_dir: Path) -> None:
    got = {name: sha256_file(source_dir / name) for name in SOURCE_SHA256}
    if got != SOURCE_SHA256:
        raise SystemExit(f"VICT source hash gate failed: {got}")

def load_model(source_dir: Path, checkpoint: Path):
    import torch
    verify_sources(source_dir)
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("Painter checkpoint size/SHA256 gate failed")
    install_import_shims()
    sys.path.insert(0, str(source_dir))
    import models_painter_shared_tt
    model = getattr(models_painter_shared_tt, ARCH)(
        loss_func="smoothl1", use_act_checkpoint=ACTIVATION_CHECKPOINTING
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"checkpoint/model mismatch: {result}")
    del state
    return model

def configure_encoder(model) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        enabled = not name.startswith("decoder")
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    if not names or not any(name.startswith("decoder") for name, _ in model.named_parameters()):
        raise SystemExit("encoder/decoder parameter partition failed")
    return names

def reset_optimizer_state(optimizer) -> None:
    import torch
    with torch.no_grad():
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()

def to_device(samples_np, targets_np, device):
    import torch
    x = torch.from_numpy(samples_np).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    y = torch.from_numpy(targets_np).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
    return x, y

def output_mode(task: str) -> str:
    if task in OUTPUT_NEAREST_TASKS:
        return "nearest"
    if task in OUTPUT_BILINEAR_TASKS:
        return "bilinear"
    return "bicubic"

def predict(model, samples, targets, mask, task: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm
    model.eval()
    with torch.no_grad():
        latent = model.forward_encoder(samples, targets, 0, mask)
        output = model.forward_decoder(latent)[:, :, 448:, :].cpu()
        mean = torch.as_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_STD)[None, :, None, None]
        output = torch.clamp((output * std + mean) * 255.0, 0.0, 255.0)
        mode = output_mode(task)
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        output = F.interpolate(output, (256, 256), mode=mode, **kwargs)
        xm.mark_step()
        xm.wait_device_ops()
    return output[0].permute(1, 2, 0).to(torch.uint8).numpy()
