"""Unchanged inference primitives extracted from the selected archived Painter runner.

Resource selection, source checks, episode replay, qualification and scheduling
are caller responsibilities. This module alone is not an authorized launcher.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np
from PIL import Image
from painter_compat import install as install_import_shims

CHECKPOINT_SHA256 = "1bdd60ce013e82253e2a031726f36d07587947438b0e561d4338d6f0c1113f44"


CHECKPOINT_BYTES = 1_483_011_937


ARCH = "painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)


IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_rgb(array: np.ndarray, *, target: bool = False) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, np.uint8), "RGB")
    resample = Image.Resampling.NEAREST if target else Image.Resampling.BICUBIC
    return np.asarray(image.resize((448, 448), resample), np.float32) / np.float32(255.0)


def preprocess(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for ep in episodes:
        prompt = resize_rgb(ep["demo_input"])
        query = resize_rgb(ep["query_input"])
        target = resize_rgb(ep["demo_target"], target=True)
        inputs.append(np.concatenate([prompt, query], axis=0))
        targets.append(np.concatenate([target, target], axis=0))
    return ((np.stack(inputs) - IMAGENET_MEAN) / IMAGENET_STD,
            (np.stack(targets) - IMAGENET_MEAN) / IMAGENET_STD)


def load_model(source: Path, checkpoint: Path, device):
    import torch
    import torch_xla.core.xla_model as xm
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise SystemExit("Painter checkpoint size/SHA256 gate failed")
    install_import_shims()
    sys.path.insert(0, str(source / "Painter"))
    import models_painter
    model = getattr(models_painter, ARCH)()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"checkpoint/model mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    model = model.eval().to(device)
    xm.mark_step()
    return model


def predict(model, inputs_np, targets_np, device, mesh) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    if inputs_np.shape != (4, 896, 448, 3):
        raise ValueError(f"padded inference batch shape differs: {inputs_np.shape}")
    x = torch.from_numpy(inputs_np).permute(0, 3, 1, 2).contiguous().to(device)
    y = torch.from_numpy(targets_np).permute(0, 3, 1, 2).contiguous().to(device)
    mask = torch.zeros((4, model.patch_embed.num_patches), dtype=torch.bool, device=device)
    mask[:, model.patch_embed.num_patches // 2:] = True
    xs.mark_sharding(x, mesh, ("data", None, None, None))
    xs.mark_sharding(y, mesh, ("data", None, None, None))
    xs.mark_sharding(mask, mesh, ("data", None))
    with torch.no_grad():
        latent = model.forward_encoder(x, y, mask)
        output = model.forward_decoder(latent)[:, :, 448:, :].cpu()
        mean = torch.as_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_STD)[None, :, None, None]
        output = torch.clamp((output * std + mean) * 255.0, 0.0, 255.0)
        output = F.interpolate(output, (256, 256), mode="nearest")
        xm.mark_step()
    return output.permute(0, 2, 3, 1).to(torch.uint8).numpy()
