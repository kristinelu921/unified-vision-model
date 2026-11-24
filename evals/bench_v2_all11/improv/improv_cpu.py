"""Archived numerical functions with only TPU sharding/barriers removed for CPU FP32."""
import hashlib,inspect,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
PROMPT = ''
BATCH = 4
NATIVE_TILE = 111
CANVAS = 224

def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()

def encode_prompts(source_dir: Path, model_dir: Path):
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_dir / 'tokenizer', local_files_only=True)
    encoder = CLIPTextModel.from_pretrained(model_dir / 'text_encoder', torch_dtype=torch.float32, local_files_only=True, low_cpu_mem_usage=False).eval()
    tokens = tokenizer([PROMPT], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    attention_mask = tokens.attention_mask if getattr(encoder.config, 'use_attention_mask', False) else None
    with torch.no_grad():
        embeddings = encoder(tokens.input_ids, attention_mask=attention_mask)[0].float().cpu().contiguous()
    del encoder, tokenizer
    if tuple(embeddings.shape) != (1, 77, 1024):
        raise SystemExit(f'prompt embedding shape differs: {tuple(embeddings.shape)}')
    return (embeddings, sha256_array(embeddings.numpy()))

def _public_config(path: Path, cls) -> dict:
    raw = json.loads(path.read_text())
    accepted = set(inspect.signature(cls.__init__).parameters)
    return {key: value for (key, value) in raw.items() if key in accepted and key != 'self'}

def _strict_safetensors(module, path: Path) -> None:
    from safetensors.torch import load_file
    result = module.load_state_dict(load_file(str(path), device='cpu'), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'strict state mismatch: {path}')

def load_runtime(source_dir: Path, model_dir: Path, vq_dir: Path):
    sys.path.insert(0, str(source_dir.resolve()))
    from improv.modeling.meta_arch.modules import PatchEmbedEncoder, TransformerMAE, VQDecoder
    from improv.pipelines.pipeline_improv import MAEScheduler
    image_encoder = PatchEmbedEncoder(**_public_config(model_dir / 'image_encoder/config.json', PatchEmbedEncoder))
    _strict_safetensors(image_encoder, model_dir / 'image_encoder/diffusion_pytorch_model.safetensors')
    decoder_config = json.loads((model_dir / 'image_decoder/config.json').read_text())
    image_decoder = VQDecoder(pretrained_model_name_or_path=str(vq_dir), pixel_mean=tuple(decoder_config['pixel_mean']), pixel_std=tuple(decoder_config['pixel_std']))
    _strict_safetensors(image_decoder, model_dir / 'image_decoder/diffusion_pytorch_model.safetensors')
    masked_model = TransformerMAE(**_public_config(model_dir / 'mask_image_model/config.json', TransformerMAE))
    _strict_safetensors(masked_model, model_dir / 'mask_image_model/diffusion_pytorch_model.safetensors')
    scheduler = MAEScheduler.from_pretrained(str(model_dir / 'scheduler'), local_files_only=True)
    return (image_encoder, image_decoder, masked_model, scheduler)

def resize_tile(array: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(np.asarray(array, np.uint8), 'RGB')
    pil = pil.resize((NATIVE_TILE, NATIVE_TILE), Image.Resampling.BICUBIC)
    return np.asarray(pil, np.float32).transpose(2, 0, 1) / np.float32(255.0)

def build_canvas(demo_input, demo_target, query_input):
    canvas = np.ones((3, CANVAS, CANVAS), np.float32)
    canvas[:, :NATIVE_TILE, :NATIVE_TILE] = resize_tile(demo_input)
    canvas[:, :NATIVE_TILE, -NATIVE_TILE:] = resize_tile(demo_target)
    canvas[:, -NATIVE_TILE:, :NATIVE_TILE] = resize_tile(query_input)
    canvas[:, -NATIVE_TILE:, -NATIVE_TILE:] = 0.0
    mask = np.zeros((1, CANVAS, CANVAS), np.float32)
    mask[:, -NATIVE_TILE:, -NATIVE_TILE:] = 1.0
    return (canvas, mask)

def official_postprocess(full_grid: np.ndarray) -> np.ndarray:
    hwc = np.clip(np.transpose(np.asarray(full_grid, np.float32), (1, 2, 0)), 0.0, 1.0)
    pil = Image.fromarray((hwc * np.float32(255.0)).astype(np.uint8), 'RGB')
    crop = pil.crop((CANVAS - NATIVE_TILE, CANVAS - NATIVE_TILE, CANVAS, CANVAS))
    return np.asarray(crop.resize((256, 256), Image.Resampling.BICUBIC), np.uint8)

def predict_batch(runtime, arrays, embedding, device, mesh):
    import torch
    (image_encoder, image_decoder, masked_model, _scheduler) = runtime
    (canvas_np, mask_np) = arrays
    canvas = torch.from_numpy(canvas_np).to(device=device, dtype=torch.float32)
    input_mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
    prompt = embedding.repeat(BATCH, 1, 1).to(device=device, dtype=torch.float32)
    mask_small = torch.nn.functional.interpolate(torch.from_numpy(mask_np[:1]), size=(14, 14), mode='bilinear')
    mask_flat = (mask_small > 0).to(torch.float32).reshape(1, 196)
    ids_shuffle = torch.argsort(mask_flat, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1).repeat(BATCH, 1).to(device)
    len_keep = int(196 * (1.0 - float(mask_flat.mean())))
    ids_keep = ids_shuffle[:, :len_keep].repeat(BATCH, 1).to(device)
    with torch.no_grad():
        masked_canvas = canvas * (input_mask < 0.5)
        latents = image_encoder(masked_canvas).permute(0, 2, 3, 1).reshape(BATCH, 196, 1024)
        visible = torch.gather(latents, 1, ids_keep.unsqueeze(-1).repeat(1, 1, 1024))
        logits = masked_model(visible, ids_restore=ids_restore, encoder_hidden_states=prompt)
        tokens = logits.argmax(dim=-1).reshape(BATCH, 14, 14)
        decoded = image_decoder(tokens)
        blended = canvas * (input_mask < 0.5) + decoded * (input_mask >= 0.5)
        grids = blended.float().cpu().numpy()
    return np.stack([official_postprocess(grid) for grid in grids])
