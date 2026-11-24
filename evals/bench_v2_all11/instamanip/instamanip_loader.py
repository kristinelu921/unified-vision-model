#!/usr/bin/env python3
"""Fail-closed loader for the released InstaManip 17B three-shot asset stack."""
from __future__ import annotations

import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

BUNDLE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("INSTAMANIP_SOURCE_DIR", BUNDLE / "official" / "InstaManip")).resolve()
ASSETS = Path(os.environ.get("INSTAMANIP_ASSETS_DIR", BUNDLE / "assets")).resolve()
SEED = ASSETS / "SEED-X-17B"
TOKENIZER_DIR = SEED / "cvlm_llama2_tokenizer_100img_and_224loc_addpatch"
LLM_DIR = SEED / "seed_x" / "llm"
VISUAL_CKPT = ASSETS / "QwenViT" / "qwen_vit_G.pt"
AGENT_CKPT = ASSETS / "InstaManip" / "InstaManip-17B-3shot" / "pytorch_model.bin"
SDXL_DIR = ASSETS / "stable-diffusion-xl-base-1.0"
SECOND_STAGE_CKPT = SEED / "seed_detokenizer" / "second_stage" / "pytorch_model.bin"
RELEASED_ADDED_TOKENS = SOURCE / "pretrained" / "added_tokens.json"


def install_source_paths() -> None:
    for path in (BUNDLE / "compat", SOURCE):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


install_source_paths()


@dataclass
class LoadAudit:
    component: str
    checkpoint: str
    checkpoint_tensors: int
    checkpoint_parameters: int
    model_tensors: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    mismatched_keys: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"component": self.component, "checkpoint": self.checkpoint,
                "checkpoint_tensors": self.checkpoint_tensors,
                "checkpoint_parameters": self.checkpoint_parameters,
                "model_tensors": self.model_tensors, "missing_count": len(self.missing_keys),
                "missing_keys": self.missing_keys, "unexpected_count": len(self.unexpected_keys),
                "unexpected_keys": self.unexpected_keys,
                "mismatched_count": len(self.mismatched_keys), "mismatched_keys": self.mismatched_keys}


def _torch_load(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(obj, dict):
        raise TypeError(f"checkpoint is not a state dict: {path}: {type(obj)!r}")
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in obj.items()):
        raise TypeError(f"checkpoint contains non-tensor entries: {path}")
    return obj


def audited_overlay_load(module: torch.nn.Module, checkpoint: Path, component: str,
                         *, allow_missing: bool) -> LoadAudit:
    state = _torch_load(checkpoint)
    model_state = module.state_dict()
    unexpected = sorted(set(state) - set(model_state))
    missing = sorted(set(model_state) - set(state))
    mismatched = [{"key": key, "checkpoint_shape": list(state[key].shape),
                   "model_shape": list(model_state[key].shape)}
                  for key in sorted(set(state) & set(model_state))
                  if tuple(state[key].shape) != tuple(model_state[key].shape)]
    if unexpected or mismatched or (missing and not allow_missing):
        raise RuntimeError(f"{component} checkpoint audit failed: unexpected={len(unexpected)} "
                           f"mismatched={len(mismatched)} missing={len(missing)}")
    result = module.load_state_dict(state, strict=not allow_missing)
    if sorted(result.unexpected_keys) != unexpected or sorted(result.missing_keys) != missing:
        raise RuntimeError(f"{component} load result changed after pre-audit")
    audit = LoadAudit(component, str(checkpoint), len(state),
                      sum(int(t.numel()) for t in state.values()), len(model_state),
                      missing, unexpected, mismatched)
    del state, model_state
    gc.collect()
    return audit


def build_tokenizer():
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    released = json.loads(RELEASED_ADDED_TOKENS.read_text())
    for token, expected_id in sorted(released.items(), key=lambda item: int(item[1])):
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual == tokenizer.unk_token_id:
            if tokenizer.add_tokens([token], special_tokens=True) != 1:
                raise RuntimeError(f"could not add released token {token!r}")
            actual = tokenizer.convert_tokens_to_ids(token)
        if int(actual) != int(expected_id):
            raise RuntimeError(f"token id mismatch for {token}: expected {expected_id}, got {actual}")
    if len(tokenizer) != 32383:
        raise RuntimeError(f"released tokenizer size must be 32383, got {len(tokenizer)}")
    return tokenizer


def build_visual(dtype: torch.dtype = torch.bfloat16):
    from src.models.tokenizer.qwen_visual import VisionTransformerWithAttnPool
    model = VisionTransformerWithAttnPool(heads=16, image_size=448, layers=48,
        mlp_ratio=4.9231, output_dim=4096, patch_size=14, width=1664)
    audit = audited_overlay_load(model, VISUAL_CKPT, "qwen_visual", allow_missing=False)
    return model.eval().to(dtype=dtype), audit


def build_llm(dtype: torch.dtype = torch.bfloat16):
    from peft import LoraConfig, TaskType
    from src.models.mllm.modeling_llama_xformer import LlamaForCausalLM
    from src.models.mllm.peft_models import get_peft_model_with_resize_embedding
    base = LlamaForCausalLM.from_pretrained(str(LLM_DIR), torch_dtype=dtype,
        local_files_only=True, low_cpu_mem_usage=True)
    config = LoraConfig(r=32, lora_alpha=32,
        modules_to_save=["input_layernorm", "post_attention_layernorm", "norm"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.05)
    return get_peft_model_with_resize_embedding(model=base, peft_config=config,
        vocab_size=32400, torch_dtype=dtype).eval()


def build_agent(llm: torch.nn.Module, dtype: torch.dtype = torch.bfloat16):
    from src.models.mllm.instamanip import InstaManip
    from src.models.tokenizer.qwen_visual import Resampler
    model = InstaManip(llm=llm,
        input_resampler=Resampler(grid_size=8, embed_dim=5120, num_heads=32, kv_dim=4096),
        output_resampler=Resampler(grid_size=8, embed_dim=4096, num_heads=32, kv_dim=5120),
        add_patch_pos=True, vit_down=True, mse=True, lm_loss_scale=1.0,
        rec_loss_scale=6.0, relation_reg_relative_scale=0.1).to(dtype=dtype)
    audit = audited_overlay_load(model, AGENT_CKPT, "instamanip_17b_3shot", allow_missing=True)
    if (audit.checkpoint_tensors != 704 or audit.checkpoint_parameters != 671_960_576
            or len(audit.missing_keys) != 363):
        raise RuntimeError("official InstaManip overlay tensor identity changed")
    return model.eval(), audit


def build_diffusion(dtype: torch.dtype = torch.bfloat16):
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from src.models.detokenizer.adapter_modules import SDXLAdapterWithLatentImage
    from src.models.detokenizer.resampler import ResamplerXLV2
    scheduler = EulerDiscreteScheduler.from_pretrained(str(SDXL_DIR), subfolder="scheduler", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(str(SDXL_DIR), subfolder="vae", variant="fp16",
                                        torch_dtype=dtype, local_files_only=True)
    unet = UNet2DConditionModel.from_pretrained(str(SDXL_DIR), subfolder="unet", variant="fp16",
                                                torch_dtype=dtype, local_files_only=True)
    resampler = ResamplerXLV2(dim=1024, depth=4, dim_head=64, heads=16, num_queries=64,
        embedding_dim=4096, output1_dim=768, output2_dim=1280, ff_mult=4, normalize=False)
    adapter = SDXLAdapterWithLatentImage(unet=unet, resampler=resampler, full_ft=True,
                                         set_trainable_late=False, vit_down=True).to(dtype=dtype)
    audit = audited_overlay_load(adapter, SECOND_STAGE_CKPT, "seed_x_second_stage", allow_missing=False)
    return adapter.eval(), vae.eval(), scheduler, audit


def parameter_count(module: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def names_with_prefix(values: Iterable[str], depth: int = 3) -> list[str]:
    return sorted({".".join(value.split(".")[:depth]) for value in values})
