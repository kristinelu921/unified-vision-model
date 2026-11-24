"""Narrow Apex/FlashAttention compatibility surface for the TPU-native port."""
from __future__ import annotations

import os
import sys
import types
from typing import Iterable

import torch
import torch.nn as nn


class FusedRMSNorm(nn.Module):
    def __init__(self, normalized_shape: int | Iterable[int], eps=1e-5,
                 elementwise_affine=True, memory_efficient=False):
        super().__init__()
        del memory_efficient
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape)) if elementwise_affine else None

    def forward(self, value):
        dims = tuple(range(-1, -len(self.normalized_shape) - 1, -1))
        variance = value.float().pow(2).mean(dims, keepdim=True)
        result = value * torch.rsqrt(variance + self.eps)
        if self.weight is None:
            return result
        return self.weight * result.to(self.weight.dtype)


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        raise RuntimeError("LaVin-DiT TPU attention refuses non-TPU PJRT backends")
    if any(t.device.type != "xla" for t in (q, k, v)):
        raise RuntimeError("LaVin-DiT TPU attention requires XLA tensors")
    if dropout_p != 0.0 or causal:
        raise RuntimeError("only non-causal evaluation attention with dropout=0 is admitted")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise RuntimeError("expected B,L,H,D Q/K/V")
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    if q.shape[1] % k.shape[1] or k.shape[1] != v.shape[1]:
        raise RuntimeError("unsupported grouped-query head layout")
    group = q.shape[1] // k.shape[1]
    if group != 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    from torch_xla.experimental.custom_kernel import flash_attention
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return flash_attention(q, k, v, causal=False, sm_scale=scale).transpose(1, 2).contiguous()


def install_import_shims():
    apex = types.ModuleType("apex")
    normalization = types.ModuleType("apex.normalization")
    fused = types.ModuleType("apex.normalization.fused_layer_norm")
    fused.FusedLayerNorm = nn.LayerNorm
    fused.FusedRMSNorm = FusedRMSNorm
    normalization.fused_layer_norm = fused
    apex.normalization = normalization
    flash = types.ModuleType("flash_attn")
    flash.flash_attn_func = flash_attn_func
    sys.modules.update({
        "apex": apex,
        "apex.normalization": normalization,
        "apex.normalization.fused_layer_norm": fused,
        "flash_attn": flash,
    })
