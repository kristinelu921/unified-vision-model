"""Inference-only import shims for the released Painter architecture.

These are the audited shims used by the completed legacy TPU run.  The
optional training packages are never executed during inference; checkpoint
loading still fails on every missing or unexpected key.
"""
from __future__ import annotations

import sys
import types


def install() -> None:
    import torch
    import torch.nn as nn

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, value):
            if self.drop_prob == 0.0 or not self.training:
                return value
            keep = 1.0 - self.drop_prob
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            random = keep + torch.rand(shape, dtype=value.dtype, device=value.device)
            return value.div(keep) * random.floor()

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None,
                     act_layer=nn.GELU, drop=0.0):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, value):
            return self.drop(self.fc2(self.drop(self.act(self.fc1(value)))))

    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_vit = types.ModuleType("timm.models.vision_transformer")
    timm_layers.DropPath = DropPath
    timm_layers.trunc_normal_ = nn.init.trunc_normal_
    timm_vit.Mlp = Mlp
    timm.models = timm_models
    timm_models.layers = timm_layers
    timm_models.vision_transformer = timm_vit

    detectron2 = types.ModuleType("detectron2")
    d2_layers = types.ModuleType("detectron2.layers")

    class CNNBlockBase(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride

    def get_norm(kind, channels):
        if kind == "LN":
            return nn.GroupNorm(1, channels)
        raise ValueError(f"unsupported inference shim norm: {kind}")

    d2_layers.CNNBlockBase = CNNBlockBase
    d2_layers.Conv2d = nn.Conv2d
    d2_layers.get_norm = get_norm
    detectron2.layers = d2_layers

    fvcore = types.ModuleType("fvcore")
    fvcore_nn = types.ModuleType("fvcore.nn")
    fvcore_weight = types.ModuleType("fvcore.nn.weight_init")

    def c2_msra_fill(module):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    fvcore_weight.c2_msra_fill = c2_msra_fill
    fvcore_nn.weight_init = fvcore_weight
    fvcore.nn = fvcore_nn

    fairscale = types.ModuleType("fairscale")
    fairscale_nn = types.ModuleType("fairscale.nn")
    fairscale_checkpoint = types.ModuleType("fairscale.nn.checkpoint")
    fairscale_checkpoint.checkpoint_wrapper = lambda module: module
    fairscale_nn.checkpoint = fairscale_checkpoint
    fairscale.nn = fairscale_nn

    sys.modules.update({
        "timm": timm, "timm.models": timm_models, "timm.models.layers": timm_layers,
        "timm.models.vision_transformer": timm_vit, "detectron2": detectron2,
        "detectron2.layers": d2_layers, "fvcore": fvcore, "fvcore.nn": fvcore_nn,
        "fvcore.nn.weight_init": fvcore_weight, "fairscale": fairscale,
        "fairscale.nn": fairscale_nn, "fairscale.nn.checkpoint": fairscale_checkpoint,
    })
