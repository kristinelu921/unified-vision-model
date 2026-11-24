#!/usr/bin/env python3
"""Value-preserving PyTorch/XLA compatibility patch for Analogist attention."""
from __future__ import annotations

import numpy as np
import torch
from einops import rearrange


def install(attention_edit_class) -> None:
    """Replace only in-place advanced indexing with functional XLA operations."""

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        del place_in_unet, kwargs
        active_layer = self.cur_att_layer // 2
        if (
            self.cur_step not in self.step_idx
            or (active_layer not in self.sa_layer_idx and active_layer not in self.ca_layer_idx)
        ):
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            return rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)

        resolution = int(np.sqrt(attn.shape[1]))
        cpu_ids = self.index_sa[resolution]
        cache = getattr(self, "_xla_index_cache", {})
        cache_key = (resolution, str(sim.device))
        if cache_key not in cache:
            ids = {name: value.to(device=sim.device, dtype=torch.int64) for name, value in cpu_ids.items()}
            a, b = torch.meshgrid(ids["id_A"], ids["id_B"], indexing="xy")
            ap, bp = torch.meshgrid(ids["id_Aprime"], ids["id_Bprime"], indexing="xy")
            width = int(sim.shape[-1])
            cache[cache_key] = {
                **ids,
                "source_ab": (a * width + b).reshape(1, -1),
                "target_apbp": (ap * width + bp).reshape(1, -1),
                "source_ba": (b * width + a).reshape(1, -1),
                "target_bpap": (bp * width + ap).reshape(1, -1),
            }
            self._xla_index_cache = cache
        ids = cache[cache_key]

        if is_cross and active_layer in self.ca_layer_idx:
            attn = attn.index_fill(1, ids["id_A"], 0.0)
            attn = attn.index_fill(1, ids["id_Aprime"], 0.0)
            attn = attn.index_fill(1, ids["id_B"], 0.0)
        if not is_cross and active_layer in self.sa_layer_idx:
            flat = sim.reshape(sim.shape[0], -1)
            for source_name, target_name in (
                ("source_ab", "target_apbp"),
                ("source_ba", "target_bpap"),
            ):
                source = ids[source_name].expand(flat.shape[0], -1)
                target = ids[target_name].expand(flat.shape[0], -1)
                values = flat.gather(1, source) * float(self.scale_sac)
                flat = flat.scatter(1, target, values)
            sim = flat.reshape_as(sim)
            attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)

    attention_edit_class.forward = forward

