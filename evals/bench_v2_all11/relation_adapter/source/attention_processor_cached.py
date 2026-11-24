import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import RMSNorm
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

class IPAFluxAttnProcessor2_0(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size # 3072
        self.cross_attention_dim = cross_attention_dim # 4096
        self.scale = scale
        self.num_tokens = num_tokens
        
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)
            
    def _parameter_signature(self):
        return (self.norm_added_k.eps, self.hidden_size, self.cross_attention_dim, self.num_tokens,
                tuple((id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device))
                      for p in self.parameters()))

    def clear_relation_cache(self):
        self._relation_cache = None

    @torch.no_grad()
    def bind_relation_cache(self, image_emb, *, identity, batch_size, heads, head_dim):
        # Identity includes every numerical dependency and ordered allowed inputs.
        # Cache is bound for one episode and must be cleared in finally.
        if self.training:
            raise RuntimeError('conditioning cache requires evaluation mode')
        if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('a complete conditioning identity SHA-256 is required')
        if image_emb.requires_grad:
            raise ValueError('conditioning must be detached and immutable')
        self.clear_relation_cache()
        version = image_emb._version
        key, value = self.relation_kv(image_emb, batch_size, heads, head_dim)
        self._relation_cache = (image_emb, version, identity, batch_size, heads,
                                head_dim, self._parameter_signature(), key, value)

    def relation_kv(self, image_emb, batch_size, heads, head_dim):
        cache = getattr(self, '_relation_cache', None)
        if cache is not None:
            bound, version, identity, b, h, d, params, key, value = cache
            if (image_emb is not bound or image_emb._version != version
                    or (batch_size, heads, head_dim) != (b, h, d)
                    or params != self._parameter_signature() or self.training):
                self.clear_relation_cache()
                raise RuntimeError('relation cache dependency changed; rebind explicitly')
            return key, value
        key = self.to_k_ip(image_emb).view(batch_size, -1, heads, head_dim).transpose(1, 2)
        value = self.to_v_ip(image_emb).view(batch_size, -1, heads, head_dim).transpose(1, 2)
        return self.norm_added_k(key), value

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        image_emb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                
        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) # torch.Size([1, 24, 4800, 128])
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if image_emb is not None:
            # `ip-adapter` projections
            ip_hidden_states_key_proj, ip_hidden_states_value_proj = self.relation_kv(
                image_emb, batch_size, attn.heads, head_dim)

            ip_hidden_states = F.scaled_dot_product_attention(query, 
                                                              ip_hidden_states_key_proj, 
                                                              ip_hidden_states_value_proj, 
                                                              dropout_p=0.0, is_causal=False)

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
                        
        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
                        
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2) # (512+3840,128)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if encoder_hidden_states is not None:

            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            if image_emb is not None:
                hidden_states = hidden_states + self.scale * ip_hidden_states
                        
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            if image_emb is not None:
                hidden_states = hidden_states + self.scale * ip_hidden_states
            
            return hidden_states
