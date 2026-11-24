#!/usr/bin/env python3
"""Fixed-shape XLA decoding for the pinned InstaManip/Transformers 4.30 stack.

The released model uses tuple KV caches whose sequence dimension grows once per
token.  That is numerically fine on CUDA, but it makes PyTorch/XLA compile a new
17B graph for every generated token.  This adapter keeps the released prefill,
greedy token rule, scope mask, and hidden-state readout while storing KV values
in a fixed-capacity tensor during single-token decoding.
"""
from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F


STATIC_CACHE_SCHEMA = "instamanip_xla_fused_image_block_v1"


def _device_data(value, *, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Upload changing values as device data instead of baking XLA constants."""
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=dtype)
    else:
        value = torch.tensor(value, dtype=dtype, device="cpu")
    return value.to(device=device)


def _store_hidden_slot(
    buffer: torch.Tensor,
    hidden: torch.Tensor,
    slot: int,
    slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shape-stably retain one image-token hidden state in a fixed buffer."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != 1:
        raise RuntimeError("fixed XLA hidden-state buffer requires [1,1,D] input")
    if buffer.ndim != 3 or buffer.shape[0] != 1 or buffer.shape[2] != hidden.shape[2]:
        raise RuntimeError("fixed XLA hidden-state buffer shape differs")
    if not 0 <= int(slot) < int(buffer.shape[1]):
        raise RuntimeError("fixed XLA hidden-state buffer slot is out of range")
    if slots is None:
        slots = _device_data(
            list(range(int(buffer.shape[1]))), dtype=torch.long, device=buffer.device)
    if tuple(slots.shape) != (int(buffer.shape[1]),):
        raise RuntimeError("fixed XLA hidden-state slot selector shape differs")
    selected = (slots == _device_data([slot], dtype=torch.long, device=buffer.device)[0]).view(
        1, int(buffer.shape[1]), 1)
    return torch.where(selected, hidden.expand_as(buffer), buffer)


def _static_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor | None = None,
    past_key_value: tuple[torch.Tensor, ...] | None = None,
    ids_exemplar_target_mask: torch.BoolTensor | None = None,
    scope_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
):
    original = type(self)._instamanip_xla_original_forward
    if past_key_value is None or len(past_key_value) != 3:
        return original(
            self,
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            ids_exemplar_target_mask=ids_exemplar_target_mask,
            scope_mask=scope_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    if position_ids is None or hidden_states.shape[0] != 1:
        raise RuntimeError("fixed XLA KV decoding requires batch=1 and explicit positions")

    key_cache, value_cache, cache_tag = past_key_value
    capacity = int(key_cache.shape[2])
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise RuntimeError("fixed XLA KV key/value cache shapes differ")
    bsz, q_len, _ = hidden_states.shape
    query_states = self.q_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(value_states, seq_len=capacity)
    query_states, key_states = type(self)._instamanip_xla_apply_rotary(
        query_states, key_states, cos, sin, position_ids)

    positions = position_ids.reshape(-1)
    if int(positions.numel()) != int(q_len):
        raise RuntimeError("fixed XLA KV position count differs from query length")
    slots = torch.arange(capacity, device=key_states.device, dtype=position_ids.dtype)
    selected = (slots.unsqueeze(0) == positions.unsqueeze(1))
    selected_numeric = selected.to(key_states.dtype)
    key_updates = torch.einsum("qc,bhqd->bhcd", selected_numeric, key_states)
    value_updates = torch.einsum("qc,bhqd->bhcd", selected_numeric, value_states)
    written_slots = selected.any(dim=0).view(1, 1, capacity, 1)
    key_cache = torch.where(written_slots, key_updates, key_cache)
    value_cache = torch.where(written_slots, value_updates, value_cache)
    present = (key_cache, value_cache, cache_tag) if use_cache else None

    query = query_states * (1 / query_states.shape[-1] ** 0.5)
    scores = query @ key_cache.transpose(-2, -1)
    valid = (slots.unsqueeze(0) <= positions.unsqueeze(1)).view(1, 1, q_len, capacity)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    if scope_mask is not None:
        if tuple(scope_mask.shape) not in ((1, 1, 1, capacity), (1, 1, q_len, capacity)):
            raise RuntimeError("fixed XLA decode scope mask shape differs")
        scores = scores + scope_mask
    probabilities = scores.softmax(-1)
    attention = probabilities @ value_cache
    attention = attention.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    attention = self.o_proj(attention)
    return attention, (probabilities if output_attentions else None), present


def install_static_attention(attention_class: type, apply_rotary) -> None:
    """Install the XLA-only three-tensor cache path once on the pinned class."""
    if hasattr(attention_class, "_instamanip_xla_original_forward"):
        return
    attention_class._instamanip_xla_original_forward = attention_class.forward
    attention_class._instamanip_xla_apply_rotary = staticmethod(apply_rotary)
    attention_class.forward = _static_attention_forward


def _pad_cache(past_key_values, capacity: int, prompt_length: int):
    padding = capacity - prompt_length
    if padding <= 0:
        raise RuntimeError("fixed XLA KV capacity does not exceed the prompt length")
    result = []
    for key, value in past_key_values:
        if key.shape[2] != prompt_length or tuple(value.shape) != tuple(key.shape):
            raise RuntimeError("released prefill KV cache shape differs")
        key = F.pad(key, (0, 0, 0, padding))
        value = F.pad(value, (0, 0, 0, padding))
        tag = torch.zeros((1,), dtype=torch.int32, device=key.device)
        result.append((key, value, tag))
    return tuple(result)


def _run_forced_image_block(
    self,
    *,
    image_ids: list[int],
    past_key_values,
    used: int,
    capacity: int,
    ids_exemplar_target_mask: torch.Tensor,
    scope_mask: torch.Tensor,
    device: torch.device | str,
):
    """Evaluate the released deterministic image-token span in one causal pass.

    Once ``<img>`` has been generated, the official logits processor forces
    ``<img_00000> ... <img_00063> </img>``.  Teacher-forcing that known suffix
    in one causal forward is mathematically the same autoregressive computation,
    while avoiding 65 separate 17B XLA launches.  The returned hidden states are
    exactly the 64 payload-token positions consumed by the released resampler.
    """
    if len(image_ids) < 3:
        raise RuntimeError("released image-token sequence is incomplete")
    payload = image_ids[1:-1]
    suffix = image_ids[1:]
    if len(payload) + 2 != len(image_ids):
        raise RuntimeError("released image-token payload contract differs")
    for item in past_key_values:
        if len(item) != 3:
            raise RuntimeError("fixed XLA KV cache entry must contain key, value, and tag")
        key, value, _ = item
        if (int(key.shape[2]) != int(capacity) or tuple(value.shape) != tuple(key.shape)
                or not 0 < int(used) < int(capacity)
                or int(used) + len(suffix) > int(capacity)):
            raise RuntimeError("fixed XLA KV block cache contract differs")
    tokens = _device_data([suffix], dtype=torch.long, device=device)
    positions = _device_data(
        [list(range(int(used), int(used) + len(suffix)))],
        dtype=torch.long,
        device=device,
    )
    output = self.llm(
        input_ids=tokens,
        position_ids=positions,
        past_key_values=past_key_values,
        attention_mask=None,
        ids_exemplar_target_mask=ids_exemplar_target_mask,
        scope_mask=scope_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = output.hidden_states[-1][:, :len(payload), :]
    if tuple(hidden.shape[:2]) != (1, len(payload)):
        raise RuntimeError("fused image-token hidden span differs")
    scores = output.logits[:, -1, :]
    updated = output.past_key_values
    del output
    return hidden, scores, updated


def _official_processor(scores: torch.Tensor, current_token: int, image_ids: list[int]) -> torch.Tensor:
    """Shape-stable equivalent of the released AutoImageTokenGenerationProcessor."""
    vocab = int(scores.shape[-1])
    if current_token in image_ids[:-1]:
        target = image_ids[image_ids.index(current_token) + 1]
        selector = torch.zeros((vocab,), dtype=scores.dtype, device="cpu")
        selector[target] = 1
        selector = selector.to(scores.device).unsqueeze(0)
        return torch.where(selector.to(torch.bool), scores.max(dim=-1, keepdim=True).values + 10, scores)
    selector = torch.zeros((vocab,), dtype=scores.dtype, device="cpu")
    selector[image_ids[1:]] = 1
    selector = selector.to(scores.device).unsqueeze(0)
    return torch.where(selector.to(torch.bool), torch.zeros_like(scores), scores)


def _blocked_image_selector(scores: torch.Tensor, image_ids: list[int]) -> torch.Tensor:
    """Build the released non-image-step mask once instead of once per token."""
    selector = torch.zeros((int(scores.shape[-1]),), dtype=torch.bool, device="cpu")
    selector[image_ids[1:]] = True
    return selector.to(scores.device).unsqueeze(0)


def _static_generate(
    self,
    tokenizer,
    prompt=None,
    input_ids=None,
    image_embeds=None,
    embeds_cmp_mask=None,
    ids_cmp_mask=None,
    ids_exemplar_target_mask=None,
    scope_mask=None,
    logits_processor=None,
    num_img_gen_tokens=64,
    temperature=0.7,
    num_beams=1,
    max_new_tokens=120,
    top_p=0.5,
    device="xla",
    patch_positions=None,
):
    del temperature, top_p
    if num_beams != 1:
        raise RuntimeError("fixed XLA decoding admits only the released greedy num_beams=1 path")
    if prompt is not None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)
    if input_ids is None or input_ids.shape[0] != 1:
        raise RuntimeError("fixed XLA decoding requires one tokenized prompt")
    input_ids = input_ids.to(device=device)
    input_embeds = self.llm.get_input_embeddings()(input_ids)
    _, prompt_length, dim = input_embeds.shape

    if image_embeds is not None:
        if embeds_cmp_mask is None or ids_cmp_mask is None:
            raise RuntimeError("released image embedding masks are required")
        with torch.no_grad():
            image_embeds_lm = {key: self.input_resampler(value) for key, value in image_embeds.items()}
            if self.add_patch_pos and patch_positions is not None:
                patch_positions = {key: value.to(image_embeds_lm[key]) for key, value in patch_positions.items()}
                relative = {key: torch.mm(torch.cat([value, 1 - value], dim=-1) / 2,
                                          self.patch_pos_embed).unsqueeze(1)
                            for key, value in patch_positions.items()}
                for key in image_embeds_lm:
                    image_embeds_lm[key] = image_embeds_lm[key] + relative[key]
        if image_embeds_lm["exemplar_source"].shape[0] == 1:
            combined = torch.cat([image_embeds_lm["exemplar_source"],
                                  image_embeds_lm["exemplar_target"], image_embeds_lm["new"]], dim=0)
            combined_mask = torch.cat([embeds_cmp_mask["exemplar_source"],
                                       embeds_cmp_mask["exemplar_target"], embeds_cmp_mask["new"]], dim=0)
        else:
            values, masks = [], []
            for index in range(image_embeds_lm["exemplar_source"].shape[0]):
                values.extend([image_embeds_lm["exemplar_source"][index:index + 1],
                               image_embeds_lm["exemplar_target"][index:index + 1]])
                masks.extend([embeds_cmp_mask["exemplar_source"][index],
                              embeds_cmp_mask["exemplar_target"][index]])
            values.append(image_embeds_lm["new"]); masks.append(embeds_cmp_mask["new"])
            combined, combined_mask = torch.cat(values, dim=0), torch.cat(masks, dim=0)
        input_embeds[ids_cmp_mask] = combined[combined_mask].view(-1, dim)

    if logits_processor is None:
        from src.models.mllm.generation import AutoImageTokenGenerationProcessor
        logits_processor = [AutoImageTokenGenerationProcessor(
            tokenizer=tokenizer, num_img_gen_tokens=num_img_gen_tokens)]
    if len(logits_processor) != 1 or not hasattr(logits_processor[0], "img_ids_list"):
        raise RuntimeError("fixed XLA decoding requires the released image-token logits processor")
    image_ids = [int(value) for value in logits_processor[0].img_ids_list]
    boi_id = tokenizer.encode("<img>", add_special_tokens=False)[0]
    eoi_id = tokenizer.encode("</img>", add_special_tokens=False)[0]
    eos = self.llm.config.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}

    capacity = prompt_length + int(max_new_tokens)
    target_mask = ids_exemplar_target_mask.detach().cpu()
    target_positions = torch.where(target_mask)[1]
    if target_positions.numel() == 0:
        raise RuntimeError("released exemplar target mask is empty")
    cutoff = int(target_positions[-1]) + 1
    static_scope_cpu = torch.zeros((1, 1, 1, capacity), dtype=input_embeds.dtype)
    static_scope_cpu[..., :cutoff + 1] = -float("inf")
    static_scope = static_scope_cpu.to(input_embeds.device)

    with torch.no_grad():
        output = self.llm(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            ids_exemplar_target_mask=ids_exemplar_target_mask,
            scope_mask=scope_mask,
            use_cache=True,
            # Prefill hidden states are never part of the released image
            # readout; only the subsequent image-token states are consumed.
            output_hidden_states=False,
            return_dict=True,
        )
        past = _pad_cache(output.past_key_values, capacity, prompt_length)
        scores = output.logits[:, -1, :]
        current_token = int(input_ids[0, -1].detach().cpu())
        generated: list[int] = []
        image_successors = {value: image_ids[index + 1]
                            for index, value in enumerate(image_ids[:-1])}
        image_payload_ids = set(image_ids[1:-1])
        blocked_selector = _blocked_image_selector(scores, image_ids)
        # Keeping one lazy tensor per image token makes every mark_step expose
        # a larger output tuple and therefore compile a different program.
        # A single fixed buffer keeps both graph shapes and live-output count
        # constant across the entire forced image-token span.
        empty_image_hidden = _device_data(
            torch.zeros((1, int(num_img_gen_tokens), int(dim)), dtype=input_embeds.dtype),
            dtype=input_embeds.dtype, device=input_embeds.device)
        active_image_hidden = empty_image_hidden
        image_hidden_slots = _device_data(
            list(range(int(num_img_gen_tokens))), dtype=torch.long, device=input_embeds.device)
        active_image_count = 0
        completed_image_features: list[torch.Tensor] = []

        step = 0
        while step < int(max_new_tokens):
            # The released processor makes the entire image-token suffix
            # deterministic after <img>.  Execute those 65 causal positions as
            # one graph while retaining the exact 64 hidden states used by the
            # official output resampler.
            if current_token == int(boi_id):
                suffix = image_ids[1:]
                if step + len(suffix) > int(max_new_tokens):
                    raise RuntimeError("max_new_tokens truncates the forced image-token block")
                used = prompt_length + step
                fused_hidden, scores, past = _run_forced_image_block(
                    self,
                    image_ids=image_ids,
                    past_key_values=past,
                    used=used,
                    capacity=capacity,
                    ids_exemplar_target_mask=ids_exemplar_target_mask,
                    scope_mask=static_scope,
                    device=input_ids.device,
                )
                completed_image_features.append(fused_hidden[0])
                generated.extend(suffix)
                current_token = int(eoi_id)
                step += len(suffix)
                if input_ids.device.type == "xla":
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                if current_token in eos_ids or step >= int(max_new_tokens):
                    break
                continue

            forced_image_step = current_token in image_successors
            if forced_image_step:
                # The released processor deterministically forces this exact
                # successor. Avoid a full-vocabulary selector and TPU->CPU
                # scalar readback for every one of the image-token steps.
                next_token = image_successors[current_token]
                next_token_tensor = _device_data(
                    [next_token], dtype=torch.long, device=input_ids.device)
            else:
                processed = torch.where(blocked_selector, torch.zeros_like(scores), scores)
                next_token_tensor = processed.argmax(dim=-1)
                next_token = int(next_token_tensor.detach().cpu())
            generated.append(next_token)
            if next_token == int(eoi_id):
                if active_image_count != int(num_img_gen_tokens):
                    raise RuntimeError("released image-token hidden-state span is incomplete")
                completed_image_features.append(active_image_hidden[0])
                active_image_hidden = empty_image_hidden
                active_image_count = 0
            step += 1
            if next_token in eos_ids or step == int(max_new_tokens):
                break
            token = next_token_tensor.reshape(1, 1)
            position = _device_data(
                [[prompt_length + step - 1]], dtype=torch.long, device=input_ids.device)
            output = self.llm(
                input_ids=token,
                position_ids=position,
                past_key_values=past,
                attention_mask=None,
                ids_exemplar_target_mask=ids_exemplar_target_mask,
                scope_mask=static_scope,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values
            last_hidden = output.hidden_states[-1] if next_token in image_payload_ids else None
            scores = output.logits[:, -1, :]
            # Drop the model-output container before mark_step so the unused
            # per-layer hidden-state tuple cannot become an XLA output root.
            del output
            if next_token in image_payload_ids:
                active_image_hidden = _store_hidden_slot(
                    active_image_hidden, last_hidden, active_image_count,
                    image_hidden_slots)
                active_image_count += 1
            current_token = next_token
            if input_ids.device.type == "xla":
                # Bound every fixed-shape decode step. Changing tokens,
                # positions, and hidden-buffer slots enter as device data, so
                # they do not create one compile hash per numerical value.
                import torch_xla.core.xla_model as xm
                xm.mark_step()

        generated_tensor = torch.tensor(generated, dtype=torch.long)
        eoi_indices = torch.where(generated_tensor == int(eoi_id))[0].tolist()
        text_mask = torch.ones_like(generated_tensor, dtype=torch.bool)
        image_features = []
        for eoi_index in eoi_indices:
            start = eoi_index - int(num_img_gen_tokens)
            if start < 0:
                raise RuntimeError("released image-token hidden-state span is incomplete")
            text_mask[start:eoi_index] = False
        if len(completed_image_features) != len(eoi_indices):
            raise RuntimeError("completed image feature count differs from generated image blocks")
        if completed_image_features:
            image_features = torch.stack(completed_image_features)
            image_output = self.output_resampler(image_features)
        else:
            image_output = None
        text_mask[generated_tensor == int(boi_id)] = False
        text = tokenizer.decode(generated_tensor[text_mask].tolist(), skip_special_tokens=False)
    return {"text": text, "has_img_output": bool(eoi_indices), "img_gen_feat": image_output,
            "num_gen_imgs": len(eoi_indices)}


def bind_static_generate(agent: Any) -> None:
    """Bind the fixed-shape implementation to this audited agent instance."""
    agent.generate = types.MethodType(_static_generate, agent)
    agent.xla_static_cache_schema = STATIC_CACHE_SCHEMA
