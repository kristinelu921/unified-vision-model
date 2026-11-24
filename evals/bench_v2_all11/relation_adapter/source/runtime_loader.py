"""Released runtime with the correct official block classes for TPU wrapping.

The archived loader is preserved. This function changes wrapping boundaries only;
weights, forward functions, precision, conditioning and sampling remain the same.
"""
import functools
import gc
import sys


def require_official_blocks(transformer, classes):
    dual = list(transformer.transformer_blocks)
    single = list(transformer.single_transformer_blocks)
    if len(dual) != 19 or len(single) != 38:
        raise ValueError('Pinned FLUX release must contain 19 dual and 38 single blocks')
    if not all(isinstance(x, classes[0]) for x in dual) or not all(isinstance(x, classes[1]) for x in single):
        raise ValueError('Wrapping policy must select the actual official block classes')
    return {id(x) for x in dual + single}


def verify_wrapped_blocks(transformer, wrapper_class, required):
    wrappers = [m for m in transformer.modules() if isinstance(m, wrapper_class)]
    wrapped = {id(m.module) for m in wrappers if id(m.module) in required}
    if wrapped != required or len(wrappers) != 58:
        raise ValueError('Every one of 57 released transformer blocks must be wrapped exactly once, plus the root')
    return {'official_blocks': len(required), 'wrapped_blocks': len(wrapped),
            'wrappers_including_root': len(wrappers), 'policy': 'official transformer_flux block classes'}


def load_runtime(baseline, args, device, mesh, task_prompts=None):
    PROMPT, ALPHA = baseline.PROMPT, baseline.ALPHA
    MLPProjModel = baseline.MLPProjModel
    recursive_shard_output = baseline.recursive_shard_output
    import torch
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel
    from transformers import AutoProcessor, SiglipVisionModel
    sys.path.insert(0, str(args.source_dir.resolve()))
    from attention_processor import IPAFluxAttnProcessor2_0
    from src.pipeline_pe_clone import FluxPipeline
    from transformer_flux import FluxTransformer2DModel, FluxTransformerBlock, FluxSingleTransformerBlock
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(args.flux_dir, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    pipe = FluxPipeline.from_pretrained(args.flux_dir, transformer=transformer, torch_dtype=dtype, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(PROMPT, PROMPT, device=torch.device("cpu"), max_sequence_length=512)
    task_embeddings = {}
    with torch.no_grad():
        for task, text in (task_prompts or {}).items():
            pe, pp, _ = pipe.encode_prompt(text, text, device=torch.device("cpu"), max_sequence_length=512)
            task_embeddings[task] = (pe.to(device, dtype=dtype), pp.to(device, dtype=dtype))
    pipe.load_lora_weights(args.adapter_dir, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora(); pipe.unload_lora_weights()
    processors = {}
    for name in transformer.attn_processors:
        if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
            processors[name] = IPAFluxAttnProcessor2_0(
                hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                cross_attention_dim=transformer.config.joint_attention_dim, num_tokens=128, scale=ALPHA).to(dtype=dtype)
        else:
            processors[name] = transformer.attn_processors[name]
    transformer.set_attn_processor(processors)
    state = torch.load(args.adapter_dir / "ip_adapter-100000.bin", map_location="cpu", weights_only=True, mmap=True)
    projection_state = state["image_proj"]
    if next(iter(projection_state)).startswith("module."):
        projection_state = {k.removeprefix("module."): v for k, v in projection_state.items()}
    projection = MLPProjModel.build(torch, transformer.config.joint_attention_dim, 128).to(dtype=dtype)
    projection.load_state_dict(projection_state, strict=True)
    adapter_state = state["ip_adapter"]
    if next(iter(adapter_state)).startswith("module."):
        adapter_state = {k.removeprefix("module."): v for k, v in adapter_state.items()}
    layers = torch.nn.ModuleList(transformer.attn_processors.values())
    missing, unexpected = layers.load_state_dict(adapter_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"adapter state mismatch missing={missing} unexpected={unexpected}")
    image_encoder = SiglipVisionModel.from_pretrained(args.siglip_dir, torch_dtype=dtype, local_files_only=True).eval()
    image_processor = AutoProcessor.from_pretrained(args.siglip_dir, local_files_only=True)
    pipe.text_encoder = pipe.text_encoder_2 = pipe.tokenizer = pipe.tokenizer_2 = None
    del state, projection_state, adapter_state, layers
    gc.collect()
    block_classes = (FluxTransformerBlock, FluxSingleTransformerBlock)
    block_ids = require_official_blocks(transformer, block_classes)
    auto_wrap = functools.partial(transformer_auto_wrap_policy,
                                  transformer_layer_cls={FluxTransformerBlock, FluxSingleTransformerBlock})
    transformer = SpmdFullyShardedDataParallel(transformer.eval(), mesh=mesh, auto_wrap_policy=auto_wrap,
                                                shard_output=recursive_shard_output)
    wrapping = verify_wrapped_blocks(transformer, SpmdFullyShardedDataParallel, block_ids)
    pipe.transformer = transformer
    pipe.vae = pipe.vae.eval().to(device)
    return {"block_wrapping": wrapping, "task_embeddings": task_embeddings, "pipe": pipe, "image_encoder": image_encoder.to(device), "image_processor": image_processor,
            "projection": projection.eval().to(device), "prompt_embeds": prompt_embeds.to(device, dtype=dtype),
            "pooled_prompt_embeds": pooled_prompt_embeds.to(device, dtype=dtype), "dtype": dtype}
