"""P=1 shape adaptation for released LVM 7B Bench v2 generation."""
from __future__ import annotations


N_DEMOS = 1
TOKENS_PER_FRAME = 256
CONTEXT_FRAMES = 2 * N_DEMOS + 1
EPISODE_CONTEXT_TOKENS = CONTEXT_FRAMES * TOKENS_PER_FRAME
EPISODE_NEW_TOKENS = TOKENS_PER_FRAME
EPISODE_TOTAL_TOKENS = EPISODE_CONTEXT_TOKENS + EPISODE_NEW_TOKENS
VQ_CODEBOOK_SIZE = 8192
VISUAL_PAD_EOS_TOKEN_ID = VQ_CODEBOOK_SIZE


def make_vqlm_7b_config(config_class):
    config = config_class.load_config("vqlm_7b")
    config.bos_token_id = 1
    # Every ID below 8192 is a visual code. The checkpoint's text-style EOS=2
    # would stop an image early, so use the first suppressed special ID.
    config.eos_token_id = VISUAL_PAD_EOS_TOKEN_ID
    config.tie_word_embeddings = False

    # P=1 produces a 1024-token prompt-plus-generation sequence. Use one exact
    # attention block at that length. Cached generation still needs the
    # official dense MLP branch for its one-token decode steps.
    config.scan_query_chunk_size = EPISODE_TOTAL_TOKENS
    config.scan_key_chunk_size = EPISODE_TOTAL_TOKENS
    config.scan_mlp = False
    return config


def adaptation_record() -> dict:
    return {
        "schema": "lvm_7b_bench_v2_p1_runtime_v1",
        "demos": N_DEMOS,
        "context_frames": CONTEXT_FRAMES,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "context_tokens": EPISODE_CONTEXT_TOKENS,
        "new_tokens": EPISODE_NEW_TOKENS,
        "total_tokens": EPISODE_TOTAL_TOKENS,
        "model_max_sequence_length": 4096,
        "scan_attention": True,
        "scan_query_chunk_size": EPISODE_TOTAL_TOKENS,
        "scan_key_chunk_size": EPISODE_TOTAL_TOKENS,
        "scan_mlp": False,
        "weight_changes": 0,
        "visual_token_range": [0, VQ_CODEBOOK_SIZE - 1],
        "pad_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "eos_token_id": VISUAL_PAD_EOS_TOKEN_ID,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 50,
        "candidates": 1,
    }
