from __future__ import annotations

from typing import Any

from transformers import AutoConfig


def build_engine_config(
    model_name_or_path: str,
    *,
    max_num_sequences: int = 16,
    max_num_batched_tokens: int = 4096,
    max_model_length: int = 2048,
    block_size: int = 16,
    world_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = False,
    max_cached_blocks: int | None = None,
) -> dict[str, Any]:
    """Build a NaivLLM configuration from a Hugging Face model config.

    NaivLLM currently implements the Qwen3 architecture. Reading model fields
    here prevents benchmark scripts and examples from silently drifting away
    from the checkpoint they load.
    """
    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(hf_config, "model_type", "")
    if model_type != "qwen3":
        raise ValueError(
            f"Unsupported model type {model_type!r}; NaivLLM currently supports qwen3"
        )

    num_heads = hf_config.num_attention_heads
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // num_heads)
    dtype = getattr(hf_config, "dtype", None)
    dtype_name = str(dtype).removeprefix("torch.") if dtype is not None else "bfloat16"
    rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
    rope_theta = getattr(hf_config, "rope_theta", None) or rope_parameters.get(
        "rope_theta", 10000
    )

    config: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "max_num_sequences": max_num_sequences,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_batch_tokens": max_num_batched_tokens,
        "max_model_length": max_model_length,
        "block_size": block_size,
        "world_size": world_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_kv_heads": hf_config.num_key_value_heads,
        "intermediate_size": hf_config.intermediate_size,
        "num_layers": hf_config.num_hidden_layers,
        "tie_word_embeddings": hf_config.tie_word_embeddings,
        "base": rope_theta,
        "rms_norm_epsilon": hf_config.rms_norm_eps,
        "qkv_bias": getattr(hf_config, "attention_bias", False),
        "ffn_bias": getattr(hf_config, "mlp_bias", False),
        "scale": 1.0,
        "max_position": hf_config.max_position_embeddings,
        "eos": hf_config.eos_token_id,
        "dtype": dtype_name,
    }
    if max_cached_blocks is not None:
        config["max_cached_blocks"] = max_cached_blocks
    else:
        tokens_per_sequence = (max_model_length + block_size - 1) // block_size
        config["max_cached_blocks"] = max_num_sequences * tokens_per_sequence
    return config
