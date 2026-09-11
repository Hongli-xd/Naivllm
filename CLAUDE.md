# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MinivLLM is a from-scratch implementation of a vLLM-style LLM inference engine, built for educational clarity and performance. It implements paged attention, flash attention, prefix caching, CUDA graphs, and tensor parallelism — currently targeting Qwen3 models on NVIDIA GPUs.

## Commands

```bash
uv sync                           # Install dependencies
uv run python main.py             # Run inference demo (Qwen3-0.6B)
uv run python benchmark_prefilling.py  # Benchmark prefill attention
uv run python benchmark_decoding.py    # Benchmark decode attention
uv run python benchmark_tps.py    # Throughput benchmark
uv run pytest                     # Run tests
uv run black src/                 # Format code
uv run isort src/                 # Sort imports
```

Python 3.11 only. Uses `uv` as the package manager.

## Architecture

The engine has four layers: `layers/` → `models/` → `engine/` → user API.

### Inference Flow

```
LLMEngine.generate(prompts)
  → tokenize → Sequence objects → Scheduler.waiting queue
  → loop: Scheduler.schedule() → (sequences, is_prefill)
         → ModelRunner.call("run", sequences, is_prefill)
              → prepare_prefill() or prepare_decode()
              → run_model() [CUDA graph for decode]
              → sample() → token_ids
         → Scheduler.postprocess() → check EOS, deallocate finished
```

### Key Components

**`engine/sequence.py`** — `Sequence` tracks token_ids, status (WAITING/RUNNING/FINISHED), and `block_table` (list of KV cache block indices). Serializable for IPC between master/worker processes.

**`engine/block_manager.py`** — `BlockManager` allocates fixed-size KV cache blocks (default 256 tokens). Implements prefix caching via xxhash content addressing with reference counting. Shared blocks are reused across sequences with identical prefixes.

**`engine/scheduler.py`** — Two queues: `waiting` and `running`. Prefill always takes priority over decode. Preempts lowest-priority running sequence when memory is full.

**`engine/model_runner.py`** — Bridges sequences and model execution. Runs in worker processes (rank 0 on master). Prepares batched inputs differently for prefill vs decode:
- Prefill: flattens all tokens, builds `cu_seqlens` for FlashAttention varlen API
- Decode: one token per sequence, slot mapping into KV cache blocks
- Pre-captures CUDA graphs for decode at batch sizes [1, 2, 4, 8, 16+]

**`engine/llm_engine.py`** — Master process. Spawns rank 1+ worker processes for tensor parallelism. Communicates via shared memory `multiprocessing.Event`.

**`models/qwen3.py`** — Qwen3 decoder: Embedding → 28 × TransformerLayer → LM head. RMS LayerNorm applied to Q and K (not V). GQA with `num_kv_heads=8`, `num_heads=16`.

### Tensor Parallelism Pattern

- QKV projection: `ColumnParallelLinear` (splits output dim across GPUs)
- Attention: each GPU handles its own heads independently (no communication)
- Output/down projection: `RowParallelLinear` (all_reduce to sum partial results)
- Vocab embedding/LM head: sharded across vocab dimension

### Layers (`src/myvllm/layers/`)

| File | Key Classes |
|------|-------------|
| `linear.py` | `ColumnParallelLinear`, `RowParallelLinear`, `QKVColumnParallelLinear`, `MergedColumnParallelLinear` |
| `attention.py` | FlashAttention (prefill) + Triton paged attention kernel (decode) |
| `embedding_head.py` | `VocabParallelEmbedding`, `ParallelLMHead` |
| `rotary_embedding.py` | RoPE with YARN/NTK scaling |
| `layernorm.py` | RMS LayerNorm with optional residual |
| `sampler.py` | Temperature-based token sampling (rank 0 only) |

### Context Passing

`utils/context.py` provides thread-local storage for batch metadata (slot mappings, block tables, cu_seqlens) that layers need during the forward pass without explicit argument threading.

## Configuration

The engine is configured via a plain dict passed to `LLMEngine`. Key fields:

```python
{
    'max_num_sequences': 16,        # max concurrent sequences
    'max_num_batched_tokens': 1024, # max tokens per step
    'max_cached_blocks': 1024,      # total KV cache blocks
    'block_size': 256,              # tokens per KV block
    'world_size': 1,                # number of GPUs
    'model_name_or_path': '...',
    # model architecture fields: vocab_size, hidden_size, num_heads, etc.
}
```

## Learning Guide

`HowToApproachvLLM.md` is a 765-line walkthrough of the entire codebase in implementation order. Read it before making significant changes to understand design decisions.
