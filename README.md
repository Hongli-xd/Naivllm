# NaivLLM

NaivLLM is a compact Qwen3 inference engine built to make paged KV caching,
continuous batching, tensor parallelism, Triton attention kernels, and CUDA
Graph decode visible in a small codebase. It is an educational and experimental
engine, not a drop-in replacement for vLLM.

## Vertical: RAG / knowledge-base QA latency engine

Beyond the narrow decode regime, NaivLLM now targets one vertical workload
end-to-end: **grounded QA over a long, shared document prefix** — the shape of
every knowledge-base assistant (enterprise docs, support bots, in-IDE doc
lookups):

- a long fixed prefix (system + retrieved document) shared by all requests,
  kept resident through the explicitly-pinned prefix blocks;
- short, extractive answers that quote the document — which makes
  **n-gram speculative decoding** (prompt lookup, no draft model) extremely
  effective: measured acceptance 83-90% on the RAG benchmark;
- speculative verification runs as a **captured CUDA Graph** (one replay per
  round), so a round costs one graph launch instead of ~28 layers of eager
  dispatch.

Fair-comparison rules used in `benchmarks/rag_bench.py`: every backend runs
the same Qwen3-0.6B weights (bf16/fp16), the same tokenized prompts, greedy
decoding, and the same GPU. vLLM is given its own ngram speculative decoding
with identical K / n-gram bounds; Ollama runs fp16 on a dedicated GPU-pinned
server (it has no ngram spec decoding and no smaller Qwen3 draft for 0.6B).

## Speculative decoding

`myvllm.spec_engine` implements lossless speculative decoding (Leviathan et
al.) with pluggable drafters:

- `NgramDrafter` — prompt-lookup drafting over prompt+output via
  `bytes.rfind` n-gram matching; no draft model, ideal for extractive RAG.
- `ModelDrafter` — a second, smaller Qwen3 drafting K autoregressive steps
  through its own paged KV cache (`draft_block_table`) and CUDA graphs.

Verification is a single multi-token forward over
`[last_confirmed, d_1..d_K]` through a dedicated Triton paged-attention
kernel with per-row causal positions, replayed as a captured CUDA graph per
batch bucket. Rejection sampling supports greedy (exact-match), stochastic
with draft distributions (min(1, p_t/p_d) + residual resample), and
distribution-free drafts (accept iff the draft equals a target sample) — all
provably lossless. Greedy spec output is verified token-identical to the
engine's own plain decoding (`scripts/check_hf_alignment.py`).

Run it:

```bash
PYTHONPATH=src python spec_main.py --model /path/to/Qwen3-0.6B --method ngram --k 5 --temperature 0
```

## Run

```bash
python -m pip install -e '.[dev]'
python main.py
```

## RAG benchmark

```bash
PYTHONPATH=src python benchmarks/rag_bench.py \
  --model /models/Qwen3-0.6B \
  --gpu 1 \
  --backends naivllm,vllm-spec,vllm,ollama \
  --batches 1,2,4,8 \
  --doc-tokens 1200 \
  --max-tokens 48 \
  --repeats 3 \
  --k 5 \
  --naiv-method ngram \
  --ollama-model qwen3:0.6b-fp16
```

Ollama expects a dedicated server pinned to the same GPU (fp16 weights, raw
prompts):

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 \
  OLLAMA_MODELS=~/ollama-gpu1/models OLLAMA_KEEP_ALIVE=-1 \
  OLLAMA_NUM_PARALLEL=8 OLLAMA_CONTEXT_LENGTH=4096 ollama serve &
OLLAMA_HOST=127.0.0.1:11435 ollama pull qwen3:0.6b-fp16
```

## Honest scope

The narrow decode regime (one GPU, Qwen3 0.6B-3B, batch 1-8, short
decode-bound requests) is where NaivLLM is competitive. vLLM regains the
lead as batches or contexts grow and the workload becomes compute-bound;
measure the crossover on your hardware with the harnesses above rather than
trusting any fixed claim.

Configuration is derived from the checkpoint's Hugging Face `config.json`:

```python
from myvllm import LLMEngine, build_engine_config

config = build_engine_config(
    "/models/Qwen3-0.6B",
    max_num_sequences=8,
    max_model_length=1024,
)
engine = LLMEngine(config)
```

## Fair comparison with vLLM

Run each backend in its own process on the same GPU. vLLM uses prefix caching,
BF16, CUDA Graphs (`enforce_eager=False`), and the same model/context limits.

```bash
PYTHONPATH=src python benchmarks/narrow_regime.py \
  --model /models/Qwen3-0.6B \
  --gpu 1 \
  --batches 1,2,4,8 \
  --contexts 32,128,512 \
  --output-tokens 32 \
  --repeats 3
```

Artifacts are written under `benchmark-results/narrow-regime/`:

- `results.json`: environment, cold-start, memory, curve, and prefix metrics;
- `curve.csv`: machine-readable batch/context sweep;
- `curve.png`: output-throughput curves for every batch size.

For an ablation, add `--enforce-eager` to disable NaivLLM CUDA Graphs. Do not
compare that result to a graph-enabled vLLM result as the primary headline;
use it only to quantify NaivLLM's graph contribution.

## Implementation map

- `src/myvllm/engine`: scheduler, block manager, model runner, CUDA Graph capture
- `src/myvllm/layers/attention.py`: varlen prefill and vectorized paged decode
- `src/myvllm/models/qwen3.py`: compact Qwen3 implementation
- `benchmarks/narrow_regime.py`: isolated, reproducible comparison harness
- `tests`: CPU unit tests for cache ownership and scheduling invariants

## Current limits

- Qwen3 is the only supported architecture.
- Prefix-reuse prefill uses a correctness-oriented PyTorch path; ordinary
  prefill and decode use Triton kernels.
- CUDA Graph capture uses fixed batch buckets, so requests may be padded to the
  next captured size.
- The benchmark reports observed results, not production-service SLOs.
