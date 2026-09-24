# NaivLLM

NaivLLM is a compact Qwen3 inference engine built to make paged KV caching,
continuous batching, tensor parallelism, Triton attention kernels, and CUDA
Graph decode visible in a small codebase. It is an educational and experimental
engine, not a drop-in replacement for vLLM.

## Narrow target

The performance experiment targets a deliberately small regime: one GPU,
Qwen3 0.6B-3B, batch sizes 1-8, and short decode-bound requests. NaivLLM removes
generality overhead and captures model forward, logits, and sampling in one CUDA
Graph. vLLM is expected to regain the lead as batches or contexts grow and the
workload becomes compute-bound.

The benchmark also measures two secondary scenarios:

- process start to engine ready/first token, host RSS, and GPU memory footprint;
- a fixed RAG prefix, including NaivLLM's explicitly pinned prefix blocks.

No performance claim is hard-coded in this repository. Use the benchmark to
find the crossover point on the actual hardware and report the full curve.

## Run

```bash
python -m pip install -e '.[dev]'
python main.py
```

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
