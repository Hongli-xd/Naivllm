"""RAG precheck: n-gram acceptance and latency vs plain decoding."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import torch

from rag_bench import build_document, build_prompts
from myvllm.config import build_engine_config
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_engine import SpecLLMEngine
from myvllm.spec_engine.drafters import NgramDrafter
from transformers import AutoTokenizer

MODEL = "/home/selom/models/Qwen3-0.6B"

tok = AutoTokenizer.from_pretrained(MODEL)
doc, facts = build_document(tok, 1200)
prompts, meta = build_prompts(tok, doc, facts, 4)
print("prompt tokens:", [len(p) for p in prompts])
print("expected:", meta[0]["expected"][:60])

config = build_engine_config(
    MODEL, max_num_sequences=4, max_num_batched_tokens=4 * 1300,
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(spec_method="ngram", num_speculative_tokens=5)
engine = SpecLLMEngine(config)
params = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)

# warm (and pin) the shared doc prefix
engine.pin_prefix(prompts[0])
engine.generate_token_ids(prompts[:1], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))

for mode in ("plain", "ngram"):
    engine.drafter = None if mode == "plain" else NgramDrafter(5, 2, 8)
    engine.spec_stats.update(rounds=0, drafted_tokens=0, accepted_tokens=0, fallback_rounds=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate_token_ids(prompts[:2], params)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    toks = out["metrics"]["output_tokens"]
    print(f"{mode}: wall={wall:.3f}s tokens={toks} tps={toks/wall:.1f}")
    if mode == "ngram":
        s = engine.spec_stats
        print("  spec:", s, "accept-rate:", s["accepted_tokens"] / max(1, s["drafted_tokens"]))
    print("  answer:", repr(tok.decode(out["token_ids"][0])[:110]))

engine.exit()
