"""Profile one speculative round: verify forward vs sampling vs bookkeeping."""

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
prompts, meta = build_prompts(tok, doc, facts, 2)

config = build_engine_config(
    MODEL, max_num_sequences=2, max_num_batched_tokens=2 * 1300,
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(spec_method="ngram", num_speculative_tokens=5)
engine = SpecLLMEngine(config)
engine.pin_prefix(prompts[0])
engine.generate_token_ids(prompts[:1], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))

runner = engine.model_runner
bm = engine.scheduler.block_manager

# instrument run_spec_verify phases
import myvllm.engine.model_runner as mr

timings = {"prepare": 0.0, "forward": 0.0, "n_rounds": 0}

orig_prepare = runner.prepare_spec_verify
orig_run = runner.run_spec_verify

def timed_verify(seqs):
    t0 = time.perf_counter()
    input_ids = runner.prepare_spec_verify(seqs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        logits = runner.model.compute_logits(hidden)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    from myvllm.utils import reset_context
    reset_context()
    timings["prepare"] += t1 - t0
    timings["forward"] += t2 - t1
    timings["n_rounds"] += 1
    return logits

runner.run_spec_verify = timed_verify

engine.drafter = NgramDrafter(5, 2, 8)
t0 = time.perf_counter()
out = engine.generate_token_ids(prompts, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True))
wall = time.perf_counter() - t0

n = max(1, timings["n_rounds"])
print(f"wall={wall:.3f}s rounds={timings['n_rounds']}")
print(f"per-round: prepare={1000*timings['prepare']/n:.1f}ms forward={1000*timings['forward']/n:.1f}ms")
print(f"other per-round: {1000*(wall - timings['prepare'] - timings['forward'])/n:.1f}ms")
engine.exit()
