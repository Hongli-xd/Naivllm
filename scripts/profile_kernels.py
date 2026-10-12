"""Torch-profiler view of one verify forward: top CUDA kernels (manual seqs)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import torch

from rag_bench import build_document, build_prompts
from myvllm.config import build_engine_config
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import Sequence
from myvllm.engine.block_manager import BlockManager
from myvllm.sampling_parameters import SamplingParams
from myvllm.utils import reset_context
from transformers import AutoTokenizer

MODEL = "/home/selom/models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
doc, facts = build_document(tok, 1200)
prompts, meta = build_prompts(tok, doc, facts, 2)

config = build_engine_config(
    MODEL, max_num_sequences=2, max_num_batched_tokens=2 * 1300,
    max_model_length=1400, gpu_memory_utilization=0.6, enforce_eager=True)
runner = ModelRunner(config, rank=0, event=[])
bm = BlockManager(config["max_cached_blocks"], config["block_size"])

seqs = [Sequence(list(p), SamplingParams(temperature=0)) for p in prompts]
for seq in seqs:
    bm.allocate(seq)

# prefill both
inp = runner.prepare_prefill(seqs)
with torch.inference_mode():
    h = runner.model(inp)
    first = runner.model.compute_logits(h)
reset_context()
for seq, t in zip(seqs, first.argmax(-1).tolist()):
    seq.append_token(t)
    bm.append(seq)

# append 5 draft tokens each
for seq in seqs:
    seq.draft_tokens = [100 + i for i in range(5)]
    for t in seq.draft_tokens:
        seq.append_token(t)
    bm.ensure_capacity(seq, len(seq))

input_ids = runner.prepare_spec_verify(seqs)
with torch.inference_mode():
    h = runner.model(input_ids)
    lg = runner.model.compute_logits(h)
reset_context()
torch.cuda.synchronize()

input_ids = runner.prepare_spec_verify(seqs)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.inference_mode():
        h = runner.model(input_ids)
        lg = runner.model.compute_logits(h)
    torch.cuda.synchronize()
reset_context()

events = prof.key_averages()
rows = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:12]
total = sum(e.self_device_time_total for e in events)
print(f"total CUDA time: {total/1000:.2f} ms")
for e in rows:
    print(f"{e.self_device_time_total/1000:8.2f} ms  {e.count:4d}x  {e.key[:90]}")

# also time the plain decode kernel path for comparison
inp = runner.prepare_decode(seqs)
with torch.inference_mode():
    h = runner.model(inp)
reset_context()
torch.cuda.synchronize()
import time
t0 = time.perf_counter()
for _ in range(20):
    inp = runner.prepare_decode(seqs)
    with torch.inference_mode():
        h = runner.model(inp)
    reset_context()
torch.cuda.synchronize()
print(f"plain decode step (eager, incl. python): {(time.perf_counter()-t0)/20*1000:.2f} ms")

runner.exit()
