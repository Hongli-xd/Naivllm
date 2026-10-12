"""Compare graph-replayed verify logits vs eager verify logits (long ctx)."""

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
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(capture_verify=True, num_spec_tokens=5)
runner = ModelRunner(config, rank=0, event=[])  # captures decode + verify graphs
bm = BlockManager(config["max_cached_blocks"], config["block_size"])

seqs = [Sequence(list(p), SamplingParams(temperature=0)) for p in prompts]
for seq in seqs:
    bm.allocate(seq)

# prefill
inp = runner.prepare_prefill(seqs)
with torch.inference_mode():
    h = runner.model(inp)
    first = runner.model.compute_logits(h).argmax(-1).tolist()
reset_context()
for seq, t in zip(seqs, first):
    seq.append_token(t)
    bm.append(seq)

# attach drafts
for seq in seqs:
    seq.draft_tokens = [101, 102, 103, 104, 105]
    for t in seq.draft_tokens:
        seq.append_token(t)
    bm.ensure_capacity(seq, len(seq))

# eager verify (bypass graph)
input_ids = runner.prepare_spec_verify(seqs)
with torch.inference_mode():
    h = runner.model(input_ids)
    eager_logits = runner.model.compute_logits(h).float().cpu()
reset_context()

# graph verify
graph_logits = runner.run_spec_verify(seqs).float().cpu()

diff = (eager_logits - graph_logits).abs()
print("eager vs graph verify logits: max|diff| =", diff.max().item(),
      "mean =", diff.mean().item())
for i in range(min(4, eager_logits.size(0))):
    print(f" row {i}: eager argmax {int(eager_logits[i].argmax())} "
          f"graph argmax {int(graph_logits[i].argmax())} "
          f"rowdiff {diff[i].max():.3f}")
runner.exit()
