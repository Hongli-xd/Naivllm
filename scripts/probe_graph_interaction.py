"""Does replaying a decode graph break subsequent verify-graph replays?"""

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
prompts, _ = build_prompts(tok, doc, facts, 1)

config = build_engine_config(
    MODEL, max_num_sequences=1, max_num_batched_tokens=1300,
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(capture_verify=True, num_spec_tokens=5)
runner = ModelRunner(config, rank=0, event=[])
bm = BlockManager(config["max_cached_blocks"], config["block_size"])

seq = Sequence(list(prompts[0]), SamplingParams(temperature=0))
bm.allocate(seq)

# prefill + first token
inp = runner.prepare_prefill([seq])
with torch.inference_mode():
    h = runner.model(inp)
    first = int(runner.model.compute_logits(h).argmax())
reset_context()
seq.append_token(first)
bm.append(seq)
base_len = len(seq.token_ids)


def verify_once(label):
    global base_len
    assert len(seq.token_ids) == base_len
    seq.draft_tokens = [101, 102, 103, 104, 105]
    for t in seq.draft_tokens:
        seq.append_token(t)
    bm.ensure_capacity(seq, len(seq.token_ids))
    logits = runner.run_spec_verify([seq]).float().cpu()
    hv = runner.verify_vars["hidden_tap"].float()
    print(f"{label}: argmax rows = {logits.argmax(-1).tolist()} "
          f"hidden_norm={float(hv.norm()):.2f} logits_absmax={float(logits.abs().max()):.3f}")
    # roll the speculative tail back
    del seq.token_ids[base_len:]
    seq.num_tokens = base_len
    seq.last_token = seq.token_ids[-1]
    seq.draft_tokens = []
    bm.trim_to(seq, base_len)


verify_once("verify #1 (fresh, no decode replay yet)")

tokens = runner.run([seq], False)   # decode-graph replay
print("decode step token:", tokens)
seq.append_token(tokens[0])
bm.append(seq)
base_len = len(seq.token_ids)

verify_once("verify #2 (after one decode-graph replay)")

tokens = runner.run([seq], False)
print("decode step token:", tokens)
seq.append_token(tokens[0])
bm.append(seq)
base_len = len(seq.token_ids)

verify_once("verify #3 (after two decode-graph replays)")

runner.exit()
