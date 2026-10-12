"""Per-round in-engine differential: graph verify vs eager recompute."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import torch

from rag_bench import build_document, build_prompts
from myvllm.config import build_engine_config
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_engine import SpecLLMEngine
from myvllm.spec_engine.drafters import NgramDrafter
from myvllm.utils import reset_context
from transformers import AutoTokenizer

MODEL = "/home/selom/models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
doc, facts = build_document(tok, 1200)
prompts, meta = build_prompts(tok, doc, facts, 1)

config = build_engine_config(
    MODEL, max_num_sequences=1, max_num_batched_tokens=1300,
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(spec_method="ngram", num_speculative_tokens=5)

from myvllm.engine.llm_engine import LLMEngine
ref_engine = LLMEngine(dict(config))
ref = ref_engine.generate_token_ids(
    prompts, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True))["token_ids"][0]
ref_engine.exit()
torch.cuda.empty_cache()

engine = SpecLLMEngine(config)
engine.pin_prefix(prompts[0])
engine.generate_token_ids(prompts[:1], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))
engine.drafter = NgramDrafter(5, 2, 8)

runner = engine.model_runner
orig_verify = runner.run_spec_verify
round_no = [0]

def dual_verify(seqs):
    graph_logits = orig_verify(seqs)
    k = runner.verify_vars["k"]
    stride = k + 1
    # eager recompute on the identical committed state
    input_ids = runner.prepare_spec_verify(seqs)
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        eager_logits = runner.model.compute_logits(hidden)
    reset_context()
    g = graph_logits.float().cpu()
    e = eager_logits.float().cpu()
    gam = g.argmax(-1).tolist()
    eam = e.argmax(-1).tolist()
    row = 0
    info = []
    for seq in seqs:
        k_i = len(seq.draft_tokens)
        info.append((gam[row:row + k_i + 1], eam[row:row + k_i + 1]))
        row += stride
    round_no[0] += 1
    n_match = sum(a == b for a, b in zip(gam, eam))
    print(f"round {round_no[0]}: rows_match {n_match}/{len(gam)}")
    if n_match < len(gam):
        for i, (ga, ea) in enumerate(info):
            if ga != ea:
                print(f"  seq{i} drafts={seqs[i].draft_tokens} graph={ga} eager={ea}")
                print(f"  seq{i} n={len(seqs[i].token_ids)-len(seqs[i].draft_tokens)} "
                      f"table={seqs[i].block_table[-3:]} ctx_len={len(seqs[i])}")
    return graph_logits

runner.run_spec_verify = dual_verify

out = engine.generate_token_ids(prompts, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True))
got = out["token_ids"][0]
match = [a == b for a, b in zip(got, ref)]
print("first divergence vs plain:", match.index(False) if False in match else -1)
print("ref:", ref[:16])
print("got:", got[:16])
engine.exit()
