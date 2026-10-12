"""Per-round differential: run the spec engine with graph verify, and after
each round re-run an eager verify on the same committed state to compare what
the NEXT round would see. Logs the first divergent round."""

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
from transformers import AutoTokenizer

MODEL = "/home/selom/models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
doc, facts = build_document(tok, 1200)
prompts, meta = build_prompts(tok, doc, facts, 1)

config = build_engine_config(
    MODEL, max_num_sequences=1, max_num_batched_tokens=1300,
    max_model_length=1400, gpu_memory_utilization=0.6)
config.update(spec_method="ngram", num_speculative_tokens=5)

# reference: plain decode engine tokens (run and tear down first so both
# engines never share the GPU)
from myvllm.engine.llm_engine import LLMEngine
ref_engine = LLMEngine(dict(config))
ref_params = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
ref_out = ref_engine.generate_token_ids(prompts, ref_params)
ref_tokens = ref_out["token_ids"][0]
ref_engine.exit()
torch.cuda.empty_cache()
print("reference (plain):", ref_tokens[:20])

engine = SpecLLMEngine(config)
engine.pin_prefix(prompts[0])
engine.generate_token_ids(prompts[:1], SamplingParams(temperature=0, max_tokens=4, ignore_eos=True))

runner = engine.model_runner
engine.drafter = NgramDrafter(5, 2, 8)

out = engine.generate_token_ids(prompts, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True))
got = out["token_ids"][0]
print("spec (graph)    :", got[:20])
match = [a == b for a, b in zip(got, ref_tokens)]
first_bad = match.index(False) if False in match else -1
print("first divergence at output position:", first_bad)
print("stats:", engine.spec_stats)
engine.exit()
