"""Lossless check for the 0.6B -> 1.7B model-draft combination.

Plain 1.7B greedy decode is the reference; ngram / model / hybrid spec
decoding on the same engine must reproduce it token-for-token.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import torch

from rag_bench import build_document, build_prompts
from myvllm.config import build_engine_config
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_engine import SpecLLMEngine
from transformers import AutoTokenizer

DRAFT = "/home/selom/models/Qwen3-0.6B"


def run(config, prompts, max_tokens):
    engine = SpecLLMEngine(config)
    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    out = engine.generate_token_ids(prompts, params)
    stats = out["metrics"].get("spec_decoding")
    engine.exit()
    torch.cuda.empty_cache()
    return out["token_ids"], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-1.7B")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    doc, facts = build_document(tok, 600)
    prompts, _ = build_prompts(tok, doc, facts, 3)

    base = build_engine_config(
        args.model, max_num_sequences=3, max_num_batched_tokens=3 * 700,
        max_model_length=800, gpu_memory_utilization=0.5,
    )

    variants = {}
    # reference first (own process-level isolation not needed; sequential engines)
    cfg = dict(base); cfg.update(spec_method="none")
    variants["plain"] = run(cfg, prompts, args.max_tokens)

    cfg = dict(base); cfg.update(spec_method="ngram", num_speculative_tokens=args.k)
    variants["ngram"] = run(cfg, prompts, args.max_tokens)

    for method in ("model", "hybrid"):
        cfg = dict(base)
        cfg.update(
            spec_method=method, num_speculative_tokens=args.k,
            draft_model=DRAFT, draft_gpu_memory_utilization=0.2,
            target_gpu_memory_utilization=0.5,
        )
        variants[method] = run(cfg, prompts, args.max_tokens)

    ref = variants["plain"][0]
    print("ref:", tok.decode(ref[0])[:80])
    ok_all = True
    for name, (toks, stats) in variants.items():
        if name == "plain":
            continue
        same = toks == ref
        ok_all &= same
        print(f"{name:7s}: lossless={same}  stats={ {k_: v for k_, v in (stats or {}).items() if k_ != 'method'} }")
        if not same:
            for i, (a, b) in enumerate(zip(toks, ref)):
                if a != b:
                    print(f"  seq{i} first divergence:",
                          next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), -1))
    print("RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
