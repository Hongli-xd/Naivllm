"""Probe: at a given decode position, compare logits from the decode-kernel
path vs the verify-kernel path on identical state. If their top-2 gap is
below cross-kernel bf16 noise, a greedy argmax flip is a benign tie-break."""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-4B")
    parser.add_argument("--seq-index", type=int, default=1)
    parser.add_argument("--position", type=int, default=27)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    doc, facts = build_document(tok, 600)
    prompts, _ = build_prompts(tok, doc, facts, 3)

    # eager runner: run_model returns raw logits on the decode path
    config = build_engine_config(
        args.model, max_num_sequences=3, max_num_batched_tokens=3 * 700,
        max_model_length=800, gpu_memory_utilization=0.5, enforce_eager=True)
    runner = ModelRunner(config, rank=0, event=[])
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])

    seqs = [Sequence(list(p), SamplingParams(temperature=0)) for p in prompts]
    for seq in seqs:
        bm.allocate(seq)
    inp = runner.prepare_prefill(seqs)
    with torch.inference_mode():
        hidden = runner.model(inp)
        first = runner.model.compute_logits(hidden)
    reset_context()
    # per-seq first tokens (prefill lm head returns last row per seq)
    firsts = first.argmax(-1).tolist()
    for seq, t in zip(seqs, firsts):
        seq.append_token(t)
        bm.append(seq)

    # advance to the requested position on the target seq only (all seqs
    # advance together, as in the engine)
    for _ in range(args.position - 1):
        inp = runner.prepare_decode(seqs)
        with torch.inference_mode():
            hidden = runner.model(inp)
            logits = runner.model.compute_logits(hidden)
        reset_context()
        toks = logits.argmax(-1).tolist()
        for seq, t in zip(seqs, toks):
            seq.append_token(t)
            bm.append(seq)

    # ---- state frozen: compare the two kernel paths at this position
    # (a) decode path
    inp = runner.prepare_decode(seqs)
    with torch.inference_mode():
        hidden = runner.model(inp)
        dec_logits = runner.model.compute_logits(hidden).float().cpu()
    reset_context()

    # (b) verify path (eager): dummy drafts, then roll back
    base_len = len(seqs[args.seq_index].token_ids)
    for seq in seqs:
        seq.draft_tokens = [1, 2, 3, 4, 5]
        for t in seq.draft_tokens:
            seq.append_token(t)
        bm.ensure_capacity(seq, len(seq.token_ids))
    ver_logits = runner.run_spec_verify(seqs).float().cpu()
    for seq in seqs:
        n0 = len(seq.token_ids) - len(seq.draft_tokens)
        del seq.token_ids[n0:]
        seq.num_tokens = n0
        seq.last_token = seq.token_ids[-1]
        seq.draft_tokens = []
        bm.trim_to(seq, n0)

    d = dec_logits[args.seq_index]
    k = 5
    v = ver_logits[args.seq_index * (k + 1)]  # rows are packed K+1 per seq
    dt = torch.topk(d, 3)
    vt = torch.topk(v, 3)
    print(f"position {args.position} of seq {args.seq_index}:")
    print("  decode path  top3:", [(int(i), round(float(x), 3)) for x, i in zip(dt.values, dt.indices)])
    print("  verify path  top3:", [(int(i), round(float(x), 3)) for x, i in zip(vt.values, vt.indices)])
    print(f"  decode top1-top2 gap: {float(dt.values[0]-dt.values[1]):.3f}")
    print(f"  verify top1-top2 gap: {float(vt.values[0]-vt.values[1]):.3f}")
    print(f"  paths agree on argmax: {int(dt.indices[0]) == int(vt.indices[0])}")
    runner.exit()


if __name__ == "__main__":
    main()
