"""Isolation probe: per-sequence (batch=1) prefill+decode vs HF, then batched.

If batch=1 matches for every sequence but batch=3 fails, the bug is in
batched decode plumbing; if a lone sequence also fails, its KV cache content
(prefill store) is wrong for that length.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-0.6B")
    args = parser.parse_args()

    from myvllm.config import build_engine_config
    from myvllm.engine.model_runner import ModelRunner
    from myvllm.engine.sequence import Sequence
    from myvllm.engine.block_manager import BlockManager
    from myvllm.sampling_parameters import SamplingParams
    from myvllm.utils import reset_context
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [
        "The capital of France is",
        "Write a haiku about rain:",
        "中国的首都是",
        "A slightly longer prompt to shift block boundaries in the paged KV cache substantially:",
    ]
    prompts = [
        tok(tok.apply_chat_template([{"role": "user", "content": t}],
             tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"]
        for t in texts
    ]
    print("lengths:", [len(p) for p in prompts])

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    def hf_next(ids, extra):
        with torch.inference_mode():
            return model(torch.tensor([list(ids) + list(extra)], device="cuda")).logits[0, -1].float().cpu()

    config = build_engine_config(args.model, max_num_sequences=8,
                                 max_num_batched_tokens=2048, max_model_length=512,
                                 gpu_memory_utilization=0.5)
    runner = ModelRunner(config, rank=0, event=[])
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])

    # ---------- batch of 1 per sequence
    print("== single-sequence rounds ==")
    for i, ids in enumerate(prompts):
        seq = Sequence(list(ids), SamplingParams(temperature=0))
        bm.allocate(seq)
        input_ids = runner.prepare_prefill([seq])
        with torch.inference_mode():
            hidden = runner.model(input_ids)
            logits = runner.model.compute_logits(hidden)[0].float().cpu()
        reset_context()
        first = int(logits.argmax())
        ref1 = hf_next(ids, [])
        seq.append_token(first)
        bm.append(seq)
        input_ids = runner.prepare_decode([seq])
        with torch.inference_mode():
            hidden = runner.model(input_ids)
            dlogits = runner.model.compute_logits(hidden)[0].float().cpu()
        reset_context()
        ref2 = hf_next(ids, [first])
        same = int(ref2.argmax()) == int(dlogits.argmax())
        diff = (ref2 - dlogits).abs().max().item()
        print(f" seq{i} (len {len(ids)}): prefill_argmax={first} hf={int(ref1.argmax())} | "
              f"decode match={same} hf={int(ref2.argmax())} naiv={int(dlogits.argmax())} max|diff|={diff:.3f}")
        bm.deallocate(seq)

    # ---------- batched all four
    print("== batched round ==")
    seqs = [Sequence(list(ids), SamplingParams(temperature=0)) for ids in prompts]
    for seq in seqs:
        bm.allocate(seq)
    input_ids = runner.prepare_prefill(seqs)
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        logits = runner.model.compute_logits(hidden).float().cpu()
    reset_context()
    firsts = [int(r.argmax()) for r in logits]
    for seq, t in zip(seqs, firsts):
        seq.append_token(t)
        bm.append(seq)
    input_ids = runner.prepare_decode(seqs)
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        dlogits = runner.model.compute_logits(hidden).float().cpu()
    reset_context()
    for i, ids in enumerate(prompts):
        ref2 = hf_next(ids, [firsts[i]])
        same = int(ref2.argmax()) == int(dlogits[i].argmax())
        print(f" seq{i} (len {len(ids)}): decode match={same} hf={int(ref2.argmax())} "
              f"naiv={int(dlogits[i].argmax())} max|diff|={(ref2 - dlogits[i]).abs().max():.3f}")

    runner.exit()


if __name__ == "__main__":
    main()
