"""Batched-execution probe: engine prefill/decode logits for a batch of 3
sequences vs per-prompt HF reference. Isolates batching bugs from numerics."""

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
    ]
    prompts = [
        tok(tok.apply_chat_template([{"role": "user", "content": t}],
             tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"]
        for t in texts
    ]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    refs = []
    with torch.inference_mode():
        for ids in prompts:
            refs.append(model(torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu())
    del model
    torch.cuda.empty_cache()

    config = build_engine_config(args.model, max_num_sequences=8,
                                 max_num_batched_tokens=512, max_model_length=256,
                                 gpu_memory_utilization=0.5)
    runner = ModelRunner(config, rank=0, event=[])
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])

    seqs = [Sequence(list(ids), SamplingParams(temperature=0)) for ids in prompts]
    for seq in seqs:
        bm.allocate(seq)

    # batched prefill: lm_head returns the last row per sequence
    input_ids = runner.prepare_prefill(seqs)
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        logits = runner.model.compute_logits(hidden).float().cpu()  # (3, vocab)
    reset_context()

    print("== batched prefill (last row per seq) ==")
    for i, ref in enumerate(refs):
        diff = (ref - logits[i]).abs()
        same = int(ref.argmax()) == int(logits[i].argmax())
        print(f" seq{i}: argmax hf={int(ref.argmax())} naiv={int(logits[i].argmax())} "
              f"match={same} max|diff|={diff.max():.3f}")

    # batched decode step: append hf argmax tokens, one decode forward
    print("== batched decode step ==")
    nexts = [int(ref.argmax()) for ref in refs]
    for seq, t in zip(seqs, nexts):
        seq.append_token(t)
        bm.append(seq)
    input_ids = runner.prepare_decode(seqs)
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        dlogits = runner.model.compute_logits(hidden).float().cpu()  # (3, vocab)
    reset_context()

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    ok = True
    for i, (ids, t) in enumerate(zip(prompts, nexts)):
        with torch.inference_mode():
            ref2 = model(torch.tensor([ids + [t]], device="cuda")).logits[0, -1].float().cpu()
        diff = (ref2 - dlogits[i]).abs()
        same = int(ref2.argmax()) == int(dlogits[i].argmax())
        ok &= same and diff.max() < 1.0
        print(f" seq{i}: argmax hf={int(ref2.argmax())} naiv={int(dlogits[i].argmax())} "
              f"match={same} max|diff|={diff.max():.3f}")

    runner.exit()
    print("BATCH PROBE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
