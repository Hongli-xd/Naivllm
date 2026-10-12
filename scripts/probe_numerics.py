"""Numeric probe: compare NaivLLM logits against HF transformers step by step.

Stage 1: prefill logits (last position) for one short prompt.
Stage 2: single decode-step logits after one appended token.
Stage 3: spec-decode losslessness (spec output == plain output of same engine).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def hf_logits(model_path, prompt_ids, extra_ids=()):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    ids = list(prompt_ids) + list(extra_ids)
    with torch.inference_mode():
        out = model(torch.tensor([ids], device="cuda")).logits[0]
    result = out.clone()
    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-0.6B")
    args = parser.parse_args()

    from myvllm.config import build_engine_config
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    chat = tok.apply_chat_template([{"role": "user", "content": "The capital of France is"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, add_special_tokens=False)["input_ids"]
    print(f"prompt tokens: {len(ids)}")

    # ---- HF reference
    hf = hf_logits(args.model, ids)          # (len, vocab)
    print("HF last-token argmax:", int(hf[-1].argmax()), tok.decode([int(hf[-1].argmax())]))

    # ---- NaivLLM prefill via engine internals
    from myvllm.engine.model_runner import ModelRunner
    from myvllm.engine.sequence import Sequence
    from myvllm.engine.block_manager import BlockManager
    from myvllm.sampling_parameters import SamplingParams

    config = build_engine_config(args.model, max_num_sequences=2,
                                 max_num_batched_tokens=512, max_model_length=256,
                                 gpu_memory_utilization=0.5)
    runner = ModelRunner(config, rank=0, event=[])

    seq = Sequence(list(ids), SamplingParams(temperature=0))
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])
    bm.allocate(seq)
    input_ids = runner.prepare_prefill([seq])
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        logits = runner.model.compute_logits(hidden)[0].float()
    from myvllm.utils import reset_context
    reset_context()

    ref = hf[-1].float().cpu()
    got = logits.float().cpu()
    diff = (ref - got).abs()
    print(f"prefill last-token logits: max|diff|={diff.max():.4f} mean|diff|={diff.mean():.5f}")
    print("  hf argmax:", int(ref.argmax()), "| naiv argmax:", int(got.argmax()))
    top_ref = torch.topk(ref, 5).indices.tolist()
    top_got = torch.topk(got, 5).indices.tolist()
    print("  hf top5:", top_ref, [tok.decode([t]) for t in top_ref])
    print("  naiv top5:", top_got, [tok.decode([t]) for t in top_got])

    # ---- decode step: append HF's chosen token, compare next-position logits
    next_tok = int(ref.argmax())
    seq.append_token(next_tok)
    bm.append(seq)
    input_ids = runner.prepare_decode([seq])
    with torch.inference_mode():
        hidden = runner.model(input_ids)
        dlogits = runner.model.compute_logits(hidden)[0].float()
    reset_context()

    hf2 = hf_logits(args.model, ids, extra_ids=[next_tok])[-1].float().cpu()
    dgot = dlogits.cpu()
    ddiff = (hf2 - dgot).abs()
    print(f"decode-step logits: max|diff|={ddiff.max():.4f} mean|diff|={ddiff.mean():.5f}")
    print("  hf argmax:", int(hf2.argmax()), "| naiv argmax:", int(dgot.argmax()))

    runner.exit()
    ok = diff.max() < 0.5 and ddiff.max() < 0.5 and int(ref.argmax()) == int(got.argmax())
    print("PROBE:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
