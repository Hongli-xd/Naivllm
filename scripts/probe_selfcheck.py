"""Self-consistency probe.

For one sequence: compute the next-token logits three ways —
  A) full-context prefill (flash varlen) on prompt + token t
  B) decode step after cached prefill of prompt
  C) HF reference
If A~C but B deviates beyond kernel noise, the decode path (paged KV read)
has a real defect; if B~A and the argmax flip only happens on a flat top-2,
it is benign bf16 noise.
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
    text = "Write a haiku about rain:"
    ids = tok(tok.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True),
              add_special_tokens=False)["input_ids"]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    with torch.inference_mode():
        c = model(torch.tensor([ids], device="cuda")).logits[0, -1].float().cpu()
    first = int(c.argmax())
    with torch.inference_mode():
        c2 = model(torch.tensor([ids + [first]], device="cuda")).logits[0, -1].float().cpu()
    del model
    torch.cuda.empty_cache()

    config = build_engine_config(args.model, max_num_sequences=4,
                                 max_num_batched_tokens=512, max_model_length=256,
                                 gpu_memory_utilization=0.5)
    runner = ModelRunner(config, rank=0, event=[])
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])

    # A) full prefill of prompt + t
    seq_a = Sequence(list(ids) + [first], SamplingParams(temperature=0))
    bm.allocate(seq_a)
    inp = runner.prepare_prefill([seq_a])
    with torch.inference_mode():
        hidden = runner.model(inp)
        logits_a = runner.model.compute_logits(hidden)[0].float().cpu()
    reset_context()

    # B) cached decode step
    seq_b = Sequence(list(ids), SamplingParams(temperature=0))
    bm.allocate(seq_b)
    inp = runner.prepare_prefill([seq_b])
    with torch.inference_mode():
        hidden = runner.model(inp)
        logits_prefill = runner.model.compute_logits(hidden)[0].float().cpu()
    reset_context()
    seq_b.append_token(first)
    bm.append(seq_b)
    inp = runner.prepare_decode([seq_b])
    with torch.inference_mode():
        hidden = runner.model(inp)
        logits_b = runner.model.compute_logits(hidden)[0].float().cpu()
    reset_context()

    print(f"prefill argmax check: engine={int(logits_prefill.argmax())} hf={first}")
    print(f"A prefill-full   argmax={int(logits_a.argmax())}")
    print(f"B cached-decode  argmax={int(logits_b.argmax())}")
    print(f"C hf             argmax={int(c2.argmax())}")
    print(f"|A-B| max={(logits_a - logits_b).abs().max():.4f} mean={(logits_a - logits_b).abs().mean():.5f}")
    print(f"|A-C| max={(logits_a - c2).abs().max():.4f}")
    print(f"|B-C| max={(logits_b - c2).abs().max():.4f}")
    top2 = torch.topk(c2, 4)
    print("hf top4:", [(int(i), round(float(v), 3)) for v, i in zip(top2.values, top2.indices)])
    top2b = torch.topk(logits_b, 4)
    print("B   top4:", [(int(i), round(float(v), 3)) for v, i in zip(top2b.values, top2b.indices)])

    runner.exit()


if __name__ == "__main__":
    main()
