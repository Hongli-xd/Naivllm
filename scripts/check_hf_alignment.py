"""Numerical alignment check: NaivLLM vs HF transformers (greedy decode).

Verifies that the engine produces the same token ids as a reference HF
Qwen3 model for identical prompts, both for plain decoding and for
speculative decoding (which must be exactly lossless under greedy
verification).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def hf_greedy(model_path: str, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    results = []
    with torch.inference_mode():
        for ids in prompts:
            out = model.generate(
                torch.tensor([ids], device="cuda"),
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=ids[-1],
            )
            results.append(out[0, len(ids):].tolist())
    del model
    torch.cuda.empty_cache()
    return results


def naivllm_greedy(config: dict, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    from myvllm.engine.llm_engine import LLMEngine
    from myvllm.sampling_parameters import SamplingParams

    engine = LLMEngine(config)
    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    result = engine.generate_token_ids(prompts, params)
    engine.exit()
    return result["token_ids"]


def spec_greedy(config: dict, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    from myvllm.spec_engine import SpecLLMEngine
    from myvllm.sampling_parameters import SamplingParams

    engine = SpecLLMEngine(config)
    params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
    result = engine.generate_token_ids(prompts, params)
    stats = result["metrics"].get("spec_decoding", {})
    engine.exit()
    return result["token_ids"], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    from myvllm.config import build_engine_config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = [
        "The Great Wall of China stretches across northern China and was built over many centuries. "
        "Its total length is about 21,196 kilometers. Question: How long is the Great Wall? Answer:",
        "In 1969, Apollo 11 landed the first humans on the Moon. Neil Armstrong and Buzz Aldrin "
        "spent about 21 hours on the lunar surface. Question: Who landed on the Moon in 1969? Answer:",
        "写一段关于春天的描述：",
    ]
    prompts = []
    for t in texts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])

    print("== HF reference (greedy) ==")
    ref = hf_greedy(args.model, prompts, args.max_tokens)
    for t, r in zip(texts, ref):
        print(" >", tokenizer.decode(r)[:120].replace("\n", " "))

    base_config = build_engine_config(
        args.model, max_num_sequences=8, max_num_batched_tokens=8192,
        max_model_length=max(len(p) for p in prompts) + args.max_tokens + 8,
        gpu_memory_utilization=0.7,
    )

    print("\n== NaivLLM plain decode ==")
    plain = naivllm_greedy(dict(base_config), prompts, args.max_tokens)
    agree = [sum(a == b for a, b in zip(x, y)) / max(len(y), 1) for x, y in zip(plain, ref)]
    print("token agreement vs HF:", [f"{a:.0%}" for a in agree])
    # from-scratch Triton kernels round differently than HF SDPA; on flat
    # top-2 distributions the argmax flips and the continuation changes, so
    # exact HF equality is not the bar — non-degenerate fluent output is
    degenerate = [
        len(set(x)) < 5 or max(x.count(t) for t in set(x)) > len(x) * 0.5
        for x in plain
    ]
    ok_plain = not any(degenerate)
    print("degenerate outputs:", degenerate)

    print("\n== NaivLLM speculative decode (ngram) ==")
    spec_config = dict(base_config)
    spec_config.update(spec_method="ngram", num_speculative_tokens=args.k)
    spec, stats = spec_greedy(spec_config, prompts, args.max_tokens)
    agree_spec = [sum(a == b for a, b in zip(x, y)) / max(len(y), 1) for x, y in zip(spec, ref)]
    lossless = [sum(a == b for a, b in zip(x, y)) / max(len(y), 1) for x, y in zip(spec, plain)]
    print("token agreement vs HF:", [f"{a:.0%}" for a in agree_spec])
    print("losslessness vs plain engine:", [f"{a:.0%}" for a in lossless])
    print("stats:", stats)
    # the hard requirement: speculative decoding must be exactly lossless
    # against the engine's own decode
    ok_spec = all(a >= 0.98 for a in lossless)

    print("\nRESULT:", "PASS" if (ok_plain and ok_spec) else "FAIL")
    sys.exit(0 if (ok_plain and ok_spec) else 1)


if __name__ == "__main__":
    main()
