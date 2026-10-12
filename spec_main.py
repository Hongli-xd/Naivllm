"""Speculative-decoding demo for NaivLLM.

Run (single GPU):
    PYTHONPATH=src python spec_main.py --model /path/to/Qwen3-0.6B \
        --method ngram --k 5 --temperature 0

Methods:
    ngram  prompt-lookup drafting (no draft model; excels when the output
           quotes the context — RAG/extractive workloads)
    model  small draft model (needs --draft-model)
    none   plain decoding, for A/B comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.config import build_engine_config
from myvllm.sampling_parameters import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--method", choices=("ngram", "model", "none"), default="ngram")
    parser.add_argument("--k", type=int, default=5, help="speculative tokens per round")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-length", type=int, default=1024)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    config = build_engine_config(
        args.model,
        max_num_sequences=8,
        max_num_batched_tokens=8192,
        max_model_length=args.max_model_length,
        gpu_memory_utilization=0.85,
        enforce_eager=args.enforce_eager,
    )
    config.update(
        spec_method=args.method,
        num_speculative_tokens=args.k,
        ngram_min_size=2,
        ngram_max_size=8,
    )
    if args.draft_model:
        config["draft_model"] = args.draft_model

    if args.method == "none":
        from myvllm.engine.llm_engine import LLMEngine
        engine = LLMEngine(config)
    else:
        from myvllm.spec_engine import SpecLLMEngine
        engine = SpecLLMEngine(config)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # extractive-style prompts: the answer is a span of the context, which is
    # exactly the regime speculative n-gram decoding targets
    prompts = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 "
        "and is 330 metres tall. Question: How tall is the Eiffel Tower? Answer:",
        "def add(a, b):\n    return a + b\n\n# add the numbers 3 and 4\nresult = add(",
        "The capital of Japan is Tokyo. The capital of France is Paris. "
        "Question: What is the capital of France? Answer:",
    ]
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_model_length=args.max_model_length,
    )

    with engine:
        output = engine.generate(formatted, params)

    for prompt, text in zip(prompts, output["text"]):
        print(f"\n--- {prompt}\n>>> {text.strip()}")
    print(f"\nmetrics: {output['metrics']}")


if __name__ == "__main__":
    main()
