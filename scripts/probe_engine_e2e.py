"""End-to-end engine dump: plain greedy decode token ids vs HF, one prompt."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-0.6B")
    parser.add_argument("--method", default="none", choices=("none", "ngram"))
    parser.add_argument("--tokens", type=int, default=12)
    args = parser.parse_args()

    from myvllm.config import build_engine_config
    from myvllm.sampling_parameters import SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = "The Great Wall of China stretches across northern China. Question: How long is it? Answer:"
    chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, add_special_tokens=False)["input_ids"]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    with torch.inference_mode():
        ref = model.generate(torch.tensor([ids], device="cuda"),
                             max_new_tokens=args.tokens, do_sample=False,
                             pad_token_id=ids[-1])[0, len(ids):].tolist()
    del model
    torch.cuda.empty_cache()

    config = build_engine_config(args.model, max_num_sequences=2,
                                 max_num_batched_tokens=1024, max_model_length=512,
                                 gpu_memory_utilization=0.5)
    params = SamplingParams(temperature=0, max_tokens=args.tokens, ignore_eos=True)

    if args.method == "none":
        from myvllm.engine.llm_engine import LLMEngine
        engine = LLMEngine(config)
        out = engine.generate_token_ids([ids], params)
        engine.exit()
        got = out["token_ids"][0]
    else:
        from myvllm.spec_engine import SpecLLMEngine
        config.update(spec_method="ngram", num_speculative_tokens=5)
        engine = SpecLLMEngine(config)
        out = engine.generate_token_ids([ids], params)
        engine.exit()
        got = out["token_ids"][0]
        print("spec stats:", out["metrics"]["spec_decoding"])

    print("hf  :", ref)
    print("ours:", got)
    print("hf  :", repr(tok.decode(ref)))
    print("ours:", repr(tok.decode(got)))
    match = [a == b for a, b in zip(got, ref)]
    print("pos-match:", match)


if __name__ == "__main__":
    main()
