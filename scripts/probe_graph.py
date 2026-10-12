"""Graph-vs-eager probe on identical decode inputs."""

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
    from myvllm.utils import get_context, reset_context
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    text = "The capital of France is"
    chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, add_special_tokens=False)["input_ids"]

    config = build_engine_config(args.model, max_num_sequences=2,
                                 max_num_batched_tokens=512, max_model_length=256,
                                 gpu_memory_utilization=0.5)
    runner = ModelRunner(config, rank=0, event=[])  # graphs captured here

    seq = Sequence(list(ids), SamplingParams(temperature=0))
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])
    bm.allocate(seq)

    # prefill (eager) and grab first token
    inp = runner.prepare_prefill([seq])
    with torch.inference_mode():
        hidden = runner.model(inp)
        logits = runner.model.compute_logits(hidden)[0].float()
    reset_context()
    first = int(logits.argmax())
    print("prefill argmax:", first)

    seq.append_token(first)
    bm.append(seq)

    # decode inputs prepared ONCE
    input_ids = runner.prepare_decode([seq])
    temperatures = runner.prepare_sample([seq])
    ctx = get_context()

    # (a) graph replay path
    out_graph = runner.run_model(input_ids, False, temperatures)
    reset_context()
    # restore context for the eager call (reset_context cleared it)
    from myvllm.utils import set_context
    set_context(is_prefill=False, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=ctx.slot_mapping, context_lens=ctx.context_lens,
                block_tables=ctx.block_tables)

    # (b) eager path with identical inputs
    runner.enforce_eager = True
    out_eager = runner.run_model(input_ids, False, temperatures)
    runner.enforce_eager = False
    reset_context()

    print("graph token:", out_graph.tolist())
    print("eager token:", out_eager.tolist())
    print("MATCH" if out_graph.tolist() == out_eager.tolist() else "MISMATCH")

    runner.exit()


if __name__ == "__main__":
    main()
