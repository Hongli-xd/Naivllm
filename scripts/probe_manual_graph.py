"""Manual CUDA-graph capture probe with an in-graph hidden-state tap."""

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
    from myvllm.utils import set_context, reset_context
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    chat = tok.apply_chat_template([{"role": "user", "content": "The capital of France is"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(chat, add_special_tokens=False)["input_ids"]

    # eager runner (no engine graphs), we capture our own
    config = build_engine_config(args.model, max_num_sequences=4,
                                 max_num_batched_tokens=512, max_model_length=256,
                                 gpu_memory_utilization=0.5, enforce_eager=True)
    runner = ModelRunner(config, rank=0, event=[])

    seq = Sequence(list(ids), SamplingParams(temperature=0))
    bm = BlockManager(config["max_cached_blocks"], config["block_size"])
    bm.allocate(seq)

    # prefill eagerly to fill KV cache
    inp = runner.prepare_prefill([seq])
    with torch.inference_mode():
        hidden = runner.model(inp)
        first_logits = runner.model.compute_logits(hidden)[0].float()
    reset_context()
    first = int(first_logits.argmax())
    print("prefill argmax:", first)
    seq.append_token(first)
    bm.append(seq)

    # ---- static tensors for manual capture
    dev = "cuda:0"
    bs = 1
    s_input = torch.zeros(4, dtype=torch.long, device=dev)
    s_slot = torch.zeros(4, dtype=torch.long, device=dev)
    s_ctx = torch.zeros(4, dtype=torch.long, device=dev)
    s_bt = torch.zeros(4, 16, dtype=torch.int32, device=dev)
    s_temp = torch.ones(4, dtype=torch.float32, device=dev)
    s_out = torch.zeros(4, dtype=torch.long, device=dev)
    s_hidden_tap = torch.zeros(4, config["hidden_size"], dtype=torch.bfloat16, device=dev)
    s_logits_tap = torch.zeros(4, config["vocab_size"], dtype=torch.bfloat16, device=dev)

    def step():
        set_context(is_prefill=False, slot_mapping=s_slot[:bs], context_lens=s_ctx[:bs],
                    block_tables=s_bt[:bs])
        h = runner.model(s_input[:bs])
        s_hidden_tap[:bs] = h
        lg = runner.model.compute_logits(h)
        s_logits_tap[:bs] = lg
        s_out[:bs] = runner.sampler(lg, s_temp[:bs])

    # fill static tensors with REAL decode inputs
    real_in = runner.prepare_decode([seq])
    from myvllm.utils import get_context
    ctx = get_context()
    s_input[:bs] = real_in[:bs]
    s_slot[:bs] = ctx.slot_mapping[:bs]
    s_ctx[:bs] = ctx.context_lens[:bs]
    s_bt[:bs, :ctx.block_tables.size(1)] = ctx.block_tables
    reset_context()

    # eager reference with same static inputs
    step(); reset_context()
    print("eager tap hidden norm:", float(s_hidden_tap[:bs].float().norm()))
    print("eager out:", s_out[:bs].tolist())

    # capture
    g = torch.cuda.CUDAGraph()
    step()  # warmup outside capture
    reset_context()
    with torch.cuda.graph(g):
        step()
    reset_context()
    torch.cuda.synchronize()

    # zero taps, set inputs again, replay
    s_hidden_tap.zero_(); s_logits_tap.zero_(); s_out.zero_()
    s_input[:bs] = real_in[:bs]
    g.replay()
    torch.cuda.synchronize()
    print("after replay: hidden norm:", float(s_hidden_tap[:bs].float().norm()),
          "logits max:", float(s_logits_tap[:bs].float().abs().max()),
          "out:", s_out[:bs].tolist())

    runner.exit()


if __name__ == "__main__":
    main()
