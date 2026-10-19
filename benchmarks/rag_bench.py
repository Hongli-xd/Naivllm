"""RAG-style grounded-QA latency benchmark: NaivLLM vs vLLM vs Ollama.

Workload (the vertical NaivLLM targets): one long fixed document prefix
shared by many short questions whose answers quote spans of the document,
decode-bound, small batch, single consumer-grade GPU.

    prefix = [system + big document]        <- pinned prefix cache
    suffix = [question + instruction]       <- varies per request
    output = short extractive answer        <- n-gram speculative decoding

Every backend runs in its own process on the same GPU with the same
pre-tokenized prompts and greedy decoding, so token streams are directly
comparable (speculative decoding is lossless). Reported per batch size:
request wall time, output tokens/s, and speculative acceptance where
applicable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROCESS_STARTED = time.perf_counter()


# --------------------------------------------------------------- workload gen
FACT_BANK = [
    ("Aurora Beacon Tower", "opened on March 14, 2021"),
    ("the Aurora Beacon Tower", "312 metres"),
    ("the northern wing", "renovated in 2019"),
    ("the beacon lantern", "visible from 40 kilometres away"),
    ("the tower lobby", "houses a collection of 17 bronze compasses"),
    ("Aurora Plaza", "paved with 12,400 granite tiles"),
    ("the observation deck", "closes at 22:30 every day"),
    ("the tower restaurant", "serves 380 guests at full capacity"),
    ("the eastern stairwell", "contains 1,911 steps"),
    ("the facade lighting", "uses 26,000 LED nodes"),
    ("the foundation slab", "poured over 44 days of continuous work"),
    ("the visiting hours", "from 09:00 to 21:00 in winter"),
    ("the tower's designer", "architect Mara Ellison"),
    ("the glass crown", "assembled from 880 panels"),
    ("the wind sway at the top", "limited to 35 centimetres"),
    ("the plaza fountain", "recycles 90,000 litres of water"),
    ("the tower radio mast", "installed on June 2, 2022"),
    ("the basement archive", "stores 60,000 blueprints"),
    ("the tower's nickname", "the Lighthouse of the North"),
    ("the annual visitor count", "about 2.3 million people"),
    ("the glass elevator", "travels at 6 metres per second"),
    ("the rooftop garden", "home to 140 plant species"),
    ("the emergency bunker", "supplied for 72 hours"),
    ("the tower clock", "accurate to within 2 seconds per year"),
    ("the north viewing hall", "features a 24-metre mural"),
    ("the tower cafe", "famous for its birch syrup latte"),
    ("the structural steel frame", "weighs 41,000 tonnes"),
    ("the heliport pad", "certified for helicopters up to 9 tonnes"),
    ("the summer light show", "runs every Friday at dusk"),
    ("the staff of the tower", "420 people in total"),
]

SYSTEM = (
    "You are a precise assistant. Answer using ONLY facts stated in the "
    "document. Reply with the exact words of the document, as a short quote."
)

# answers must be GENERATED (paraphrase/comparison), not copied — this
# collapses n-gram drafting and is where a model drafter can pay off
REASONING_SYSTEM = (
    "You are a careful analyst. Answer with your own analysis and wording. "
    "Do NOT copy sentences from the document."
)


def build_document(tokenizer, target_tokens: int) -> tuple[str, list[tuple[str, str]]]:
    """Deterministic pseudo-wiki document padded to ~target_tokens."""
    parts: list[str] = []
    facts: list[tuple[str, str]] = []
    idx = 0
    approx = 0
    while approx < target_tokens * 4:  # ~4 chars/token heuristic first pass
        chapter = idx % len(FACT_BANK)
        subject, fact = FACT_BANK[chapter]
        pad_idx = (idx * 7) % len(FACT_BANK)
        pad_subject, pad_fact = FACT_BANK[pad_idx]
        parts.append(
            f"{idx + 1}. {subject[0].upper() + subject[1:]}: the records state "
            f"that {subject} {fact}. By comparison, {pad_subject} {pad_fact}; "
            f"this has been verified twice, once in spring and once in autumn, "
            f"and the registry entry for {subject} remains unchanged since then."
        )
        facts.append((subject, fact))
        approx += len(parts[-1])
        idx += 1
    text = " ".join(parts)
    # trim to token budget
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        text = tokenizer.decode(ids[:target_tokens])
    return text, facts


def build_prompts(tokenizer, doc: str, facts, num_questions: int,
                  style: str = "extractive"):
    """style: extractive (answers quote the doc — n-gram friendly),
    reasoning (answers must be generated — model-drafter friendly),
    mixed (alternates the two; the hybrid drafter's home turf)."""
    prompts, meta = [], []
    for i in range(num_questions):
        subject, fact = facts[(i * 3) % len(facts)]
        other, _ = facts[(i * 5 + 1) % len(facts)]
        reasoning = style == "reasoning" or (style == "mixed" and i % 2 == 1)
        if reasoning:
            system, question = REASONING_SYSTEM, (
                f"Document: {doc}\n\nQuestion: Compare \"{subject}\" and "
                f"\"{other}\" based on the document, then summarize the "
                f"difference in your own words. Do not quote the document."
            )
        else:
            system, question = SYSTEM, (
                f"Document: {doc}\n\nQuestion: What do the records state "
                f"about {subject}? Answer:"
            )
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(tokenizer(chat, add_special_tokens=False)["input_ids"])
        meta.append({"subject": subject, "expected": fact, "style": "reasoning" if reasoning else "extractive"})
    return prompts, meta


# ------------------------------------------------------------------- backends
class NaivBackend:
    name = "naivllm"

    def __init__(self, args):
        import torch
        from myvllm.config import build_engine_config
        self.torch = torch
        config = build_engine_config(
            args.model,
            max_num_sequences=max(args.batches),
            max_num_batched_tokens=max(args.batches) * (args.doc_tokens + 64),
            max_model_length=args.doc_tokens + args.max_tokens + 96,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        config["verbose"] = False
        config.update(
            spec_method=args.naiv_method,
            num_speculative_tokens=args.k,
            ngram_min_size=2,
            ngram_max_size=8,
        )
        if args.naiv_method in ("model", "hybrid"):
            config["draft_model"] = args.draft_model
            config["draft_gpu_memory_utilization"] = 0.25
            config["target_gpu_memory_utilization"] = args.gpu_memory_utilization - 0.25
        if args.naiv_method != "none":
            from myvllm.spec_engine import SpecLLMEngine
            self.engine = SpecLLMEngine(config)
        else:
            from myvllm.engine.llm_engine import LLMEngine
            self.engine = LLMEngine(config)
        self.spec = args.naiv_method != "none"

    def warm_prefix(self, prompts):
        # first question primes (and pins) the shared doc prefix
        self.engine.pin_prefix(prompts[0])

    def generate(self, prompts, max_tokens: int) -> dict[str, Any]:
        from myvllm.sampling_parameters import SamplingParams
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        self.torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.engine.generate_token_ids(prompts, params)
        self.torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        metrics = out["metrics"]
        return {
            "wall_s": wall,
            "output_tokens": metrics["output_tokens"],
            "ttft_s": metrics["ttft_s"],
            "spec": metrics.get("spec_decoding"),
        }

    def close(self):
        self.engine.exit()


class VllmBackend:
    def __init__(self, args, spec):
        import vllm
        from transformers import AutoConfig, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.speculative_config = None
        if spec == "draft":
            # draft-model speculation: vLLM infers the method from "model"
            self.speculative_config = {
                "model": args.draft_model,
                "num_speculative_tokens": args.k,
            }
        elif spec:
            self.speculative_config = {
                "method": "ngram",
                "num_speculative_tokens": args.k,
                # field names as of vLLM 0.23 (formerly *_ngram_size)
                "prompt_lookup_max": 8,
                "prompt_lookup_min": 2,
            }
        kwargs = dict(
            model=args.model,
            tokenizer=args.model,
            dtype="bfloat16",
            max_model_len=args.doc_tokens + args.max_tokens + 96,
            max_num_seqs=max(args.batches),
            max_num_batched_tokens=max(args.batches) * (args.doc_tokens + 64),
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            disable_log_stats=True,
        )
        if self.speculative_config:
            kwargs["speculative_config"] = self.speculative_config
        self.engine = vllm.LLM(**kwargs)

    def warm_prefix(self, prompts):
        self.generate(prompts[:1], 1)

    def generate(self, prompts, max_tokens: int) -> dict[str, Any]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True)
        inputs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        t0 = time.perf_counter()
        outs = self.engine.generate(inputs, params, use_tqdm=False)
        wall = time.perf_counter() - t0
        return {
            "wall_s": wall,
            "output_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
            "ttft_s": None,
            "spec": None,
        }

    def close(self):
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown:
            shutdown()


class OllamaBackend:
    def __init__(self, args):
        import requests
        self.requests = requests
        # dedicated user-level ollama pinned to the same GPU as the other
        # backends (CUDA_VISIBLE_DEVICES=1), fp16 weights, bounded KV pool
        self.host = os.environ.get("OLLAMA_BENCH_HOST", "http://127.0.0.1:11435")
        self.model = args.ollama_model
        self.session = requests.Session()

    def warm_prefix(self, prompts):
        self.generate(prompts[:1], 1)

    def generate(self, prompts, max_tokens: int) -> dict[str, Any]:
        # ollama serves requests concurrently up to OLLAMA_NUM_PARALLEL
        import concurrent.futures
        def one(ids):
            t0 = time.perf_counter()
            resp = self.session.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    # raw mode: our prompts already carry the full chat
                    # template, so ollama must not wrap them again
                    "prompt": TOKENIZER.decode(ids),
                    "raw": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0,
                        "num_ctx": 4096,
                    },
                    "stream": False,
                },
                timeout=1800,
            )
            resp.raise_for_status()
            body = resp.json()
            return time.perf_counter() - t0, body.get("eval_count", 0)

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            results = list(pool.map(one, prompts))
        wall = time.perf_counter() - t0
        return {
            "wall_s": wall,
            "output_tokens": sum(r[1] for r in results),
            "ttft_s": None,
            "spec": None,
        }

    def close(self):
        pass


# ------------------------------------------------------------------ benchmark
def run_worker(args) -> dict[str, Any]:
    backend = None
    init_t0 = time.perf_counter()
    if args.worker_backend == "naivllm":
        backend = NaivBackend(args)
    elif args.worker_backend == "vllm":
        backend = VllmBackend(args, spec=False)
    elif args.worker_backend == "vllm-spec":
        backend = VllmBackend(args, spec=True)
    elif args.worker_backend == "vllm-draft":
        backend = VllmBackend(args, spec="draft")
    elif args.worker_backend == "ollama":
        backend = OllamaBackend(args)
    init_s = time.perf_counter() - init_t0

    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    doc, facts = build_document(TOKENIZER, args.doc_tokens)
    prompts, meta = build_prompts(TOKENIZER, doc, facts, args.questions, args.answer_style)

    # prime the shared document prefix so measured requests are "warm",
    # which is the realistic steady state of a knowledge-base assistant
    backend.warm_prefix(prompts)

    rows = []
    for batch in args.batches:
        samples = []
        for r in range(args.repeats):
            shard = [(r * batch + i) % args.questions for i in range(batch)]
            batch_prompts = [prompts[i] for i in shard]
            res = backend.generate(batch_prompts, args.max_tokens)
            res["batch_size"] = batch
            res["prompt_tokens"] = len(batch_prompts[0])
            samples.append(res)
        wall = statistics.median(s["wall_s"] for s in samples)
        toks = int(statistics.median(s["output_tokens"] for s in samples))
        rows.append({
            "batch_size": batch,
            "wall_s": wall,
            "output_tokens": toks,
            "output_tps": toks / wall,
            "spec": samples[-1]["spec"],
        })

    result = {
        "backend": args.worker_backend,
        "init_s": init_s,
        "doc_tokens": args.doc_tokens,
        "answer_style": args.answer_style,
        "max_tokens": args.max_tokens,
        "rows": rows,
    }
    backend.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/selom/models/Qwen3-0.6B")
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--ollama-model", default="qwen3:0.6b-fp16")
    parser.add_argument("--backends", default="naivllm,vllm-spec,vllm,ollama")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--doc-tokens", type=int, default=1200)
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answer-style", default="extractive", choices=("extractive", "reasoning", "mixed"))
    parser.add_argument("--naiv-method", default="ngram", choices=("ngram", "model", "hybrid", "none"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--output", default="benchmark-results/rag.json")
    parser.add_argument("--worker-backend", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.batches = [int(b) for b in args.backends and args.batches.split(",")]

    if args.worker_backend:
        result = run_worker(args)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("RAG_BENCH_RESULT=" + args.output)
        return

    results = []
    for backend in args.backends.split(","):
        out_path = Path(args.output).with_name(f"{Path(args.output).stem}-{backend}.json")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-backend", backend,
            "--output", str(out_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        print(f"\n===== backend: {backend} =====")
        proc = subprocess.run(cmd, env=env, text=True)
        if proc.returncode:
            print(f"!! {backend} worker failed ({proc.returncode}); continuing")
            continue
        results.append(json.loads(out_path.read_text(encoding="utf-8")))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"{'backend':<14}{'batch':>6}{'wall(s)':>10}{'out tps':>10}  spec")
    print("-" * 78)
    for result in results:
        for row in result["rows"]:
            spec = ""
            if row.get("spec"):
                s = row["spec"]
                acc = s.get("accepted_tokens", 0) / max(1, s.get("drafted_tokens", 1))
                spec = f"acc={acc:.2f} tok/round={s.get('accepted_tokens', 0)/max(1, s.get('rounds', 1)):.2f}"
            print(f"{result['backend']:<14}{row['batch_size']:>6}{row['wall_s']:>10.3f}"
                  f"{row['output_tps']:>10.1f}  {spec}")
    print("=" * 78)
    print(f"full results: {args.output}")


if __name__ == "__main__":
    main()
