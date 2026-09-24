"""Fair, process-isolated benchmarks for NaivLLM's narrow target regimes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - benchmark workers run on Linux
    resource = None


PROCESS_STARTED = time.perf_counter()
RESULT_MARKER = "NAIVLLM_BENCHMARK_RESULT="


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def make_prompts(vocab_size: int, batch: int, length: int, nonce: int) -> list[list[int]]:
    usable_vocab = max(vocab_size - 2048, 1)
    return [
        [1024 + ((nonce * 104729 + seq * 8191 + pos * 131) % usable_vocab) for pos in range(length)]
        for seq in range(batch)
    ]


class NaivBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        from myvllm import LLMEngine, SamplingParams, build_engine_config

        self.torch = torch
        self.SamplingParams = SamplingParams
        config = build_engine_config(
            args.model,
            max_num_sequences=max(args.batches),
            max_num_batched_tokens=max(args.batches) * max(args.contexts),
            max_model_length=max(args.contexts) + args.output_tokens + 8,
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        config["verbose"] = False
        self.vocab_size = config["vocab_size"]
        self.engine = LLMEngine(config)

    def generate(self, prompts: list[list[int]], output_tokens: int) -> dict[str, Any]:
        params = self.SamplingParams(
            temperature=1.0,
            max_tokens=output_tokens,
            ignore_eos=True,
            max_model_length=max(map(len, prompts)) + output_tokens,
        )
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        output = self.engine.generate_token_ids(prompts, params)
        self.torch.cuda.synchronize()
        total_s = time.perf_counter() - started
        metrics = output["metrics"]
        return {
            "total_s": total_s,
            "ttft_p50_s": metrics["ttft_s"],
            "ttft_p99_s": metrics["ttft_s"],
            "output_tokens": metrics["output_tokens"],
            "cache": metrics["prefix_cache"],
        }

    def pin_prefix(self, token_ids: list[int]) -> int:
        return self.engine.pin_prefix(token_ids)

    def close(self) -> None:
        self.engine.exit()


class VllmBackend:
    def __init__(self, args: argparse.Namespace):
        import torch
        import vllm
        from transformers import AutoConfig
        from vllm import LLM, SamplingParams

        self.torch = torch
        self.SamplingParams = SamplingParams
        self.vllm_version = vllm.__version__
        config = AutoConfig.from_pretrained(args.model)
        self.vocab_size = config.vocab_size
        self.engine = LLM(
            model=args.model,
            tokenizer=args.model,
            dtype="bfloat16",
            max_model_len=max(args.contexts) + args.output_tokens + 8,
            max_num_seqs=max(args.batches),
            max_num_batched_tokens=max(args.batches) * max(args.contexts),
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            enforce_eager=False,
            trust_remote_code=False,
            disable_log_stats=True,
        )

    def generate(self, prompts: list[list[int]], output_tokens: int) -> dict[str, Any]:
        from vllm.inputs import TokensPrompt

        params = self.SamplingParams(
            temperature=1.0,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        inputs = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = self.engine.generate(inputs, params, use_tqdm=False)
        self.torch.cuda.synchronize()
        total_s = time.perf_counter() - started
        ttfts = []
        for output in outputs:
            metrics = getattr(output, "metrics", None)
            first = getattr(metrics, "first_token_time", None)
            arrival = getattr(metrics, "arrival_time", None)
            if first is not None and arrival is not None:
                ttfts.append(first - arrival)
        return {
            "total_s": total_s,
            # Offline ``LLM.generate`` does not expose per-request TTFT in
            # vLLM 0.23. Use the measured request wall time as a conservative
            # fallback rather than emitting a misleading zero.
            "ttft_p50_s": statistics.median(ttfts) if ttfts else total_s,
            "ttft_p99_s": percentile(ttfts, 0.99) if ttfts else total_s,
            "output_tokens": sum(len(item.outputs[0].token_ids) for item in outputs),
            "cache": None,
        }

    def pin_prefix(self, token_ids: list[int]) -> int:
        return 0

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if shutdown is not None:
            shutdown()


def gpu_used_mb(torch_module: Any) -> float:
    free, total = torch_module.cuda.mem_get_info()
    return (total - free) / 2**20


def benchmark_curve(backend: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    nonce = 10
    for context in args.contexts:
        for batch in args.batches:
            warmup = make_prompts(backend.vocab_size, batch, context, nonce)
            backend.generate(warmup, min(args.output_tokens, 4))
            nonce += 1
            samples = []
            for _ in range(args.repeats):
                prompts = make_prompts(backend.vocab_size, batch, context, nonce)
                nonce += 1
                samples.append(backend.generate(prompts, args.output_tokens))
            total_s = statistics.median(item["total_s"] for item in samples)
            output_tokens = int(statistics.median(item["output_tokens"] for item in samples))
            rows.append(
                {
                    "batch_size": batch,
                    "context_tokens": context,
                    "output_tokens": output_tokens,
                    "latency_s": total_s,
                    "output_tps": output_tokens / total_s,
                    "ttft_p50_ms": 1000 * statistics.median(item["ttft_p50_s"] for item in samples),
                    "ttft_p99_ms": 1000 * statistics.median(item["ttft_p99_s"] for item in samples),
                }
            )
    return rows


def benchmark_prefix(backend: Any, args: argparse.Namespace) -> dict[str, Any]:
    prefix_len = min(args.prefix_tokens, max(args.contexts) - args.prefix_suffix_tokens)
    prefix_len -= prefix_len % args.block_size
    prefix = make_prompts(backend.vocab_size, 1, prefix_len, 9000)[0]
    suffix_a = make_prompts(backend.vocab_size, 1, args.prefix_suffix_tokens, 9001)[0]
    suffix_b = make_prompts(backend.vocab_size, 1, args.prefix_suffix_tokens, 9002)[0]

    # Compile/warm the cached-prefix path with a disjoint prefix so its one-time
    # cost is not attributed to the measured warm request.
    calibration = make_prompts(backend.vocab_size, 1, prefix_len, 8000)[0]
    calibration_suffix_a = make_prompts(
        backend.vocab_size, 1, args.prefix_suffix_tokens, 8001
    )[0]
    calibration_suffix_b = make_prompts(
        backend.vocab_size, 1, args.prefix_suffix_tokens, 8002
    )[0]
    backend.generate([calibration + calibration_suffix_a], 1)
    backend.generate([calibration + calibration_suffix_b], 1)

    cold = backend.generate([prefix + suffix_a], 1)
    warm = backend.generate([prefix + suffix_b], 1)
    pinned_tokens = backend.pin_prefix(prefix)
    suffix_c = make_prompts(backend.vocab_size, 1, args.prefix_suffix_tokens, 9003)[0]
    pinned = backend.generate([prefix + suffix_c], 1)
    return {
        "prefix_tokens": prefix_len,
        "suffix_tokens": args.prefix_suffix_tokens,
        "cold_ttft_ms": cold["ttft_p50_s"] * 1000,
        "warm_ttft_ms": warm["ttft_p50_s"] * 1000,
        "pinned_ttft_ms": pinned["ttft_p50_s"] * 1000,
        "warm_speedup": cold["ttft_p50_s"] / max(warm["ttft_p50_s"], 1e-12),
        "pinned_tokens": pinned_tokens,
        "cache": pinned["cache"],
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    free_before, total_memory = torch.cuda.mem_get_info()
    used_before_mb = (total_memory - free_before) / 2**20
    init_started = time.perf_counter()
    backend = NaivBackend(args) if args.worker_backend == "naivllm" else VllmBackend(args)
    torch.cuda.synchronize()
    ready_at = time.perf_counter()
    memory_after_init = gpu_used_mb(torch)

    cold_prompt = make_prompts(backend.vocab_size, 1, min(args.contexts), 1)
    backend.generate(cold_prompt, 1)
    first_token_at = time.perf_counter()
    result = {
        "backend": args.worker_backend,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vllm": getattr(backend, "vllm_version", None),
            "model": args.model,
            "enforce_eager": args.enforce_eager if args.worker_backend == "naivllm" else False,
            "prefix_caching": True,
        },
        "cold_start": {
            "engine_init_s": ready_at - init_started,
            "process_to_ready_s": ready_at - PROCESS_STARTED,
            "process_to_first_token_s": first_token_at - PROCESS_STARTED,
            "gpu_memory_delta_mb": memory_after_init - used_before_mb,
            "gpu_memory_used_after_init_mb": memory_after_init,
            "host_peak_rss_mb": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if resource is not None
                else None
            ),
        },
        "curve": benchmark_curve(backend, args),
        "prefix": benchmark_prefix(backend, args),
    }
    backend.close()
    return result


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["backend", "batch_size", "context_tokens", "output_tokens", "latency_s", "output_tps", "ttft_p50_ms", "ttft_p99_ms"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for row in result["curve"]:
                writer.writerow({"backend": result["backend"], **row})


def write_plot(results: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    batches = sorted({row["batch_size"] for result in results for row in result["curve"]})
    fig, axes = plt.subplots(1, len(batches), figsize=(6 * len(batches), 4), squeeze=False)
    for axis, batch in zip(axes[0], batches):
        for result in results:
            rows = sorted(
                (row for row in result["curve"] if row["batch_size"] == batch),
                key=lambda row: row["context_tokens"],
            )
            axis.plot([row["context_tokens"] for row in rows], [row["output_tps"] for row in rows], marker="o", label=result["backend"])
        axis.set_title(f"batch={batch}")
        axis.set_xlabel("Context tokens")
        axis.set_ylabel("Output tokens/s")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def orchestrate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in ("naivllm", "vllm"):
        output_path = output_dir / f"{backend}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker-backend", backend,
            "--worker-output", str(output_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        completed = subprocess.run(command, env=env, text=True)
        if completed.returncode:
            raise SystemExit(f"{backend} worker failed with exit code {completed.returncode}")
        results.append(json.loads(output_path.read_text(encoding="utf-8")))

    combined = output_dir / "results.json"
    combined.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(results, output_dir / "curve.csv")
    write_plot(results, output_dir / "curve.png")
    print(f"Results: {combined}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="benchmark-results/narrow-regime")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batches", type=parse_ints, default=parse_ints("1,2,4,8"))
    parser.add_argument("--contexts", type=parse_ints, default=parse_ints("32,128,512"))
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--prefix-suffix-tokens", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--worker-backend", choices=("naivllm", "vllm"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_backend:
        result = run_worker(args)
        Path(args.worker_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(RESULT_MARKER + args.worker_output)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
