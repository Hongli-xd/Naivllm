"""Render RAG-benchmark charts from the definitive result JSONs.

Outputs (docs/):
  rag-throughput.png   output tokens/s vs batch size, all backends
  rag-spec-decode.png  NaivLLM speculative-decoding self-speedup + acceptance
  rag-coldstart.png    engine init time
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "benchmark-results"
DOCS = Path(__file__).resolve().parent.parent / "docs"

COLD_START_S = {"NaivLLM +spec": 3.1, "vLLM 0.23 +spec": 44.9}


def load(name):
    data = json.loads((RESULTS / name).read_text())
    if isinstance(data, list):
        data = data[0]
    return {row["batch_size"]: row for row in data["rows"]}


def main():
    DOCS.mkdir(exist_ok=True)
    ngram = load("definitive.json")
    plain = load("definitive-plain.json")
    vspec = load("definitive-vspec-k8.json") if (RESULTS / "definitive-vspec-k8.json").exists() else None
    vllm_rows = None
    for name in ("definitive-vllm.json",):
        if (RESULTS / name).exists():
            vllm_rows = load(name)
    # definitive.json holds naivllm(ngram) / vllm-spec / vllm / ollama workers
    definitive_all = json.loads((RESULTS / "definitive.json").read_text())
    per_backend = {}
    for entry in definitive_all:
        per_backend[entry["backend"]] = {r["batch_size"]: r for r in entry["rows"]}

    ollama = per_backend.get("ollama")
    vllm = per_backend.get("vllm")
    vspec = per_backend.get("vllm-spec")
    ngram = per_backend.get("naivllm")
    batches = sorted(ngram)

    # ---------------- fig 1: throughput ----------------
    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = [
        ("NaivLLM + ngram spec (ours)", ngram, "#d62728", "o"),
        ("vLLM + ngram spec", vspec, "#1f77b4", "s"),
        ("vLLM 0.23", vllm, "#7fb3d5", "^"),
        ("NaivLLM plain", plain, "#e6a0a0", "v"),
        ("Ollama fp16", ollama, "#7f7f7f", "D"),
    ]
    for label, rows, color, marker in series:
        if rows:
            ax.plot(batches, [rows[b]["output_tps"] for b in batches],
                    marker=marker, color=color, label=label, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels(batches)
    ax.set_xlabel("Concurrent requests (batch size)")
    ax.set_ylabel("Output tokens/s (log)")
    ax.set_title("RAG grounded-QA throughput — Qwen3-0.6B, 1200-token doc prefix,\n"
                 "48-token extractive answers, greedy, single RTX 4090", fontsize=10)
    if ollama and ngram:
        ax.annotate(f"{ngram[1]['output_tps']/ollama[1]['output_tps']:.1f}× vs Ollama",
                    xy=(1, ngram[1]["output_tps"]), xytext=(1.15, 200),
                    fontsize=9, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "rag-throughput.png", dpi=160)
    plt.close(fig)

    # ---------------- fig 2: spec self-speedup ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    width = 0.36
    x = range(len(batches))
    ax1.bar([i - width / 2 for i in x],
            [plain[b]["output_tps"] for b in batches], width,
            label="NaivLLM plain", color="#e6a0a0")
    ax1.bar([i + width / 2 for i in x],
            [ngram[b]["output_tps"] for b in batches], width,
            label="NaivLLM + ngram (K=8)", color="#d62728")
    for i, b in enumerate(batches):
        speedup = ngram[b]["output_tps"] / plain[b]["output_tps"]
        ax1.text(i + width / 2, ngram[b]["output_tps"] + 25, f"{speedup:.2f}×",
                 ha="center", fontsize=9, color="#d62728", fontweight="bold")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(batches)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("Output tokens/s")
    ax1.set_title("Speculative decoding speedup (self-A/B)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    acc = [ngram[b]["spec"]["accepted_tokens"] / max(1, ngram[b]["spec"]["drafted_tokens"])
           for b in batches]
    tpr = [ngram[b]["spec"]["accepted_tokens"] / max(1, ngram[b]["spec"]["rounds"])
           for b in batches]
    ax2_t = ax2.twinx()
    ax2.bar(list(x), acc, 0.5, color="#9ecae1", label="acceptance rate")
    ax2_t.plot(list(x), tpr, "o-", color="#08519c", label="accepted tokens / round")
    for i, a in enumerate(acc):
        ax2.text(i, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2_t.set_ylim(0, max(tpr) * 1.35)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(batches)
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("draft acceptance rate")
    ax2_t.set_ylabel("accepted tokens per round")
    ax2.set_title("n-gram drafting quality on extractive answers", fontsize=10)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "rag-spec-decode.png", dpi=160)
    plt.close(fig)

    # ---------------- fig 3: cold start ----------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names = list(COLD_START_S)
    vals = [COLD_START_S[n] for n in names]
    bars = ax.barh(names, vals, color=["#d62728", "#1f77b4"])
    for bar, v in zip(bars, vals):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}s",
                va="center", fontsize=10)
    ax.set_xlabel("process start → engine ready (s, lower is better)")
    ax.set_title("Cold start with speculative decoding", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(DOCS / "rag-coldstart.png", dpi=160)
    plt.close(fig)

    print("charts written to", DOCS)


if __name__ == "__main__":
    sys.exit(main())
