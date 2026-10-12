"""Speculative-decoding engine (target model + pluggable drafter).

Flow per decode round (cf. vLLM's spec-decode worker and Leviathan et al.):

1. drafting — either n-gram lookup over prompt+output (no extra model) or a
   small draft model running K autoregressive decode steps against its own
   KV cache;
2. verification — the target model runs ONE multi-token forward over
   [last_confirmed, d_1..d_K] through the new paged verify kernel, producing
   K+1 logits rows;
3. rejection sampling — greedy exact-match or distribution-preserving
   stochastic verification; accepted tokens (plus one bonus token) are
   committed, the rest of the speculative tail is rolled back in both the
   sequence and the block tables.
"""

from __future__ import annotations

import time

import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.block_manager import BlockManager
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.sequence import SequenceStatus
from myvllm.sampling_parameters import SamplingParams
from myvllm.spec_engine.drafters import NgramDrafter, ModelDrafter
from myvllm.spec_engine.rejection_sampler import RejectionSampler


class SpecLLMEngine(LLMEngine):
    def __init__(self, config: dict):
        assert config.get("world_size", 1) == 1, \
            "speculative decoding currently supports world_size == 1"
        self.spec_method = config.get("spec_method", "ngram")
        self.num_speculative_tokens = config.get("num_speculative_tokens", 5)

        # the target model may share the GPU with a draft model; reserve the
        # draft share before the target grabs the rest
        self.draft_runner = None
        self.draft_block_manager = None
        target_config = dict(config)
        if self.spec_method != "none":
            # verification forwards replay a captured CUDA graph (one per
            # batch bucket) instead of ~28 layers of eager Python dispatch
            target_config["capture_verify"] = True
            target_config["num_spec_tokens"] = self.num_speculative_tokens
        if self.spec_method == "model":
            target_config["gpu_memory_utilization"] = config.get(
                "target_gpu_memory_utilization",
                max(0.1, config.get("gpu_memory_utilization", 0.9) - 0.3),
            )

        super().__init__(target_config)

        self.sampler = RejectionSampler()
        self.spec_stats = {
            "method": self.spec_method,
            "num_speculative_tokens": self.num_speculative_tokens,
            "rounds": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "fallback_rounds": 0,
        }

        if self.spec_method == "model":
            draft_config = dict(config)
            draft_config.update(
                model_name_or_path=config.get("draft_model"),
                gpu_memory_utilization=config.get("draft_gpu_memory_utilization", 0.3),
                seq_table_attr="draft_block_table",
                seq_cached_attr="draft_num_cached_tokens",
                capture_logits=True,
                enforce_eager=False,
                world_size=1,
                dist_port=config.get("dist_port", 12345) + 1,
            )
            self._build_draft_runner(draft_config)
            self.drafter = ModelDrafter(
                self.draft_runner, self.draft_block_manager, self.num_speculative_tokens
            )
        elif self.spec_method == "ngram":
            self.drafter = NgramDrafter(
                num_speculative_tokens=self.num_speculative_tokens,
                min_ngram_size=config.get("ngram_min_size", 2),
                max_ngram_size=config.get("ngram_max_size", 8),
            )
        else:
            self.drafter = None

    def _build_draft_runner(self, draft_config: dict) -> None:
        """Load the draft model architecture from its HF config."""
        from myvllm.config import build_engine_config

        base = build_engine_config(
            draft_config["model_name_or_path"],
            max_num_sequences=draft_config.get("max_num_sequences", 16),
            max_num_batched_tokens=draft_config.get("max_num_batched_tokens", 4096),
            max_model_length=draft_config.get("max_model_length", 2048),
            block_size=draft_config.get("block_size", 16),
            gpu_memory_utilization=draft_config.get("gpu_memory_utilization", 0.3),
            enforce_eager=False,
        )
        assert base["vocab_size"] == self.model_runner.config["vocab_size"], \
            "draft and target models must share the tokenizer/vocabulary"
        base.update(draft_config)
        # per-draft-step block bookkeeping happens through draft_block_table
        self.draft_block_manager = BlockManager(
            base["max_cached_blocks"],
            base["block_size"],
            table_attr="draft_block_table",
            cached_attr="draft_num_cached_tokens",
        )
        self.draft_runner = ModelRunner(base, rank=0, event=[])

    # ------------------------------------------------------------------ step
    def step(self):
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill

        if is_prefill:
            outputs = self.model_runner.call("run", scheduled_sequences, True)
            if isinstance(self.drafter, ModelDrafter):
                # mirror the prompt prefill into the draft KV cache before
                # the first completion token is appended
                self.drafter.prefill(scheduled_sequences)
            self.scheduler.postprocess(scheduled_sequences, outputs)
        elif self.drafter is not None:
            self._speculative_step(scheduled_sequences)
        else:
            outputs = self.model_runner.call("run", scheduled_sequences, False)
            self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in scheduled_sequences
            if seq.is_finished
        ]
        num_processed_tokens = (
            sum(len(seq) for seq in scheduled_sequences)
            if is_prefill
            else len(scheduled_sequences)
        )
        return outputs, num_processed_tokens, is_prefill

    # ------------------------------------------------------- speculative core
    def _can_speculate(self, seqs) -> bool:
        K = self.num_speculative_tokens
        block_manager = self.scheduler.block_manager
        if not all(block_manager.can_ensure_capacity(seq, len(seq) + K) for seq in seqs):
            return False
        if isinstance(self.drafter, ModelDrafter) and not self.drafter.can_draft(seqs, K):
            return False
        return True

    def _speculative_step(self, scheduled_sequences) -> None:
        if not self._can_speculate(scheduled_sequences):
            # not enough KV room for a full draft round: plain decode
            self.spec_stats["fallback_rounds"] += 1
            outputs = self.model_runner.call("run", scheduled_sequences, False)
            self.scheduler.postprocess(scheduled_sequences, outputs)
            return

        K = self.num_speculative_tokens

        # ---- 1. drafting
        if isinstance(self.drafter, NgramDrafter):
            draft_lists = self.drafter.draft(scheduled_sequences)
            draft_logit_rows = [None] * len(scheduled_sequences)
            for seq, drafted in zip(scheduled_sequences, draft_lists):
                seq.draft_tokens = list(drafted)
                for token in drafted:
                    seq.append_token(token)
        else:
            draft_lists, draft_logit_rows = self.drafter.draft(scheduled_sequences)

        stats = self.spec_stats
        stats["rounds"] += 1
        stats["drafted_tokens"] += sum(len(d) for d in draft_lists)

        # sequences that drafted nothing fall back to a single decode token
        plain_seqs = [seq for seq in scheduled_sequences if not seq.draft_tokens]
        spec_seqs = [seq for seq in scheduled_sequences if seq.draft_tokens]

        if plain_seqs:
            outputs = self.model_runner.call("run", plain_seqs, False)
            self.scheduler.postprocess(plain_seqs, outputs)

        if spec_seqs:
            # ---- 2. verification forward
            for seq in spec_seqs:
                # cover positions n-1 .. n+K-1 in the target cache
                self.scheduler.block_manager.ensure_capacity(seq, len(seq))
            target_logits = self.model_runner.call("run_spec_verify", spec_seqs)
            # graph-replayed verifications return one padded K+1 block per
            # sequence; eager verifications pack rows tightly
            graphed = bool(getattr(self.model_runner, "verify_graphs", None))
            stride = (
                self.model_runner.verify_vars["k"] + 1 if graphed else None
            )

            # ---- 3. rejection sampling + commit
            row = 0
            for i, seq in enumerate(spec_seqs):
                k_i = len(seq.draft_tokens)
                rows = target_logits[row : row + k_i + 1]
                row += stride if stride is not None else k_i + 1
                draft_rows = (
                    torch.cat(draft_logit_rows[i]) if draft_logit_rows[i] else None
                )
                accepted = self.sampler.verify(
                    seq.draft_tokens,
                    rows,
                    draft_rows,
                    seq.temperature,
                )
                self._commit(seq, accepted)

    def _commit(self, seq, accepted: list[int]) -> None:
        """Fix the sequence at base + accepted, roll back speculative tail."""
        K = len(seq.draft_tokens)
        n = seq.num_tokens - K  # confirmed length before drafting

        # stop conditions cap how much of `accepted` we may emit
        allowed = max(0, seq.max_tokens - (n - seq.num_prompt_tokens))
        accepted = accepted[:allowed]

        eos = self.scheduler.eos
        final_tokens: list[int] = []
        for token in accepted:
            final_tokens.append(token)
            reached_eos = not seq.ignore_eos and token == eos
            reached_length = (
                seq.max_model_length is not None
                and n + len(final_tokens) >= seq.max_model_length
            )
            if reached_eos or reached_length:
                break

        seq.token_ids = seq.token_ids[:n] + final_tokens
        seq.num_tokens = n + len(final_tokens)
        seq.last_token = seq.token_ids[-1]
        seq.draft_tokens = []
        self.spec_stats["accepted_tokens"] += len(final_tokens)

        # release speculative blocks in every cache that tracks this sequence
        self.scheduler.block_manager.trim_to(seq, seq.num_tokens)
        if self.draft_block_manager is not None:
            self.draft_block_manager.trim_to(seq, seq.num_tokens)

        stop_eos = final_tokens and not seq.ignore_eos and final_tokens[-1] == eos
        stop_max_tokens = seq.num_completion_tokens >= seq.max_tokens
        stop_max_length = (
            seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length
        )
        if stop_eos or stop_max_tokens or stop_max_length:
            seq.status = SequenceStatus.FINISHED
            self.scheduler.block_manager.deallocate(seq)
            if self.draft_block_manager is not None:
                self.draft_block_manager.deallocate(seq)
            self.scheduler.running.remove(seq)

    # ---------------------------------------------------------------- teardown
    def exit(self):
        if self._closed:
            return
        if self.draft_runner is not None:
            try:
                self.draft_runner.call("exit")
            except Exception:
                pass
        super().exit()

    # ---------------------------------------------------------------- metrics
    def _generate(self, request_ids):
        result = super()._generate(request_ids)
        result["metrics"]["spec_decoding"] = dict(self.spec_stats)
        return result
