"""Token drafters for speculative decoding.

A drafter proposes, per running sequence, up to ``num_speculative_tokens``
candidate continuations. Two strategies are implemented, mirroring vLLM's
speculative-decoding methods:

- ``NgramDrafter`` ("ngram" / prompt-lookup decoding): continuations are
  copied from n-gram matches against the prompt plus everything generated so
  far. No draft model, no extra forward pass — ideal for extractive
  workloads (RAG answers, code edits, quoting) where the output repeats
  spans of the context.
- ``ModelDrafter`` ("draft model"): a small draft model runs K autoregressive
  decode steps through its own KV cache.

Both feed the same target-side verification path (ModelRunner.run_spec_verify
followed by rejection sampling).
"""

from __future__ import annotations

import numpy as np

from myvllm.engine.sequence import Sequence


class NgramDrafter:
    """Prompt-lookup drafting: copy continuations of n-gram matches.

    For n from ``max_ngram_size`` down to ``min_ngram_size``, look for the
    most recent occurrence of the current n-gram suffix inside the sequence
    history; if found, continue with the tokens that followed it. Longer
    matches win because they are stronger predictors.

    Matching runs on a fixed-width big-endian byte encoding of the token ids
    so the hot loop is ``bytes.rfind`` (C speed) instead of Python slicing.
    """

    name = "ngram"

    def __init__(self, num_speculative_tokens: int = 5,
                 min_ngram_size: int = 2, max_ngram_size: int = 8):
        self.num_speculative_tokens = num_speculative_tokens
        self.min_ngram_size = min_ngram_size
        self.max_ngram_size = max_ngram_size

    def draft(self, seqs: list[Sequence]) -> tuple[list[list[int]], list]:
        """Append drafted tokens onto each sequence (rolled back by the
        engine after verification); returns (draft_lists, draft_logit_rows).

        draft_logit_rows is None per sequence — n-gram drafts carry no
        distribution (the rejection sampler uses its distribution-free
        accept-if-equals-target-sample rule).
        """
        drafts = []
        for seq in seqs:
            drafted = self._lookup(seq.token_ids, self.num_speculative_tokens)
            seq.draft_tokens = list(drafted)
            for token in drafted:
                seq.append_token(token)
            drafts.append(drafted)
        return drafts, [None] * len(seqs)

    def _lookup(self, history: list[int], k: int) -> list[int]:
        length = len(history)
        for n in range(min(self.max_ngram_size, length - 1),
                       self.min_ngram_size - 1, -1):
            needle = np.asarray(history[length - n:], dtype='>u4').tobytes()
            haystack = np.asarray(history, dtype='>u4').tobytes()
            # a usable match leaves at least one continuation token, so the
            # match must end before the final token
            end = 4 * (length - 1)
            while True:
                idx = haystack.rfind(needle, 0, end)
                if idx == -1 or idx % 4 == 0:
                    break
                # misaligned hit inside a token; keep searching left
                end = idx
            if idx != -1:
                start = idx // 4 + n
                continuation = history[start : start + k]
                if continuation:
                    return continuation
        return []


class ModelDrafter:
    """Draft with a small model running K autoregressive decode steps.

    The draft ModelRunner owns its own paged KV cache (tracked through
    ``seq.draft_block_table`` by a separate BlockManager) and its own CUDA
    graphs, so drafting costs K cheap decode replays per round.
    """

    name = "model"

    def __init__(self, runner, block_manager, num_speculative_tokens: int = 5):
        self.runner = runner
        self.block_manager = block_manager
        self.num_speculative_tokens = num_speculative_tokens

    def can_draft(self, seqs: list[Sequence], k: int) -> bool:
        # all sequences must have draft-cache capacity for k more tokens;
        # otherwise the round falls back to plain decoding
        return all(
            self.block_manager.can_ensure_capacity(seq, len(seq) + k + 1)
            for seq in seqs
        )

    def prefill(self, seqs: list[Sequence]) -> None:
        """Mirror the target-side prefill into the draft KV cache."""
        for seq in seqs:
            if not getattr(seq, 'draft_block_table'):
                self.block_manager.allocate(seq)
        self.runner.call("run", seqs, True)

    def draft(self, seqs: list[Sequence]) -> tuple[list[list[int]], list]:
        """Run K draft decode steps; returns (draft_tokens, draft_logit_rows).

        Drafted tokens are appended to seq.token_ids (rolled back by the
        engine after verification) and their distributions are kept for
        rejection sampling. The hot path is per-round amortized: capacity is
        ensured once for all K steps and the temperature tensor is built once.
        """
        k = self.num_speculative_tokens
        for seq in seqs:
            self.block_manager.ensure_capacity(seq, len(seq) + k + 1)
        temperatures = self.runner.prepare_sample(seqs)
        draft_tokens = [[] for _ in seqs]
        draft_logit_rows = [[] for _ in seqs]
        for _ in range(k):
            tokens, logits = self.runner.call("run", seqs, False, temperatures)
            for i, (seq, token) in enumerate(zip(seqs, tokens)):
                seq.append_token(token)
                seq.draft_tokens.append(token)
                draft_tokens[i].append(token)
                if logits is not None:
                    draft_logit_rows[i].append(logits[i])
        return draft_tokens, draft_logit_rows


class HybridDrafter:
    """Cascade drafter: free n-gram lookup first, small-model fallback.

    Per round per sequence: if the n-gram match hits, its (zero-cost)
    continuation is used; only sequences with no match fall through to the
    model drafter, which costs K decode replays for that subset. Sequences
    that draft nothing either way degrade to a plain decode step in the
    engine. ``model_rounds``/``model_seqs`` count the fallback usage so the
    benchmark can attribute where accepted tokens came from.
    """

    name = "hybrid"

    def __init__(self, ngram: NgramDrafter, model: ModelDrafter):
        self.ngram = ngram
        self.model = model
        self.model_rounds = 0
        self.model_seqs = 0

    def prefill(self, seqs: list[Sequence]) -> None:
        self.model.prefill(seqs)

    def can_draft(self, seqs: list[Sequence], k: int) -> bool:
        # n-gram needs no capacity; the model fallback degrades gracefully
        # (empty drafts → plain decode), so a hybrid round can always run
        return True

    def draft(self, seqs: list[Sequence]) -> tuple[list[list[int]], list]:
        rows = [None] * len(seqs)
        self.ngram.draft(seqs)
        rest_idx = [i for i, seq in enumerate(seqs) if not seq.draft_tokens]
        if rest_idx:
            rest = [seqs[i] for i in rest_idx]
            if self.model.can_draft(rest, self.model.num_speculative_tokens):
                _, model_rows = self.model.draft(rest)
                self.model_rounds += 1
                self.model_seqs += len(rest)
                for j, i in enumerate(rest_idx):
                    rows[i] = model_rows[j] if j < len(model_rows) else None
        return [list(s.draft_tokens) for s in seqs], rows
