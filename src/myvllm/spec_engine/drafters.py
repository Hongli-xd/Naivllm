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

    def draft(self, seqs: list[Sequence]) -> list[list[int]]:
        drafts = []
        for seq in seqs:
            history = seq.token_ids
            drafts.append(self._lookup(history, self.num_speculative_tokens))
        return drafts

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
        """Run K draft decode steps; returns (draft_tokens, draft_logits).

        Drafted tokens are appended to seq.token_ids (rolled back by the
        engine after verification) and their distributions are kept for
        rejection sampling.
        """
        k = self.num_speculative_tokens
        draft_tokens = [[] for _ in seqs]
        draft_logit_rows = [[] for _ in seqs]
        for _ in range(k):
            for seq in seqs:
                self.block_manager.ensure_capacity(seq, len(seq) + 1)
            outputs = self.runner.call("run", seqs, False)
            tokens, logits = outputs
            tokens = tokens.tolist() if hasattr(tokens, "tolist") else tokens
            for i, (seq, token) in enumerate(zip(seqs, tokens)):
                seq.append_token(token)
                seq.draft_tokens.append(token)
                draft_tokens[i].append(token)
                if logits is not None:
                    draft_logit_rows[i].append(logits[i])
        return draft_tokens, draft_logit_rows
