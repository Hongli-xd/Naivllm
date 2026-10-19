"""Rejection sampling for speculative decoding (Leviathan et al., 2023).

Three verification modes:

- greedy (temperature == 0): a drafted token is accepted iff it equals the
  target argmax; outputs are exactly the target's greedy decode.
- stochastic with draft distribution (model drafter): accept token t ~
  min(1, p_target(d)/p_draft(d)); on rejection resample from the positive
  part of p_target - p_draft. Provably samples from the target distribution.
- stochastic without a draft distribution (n-gram drafter): draw x ~
  p_target at every drafted position and accept while x_i == d_i; the first
  mismatch emits x_j. This also preserves the target distribution exactly,
  because the emitted token is always a target sample.

In every mode, when all K drafted tokens are accepted, one bonus token is
emitted from the target distribution of position K.
"""

from __future__ import annotations

import torch


class RejectionSampler:
    def verify_greedy(self, draft_tokens, target_rows) -> list[int]:
        """target_rows: (K+1, vocab) logits, or a ready (K+1,) token list."""
        target_tokens = (target_rows if isinstance(target_rows, list)
                         else target_rows.argmax(dim=-1).tolist())
        accepted = []
        for drafted, target_tok in zip(draft_tokens, target_tokens):
            if drafted != target_tok:
                return accepted + [target_tok]
            accepted.append(drafted)
        # every draft accepted: emit the bonus token
        accepted.append(target_tokens[len(draft_tokens)])
        return accepted

    def verify_stochastic(
        self,
        draft_tokens: list[int],
        target_rows: torch.Tensor,
        draft_rows: torch.Tensor | None,
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """target_rows: (K+1, vocab); draft_rows: (K, vocab) or None."""
        p_target = torch.softmax(target_rows / temperature, dim=-1)
        K = len(draft_tokens)
        accepted: list[int] = []

        if draft_rows is not None:
            p_draft = torch.softmax(draft_rows / temperature, dim=-1)
            draft_ids = torch.tensor(draft_tokens, device=target_rows.device, dtype=torch.long)
            probs_draft = p_draft.gather(1, draft_ids[:, None]).squeeze(1)
            probs_target = p_target[:-1].gather(1, draft_ids[:, None]).squeeze(1)
            ratio = (probs_target / probs_draft.clamp_min(1e-20)).clamp(max=1.0)
            uniforms = torch.rand(K, device=target_rows.device, generator=generator)
            for i in range(K):
                if uniforms[i] < ratio[i]:
                    accepted.append(draft_tokens[i])
                    continue
                # rejected: resample from the positive part of p_t - p_d
                residual = (p_target[i] - p_draft[i]).clamp_min(0)
                if residual.sum() <= 0:
                    residual = p_target[i]
                new_token = int(torch.multinomial(
                    residual / residual.sum(), 1, generator=generator).item())
                accepted.append(new_token)
                return accepted
            # all accepted: bonus token from the K-th target distribution
            accepted.append(int(torch.multinomial(
                p_target[K], 1, generator=generator).item()))
            return accepted

        # no draft distribution (n-gram): compare against target samples
        samples = torch.multinomial(
            p_target, 1, generator=generator).squeeze(1).tolist()
        for i in range(K):
            if samples[i] == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                accepted.append(samples[i])
                return accepted
        accepted.append(samples[K])
        return accepted

    def verify(
        self,
        draft_tokens: list[int],
        target_rows: torch.Tensor,
        draft_rows: torch.Tensor | None = None,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        if temperature <= 1e-6:
            return self.verify_greedy(draft_tokens, target_rows)
        return self.verify_stochastic(
            draft_tokens, target_rows, draft_rows, temperature, generator)
