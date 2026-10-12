from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    # speculative-decoding verification: multi-query decode over the paged
    # cache (queries = drafted tokens, causal mask via absolute positions)
    is_verify: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seqlens_q: list[int] | None = None
    seqlens_k: list[int] | None = None
    # absolute position of every token in the current flattened batch
    # (prefill-with-cache and verify modes; used for RoPE and the verify
    # kernel's causal mask)
    positions: torch.Tensor | None = None

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, seqlens_q=None, seqlens_k=None, is_verify=False, positions=None):
    global _context
    _context = Context(is_prefill, is_verify, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, seqlens_q, seqlens_k, positions)
