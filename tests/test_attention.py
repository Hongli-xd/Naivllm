import math

import pytest
import torch
import torch.nn.functional as F

from myvllm.layers.attention import flash_attention_prefill, paged_attention_decode


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_varlen_prefill_matches_sdpa_for_gqa():
    torch.manual_seed(0)
    lengths = [7, 5]
    num_heads, num_kv_heads, head_dim = 4, 2, 64
    total = sum(lengths)
    q = torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cu = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(head_dim)

    actual = flash_attention_prefill(q, k, v, cu, scale, num_heads, num_kv_heads, head_dim)
    expected_parts = []
    start = 0
    for length in lengths:
        q_seq = q[start : start + length].transpose(0, 1).unsqueeze(0)
        k_seq = k[start : start + length].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0)
        v_seq = v[start : start + length].repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0)
        expected = F.scaled_dot_product_attention(q_seq, k_seq, v_seq, is_causal=True)
        expected_parts.append(expected.squeeze(0).transpose(0, 1))
        start += length

    torch.testing.assert_close(actual, torch.cat(expected_parts), atol=4e-2, rtol=4e-2)


def test_vectorized_paged_decode_matches_sdpa():
    torch.manual_seed(1)
    batch, num_heads, num_kv_heads, head_dim = 2, 4, 2, 64
    block_size, context_len = 4, 7
    query = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, context_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_cache = torch.zeros(batch * 2, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    for batch_index in range(batch):
        for token in range(context_len):
            block = int(block_tables[batch_index, token // block_size])
            k_cache[block, token % block_size] = k[batch_index, token]
            v_cache[block, token % block_size] = v[batch_index, token]
    context_lens = torch.full((batch,), context_len, device="cuda", dtype=torch.long)
    scale = 1 / math.sqrt(head_dim)

    actual = paged_attention_decode(
        query, k_cache, v_cache, block_tables, context_lens,
        scale, num_heads, num_kv_heads, head_dim, block_size,
    )
    expanded_k = k.repeat_interleave(2, dim=2)
    expanded_v = v.repeat_interleave(2, dim=2)
    scores = torch.einsum("bhd,bkhd->bhk", query, expanded_k).float() * scale
    probabilities = torch.softmax(scores, dim=-1).to(expanded_v.dtype)
    expected = torch.einsum("bhk,bkhd->bhd", probabilities, expanded_v)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
