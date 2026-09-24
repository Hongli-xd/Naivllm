import torch

from myvllm.layers.rotary_embedding import RotaryEmbedding


def test_rotary_embedding_preserves_activation_dtype():
    rotary = RotaryEmbedding(base=1_000_000, rotary_embedding=64, max_position=16)
    query = torch.randn(4, 2, 64, dtype=torch.bfloat16)
    key = torch.randn(4, 1, 64, dtype=torch.bfloat16)
    positions = torch.arange(4)

    rotated_query, rotated_key = rotary(positions, query, key)

    assert rotated_query.dtype == query.dtype
    assert rotated_key.dtype == key.dtype
