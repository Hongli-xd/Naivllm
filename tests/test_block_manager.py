from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def test_exact_multiple_has_a_full_last_block():
    sequence = Sequence([1, 2, 3, 4])
    sequence.block_size = 4
    assert sequence.last_block_num_tokens == 4
    assert sequence.block(0) == [1, 2, 3, 4]


def test_cached_block_survives_release_and_is_reused():
    manager = BlockManager(num_blocks=4, block_size=4)
    first = Sequence([1, 2, 3, 4, 5])
    manager.allocate(first)
    cached_id = first.block_table[0]
    manager.deallocate(first)

    second = Sequence([1, 2, 3, 4, 9])
    manager.allocate(second)

    assert second.block_table[0] == cached_id
    assert second.num_cached_tokens == 4
    assert manager.blocks[cached_id].token_ids == [1, 2, 3, 4]
    assert manager.cache_stats()["hits"] == 1


def test_pinned_prefix_is_not_returned_to_free_pool():
    manager = BlockManager(num_blocks=4, block_size=4)
    sequence = Sequence([1, 2, 3, 4, 5])
    manager.allocate(sequence)
    prefix_id = sequence.block_table[0]
    manager.deallocate(sequence)

    assert manager.pin_prefix([1, 2, 3, 4]) == 4
    assert prefix_id in manager.used_block_ids
    assert prefix_id not in manager.free_block_ids

    manager.unpin_all()
    assert prefix_id in manager.free_block_ids


def test_rollback_releases_extra_blocks():
    manager = BlockManager(num_blocks=4, block_size=4)
    sequence = Sequence([1, 2, 3, 4])
    manager.allocate(sequence)
    sequence.append_token(5)
    manager.append(sequence)

    manager.rollback(sequence, 1)

    assert sequence.token_ids == [1, 2, 3, 4]
    assert len(sequence.block_table) == 1
