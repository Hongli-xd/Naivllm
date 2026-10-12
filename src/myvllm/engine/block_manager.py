import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1
        self.ref_count = 0
        self.pin_count = 0
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.ref_count = 0
        self.pin_count = 0
        self.token_ids = []

class BlockManager:
    """Paged KV-cache block bookkeeping.

    ``table_attr``/``cached_attr`` name the Sequence attributes this manager
    owns. The defaults keep the single-model behavior; a speculative-decoding
    draft model uses a second manager over ``draft_block_table`` so target and
    draft caches stay independent.
    """

    def __init__(self, num_blocks: int, block_size: int,
                 table_attr: str = 'block_table', cached_attr: str = 'num_cached_tokens'):
        # block_size: number of tokens per block
        self.block_size: int = block_size
        self.table_attr = table_attr
        self.cached_attr = cached_attr
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()
        self.cache_hits = 0
        self.cache_misses = 0

    def _table(self, seq: Sequence) -> list[int]:
        return getattr(seq, self.table_attr)

    def _cached(self, seq: Sequence) -> int:
        return getattr(seq, self.cached_attr)

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _acquire_cached_block(self, block_id: int) -> Block:
        """Acquire a cached block without clearing its hash or token metadata."""
        block = self.blocks[block_id]
        if block_id not in self.used_block_ids:
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        block.ref_count += 1
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        seq.block_size = self.block_size
        h = -1
        misses = 0
        # at least one token must be (re)computed so the sequence gets a
        # first token from the model; the prefix cache may therefore not
        # cover the very last token
        max_cached_tokens = seq.num_tokens - 1
        cached_tokens = 0
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            block_id = self.hash_to_block_id.get(h, -1)
            if (
                h == -1
                or block_id == -1
                or self.blocks[block_id].token_ids != token_ids
                or cached_tokens + self.block_size > max_cached_tokens
            ):
                misses += 1
            else:
                cached_tokens += self.block_size
        return len(self.free_block_ids) >= misses


    def allocate(self, seq: Sequence) -> None:
        seq.block_size = self.block_size
        table = self._table(seq)
        h = -1
        max_cached_tokens = seq.num_tokens - 1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)

            # if cache miss or hash collision or the block would consume the
            # last token (leaving nothing to prefill), allocate a fresh block
            if (
                block_id == -1
                or self.blocks[block_id].token_ids != token_ids
                or self._cached(seq) + self.block_size > max_cached_tokens
            ):
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                setattr(seq, self.cached_attr, self._cached(seq) + self.block_size) # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                block = self._acquire_cached_block(block_id)
                self.cache_hits += 1
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                block.ref_count = 1
                self.cache_misses += 1
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            table.append(block.block_id)

    def deallocate(self, seq: Sequence) -> None:
        # update block information
        for block_id in self._table(seq):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        setattr(seq, self.table_attr, [])
        setattr(seq, self.cached_attr, 0)

    def pin_prefix(self, token_ids: list[int]) -> int:
        """Keep every cached full block in ``token_ids`` resident.

        Returns the number of newly pinned tokens. The prefix must have been
        evaluated once so its KV blocks already exist in the cache.
        """
        h = -1
        pinned_tokens = 0
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            block_tokens = token_ids[start : start + self.block_size]
            h = self.compute_hash(block_tokens, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != block_tokens:
                break
            block = self.blocks[block_id]
            if block.pin_count == 0:
                self._acquire_cached_block(block_id)
            block.pin_count += 1
            pinned_tokens += self.block_size
        return pinned_tokens

    def unpin_all(self) -> None:
        for block in self.blocks:
            if block.pin_count == 0:
                continue
            block.ref_count -= 1
            block.pin_count = 0
            if block.ref_count == 0:
                self._deallocate_block(block.block_id)

    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "pinned_blocks": sum(block.pin_count > 0 for block in self.blocks),
            "free_blocks": len(self.free_block_ids),
        }

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        table = self._table(seq)
        n = seq.num_tokens

        if n % self.block_size == 0:
            # the last block is now full: finalize (or re-confirm) its hash
            block = self.blocks[table[-1]]
            if block.hash == -1:
                prev_hash = self.blocks[table[-2]].hash if len(table) > 1 else -1
                token_ids = seq.block(len(table) - 1)
                h = self.compute_hash(token_ids, prev_hash)
                block.update(h=h, token_ids=token_ids)
                self.hash_to_block_id[h] = block.block_id
        elif len(table) < seq.num_blocks:
            # the next token crosses into a new block that is not allocated
            # yet (speculative rounds may have pre-allocated it already)
            block = self._allocate_block(self.free_block_ids[0])
            block.ref_count = 1
            table.append(block.block_id)
        else:
            assert table[-1] in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[table[-1]].hash == -1, "Last block should be partial block with hash -1"

    # how many extra blocks ``target_num_tokens`` needs beyond the current table
    def _blocks_needed_for(self, seq: Sequence, target_num_tokens: int) -> int:
        needed = (target_num_tokens + self.block_size - 1) // self.block_size
        return max(0, needed - len(self._table(seq)))

    def can_ensure_capacity(self, seq: Sequence, target_num_tokens: int) -> bool:
        return len(self.free_block_ids) >= self._blocks_needed_for(seq, target_num_tokens)

    def ensure_capacity(self, seq: Sequence, target_num_tokens: int) -> None:
        """Extend the table so it covers ``target_num_tokens`` positions.

        Used by speculative decoding before a multi-token forward. Fresh
        blocks stay hash-less; they are either confirmed later or rolled back.
        """
        while self._blocks_needed_for(seq, target_num_tokens) > 0:
            block = self._allocate_block(self.free_block_ids[0])
            block.ref_count = 1
            self._table(seq).append(block.block_id)

    def rollback(self, seq: Sequence, num_tokens: int) -> None:
        """Roll back speculative tokens and release no-longer-needed blocks."""
        if num_tokens <= 0:
            return
        if num_tokens > seq.num_completion_tokens:
            raise ValueError("rollback cannot remove prompt tokens")

        seq.token_ids = seq.token_ids[:-num_tokens]
        seq.num_tokens -= num_tokens
        seq.last_token = seq.token_ids[-1]
        self.trim_to(seq, seq.num_tokens)

    # free blocks beyond what ``new_num_tokens`` needs and clear the partial
    # tail block's hash; token_ids is left untouched (the caller owns it)
    def trim_to(self, seq: Sequence, new_num_tokens: int) -> None:
        needed_blocks = (new_num_tokens + self.block_size - 1) // self.block_size
        table = self._table(seq)
        while len(table) > needed_blocks:
            block_id = table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        if not table:
            return
        if new_num_tokens % self.block_size:
            block = self.blocks[table[-1]]
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id:
                del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []
        else:
            # confirmed tokens land exactly on a block boundary: finalize the
            # block's hash so later requests can prefix-cache against it (the
            # speculative path bypasses append()'s finalization)
            block = self.blocks[table[-1]]
            prev_hash = self.blocks[table[-2]].hash if len(table) > 1 else -1
            token_ids = seq.block(needed_blocks - 1)
            h = self.compute_hash(token_ids, prev_hash)
            block.update(h=h, token_ids=token_ids)
            self.hash_to_block_id[h] = block.block_id
