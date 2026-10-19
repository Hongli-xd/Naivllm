import math
import torch
import pickle
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *
'''
[Transformer层0]
    ├─ LayerNorm
    ├─ Self-Attention
    │   ├─ Q投影
    │   ├─ K投影（从缓存读取）
    │   ├─ V投影（从缓存读取）
    │   ├─ 注意力计算
    │   └─ 输出投影
    ├─ LayerNorm
    └─ FFN
        ├─ 门控投影
        ├─ 激活函数
        ├─ 上投影
        └─ 下投影'''
class ModelRunner:
    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)
        # sequence attributes this runner's cache is mapped through; a
        # speculative-decoding draft model tracks its own block table so
        # draft and target caches stay independent
        self.table_attr = config.get('seq_table_attr', 'block_table')
        self.cached_attr = config.get('seq_cached_attr', 'num_cached_tokens')
        # draft runners can capture their sampled-token graph output AND the
        # pre-sampling logits (needed for stochastic rejection sampling)
        self.capture_logits = config.get('capture_logits', False)
        # reusable pinned-CPU + GPU staging buffers: pin_memory allocation
        # costs ~1ms per tensor and the draft hot path runs prepare several
        # times per round — allocate once, copy in place
        self._stage = {}

        self.rank = rank
        # A process may host several runners (e.g. draft + target models for
        # speculative decoding); init the process group only once, and only
        # when it is actually needed for multi-GPU execution.
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                'nccl',
                init_method=f"tcp://localhost:{config.get('dist_port', 12345)}",
                world_size=self.world_size,
                rank=rank,
            )
        elif not dist.is_initialized():
            # single-process, single-GPU: a trivial per-process group so layer
            # helpers (dist.get_rank/get_world_size) keep working; a file
            # store avoids port clashes between independent engines
            import os, tempfile
            store_file = os.path.join(
                tempfile.gettempdir(), f"myvllm_dist_{os.getpid()}.store"
            )
            dist.init_process_group(
                'gloo', store=dist.FileStore(store_file, 1), world_size=1, rank=0
            )
        torch.cuda.set_device(rank)

        dtype_name = config.get("dtype", "bfloat16")
        try:
            self.default_dtype = getattr(torch, dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported dtype: {dtype_name}") from exc
        if self.default_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported model dtype: {self.default_dtype}")
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.default_dtype)

        # set model
        self.model = Qwen3ForCausalLM(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            head_dim=config['head_dim'],
            scale=config['scale'],
            num_kv_heads=config['num_kv_heads'],
            rms_norm_epsilon=config['rms_norm_epsilon'],
            qkv_bias=config['qkv_bias'],
            base=config['base'],
            max_position=config['max_position'],
            intermediate_size=config['intermediate_size'],
            ffn_bias=config['ffn_bias'],
            num_layers=config['num_layers'],
            tie_word_embeddings=config['tie_word_embeddings'],
            block_size=self.block_size,
        )

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # Debug flag for first decode step
        self._first_decode = False

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        '''
        allocated_kv_cache[2][32][1000][16][8][128]
                            │    │    │    │    │     │
                            │    │    │    │    │     └─ 每个头128维 (head_dim)
                            │    │    │    │    └─ 8个KV头 (num_kv_heads)
                            │    │    │    └─ 每个块16个token (block_size)
                            │    │    └─ 1000个块 (num_blocks)
                            │    └─ 32层 (num_layers)
                            └─ K(0)或V(1) (kv_cache_type)    '''
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()
            # speculative-decoding verification forwards are decode-bound
            # too; graph them so a round costs one replay instead of ~28
            # layers of eager Python dispatch
            if self.config.get('capture_verify'):
                import os
                if not os.environ.get('NAIVLLM_NO_VERIFY_GRAPH'):
                    self.capture_verify_graphs()

        torch.set_default_dtype(original_dtype)

        # IMPORTANT: Set up shared memory and barrier AFTER all model initialization
        # This ensures both ranks complete warmup/allocation before rank 1 enters its event loop
        if self.world_size > 1:
            # Synchronize before setting up shared memory
            dist.barrier()
            if self.rank == 0:
                # Try to clean up existing shared memory first
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass  # Doesn't exist, which is fine
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Barrier to ensure rank 1 waits until shared memory is created
                dist.barrier()
            else:
                # Wait for rank 0 to create shared memory
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')
                # Don't call self.loop() here - let the spawning code handle it
                # Otherwise we'll be stuck in an infinite loop during __init__

    # only use read when rank != 0
    def read_shm(self):
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], 'little') # read length
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    # only use write when rank == 0
    def write_shm(self, method_name: str, args: tuple):
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # encode the length first
        # Flatten: (method_name, args) where args is a tuple -> (method_name, *args)
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # close shared memory, destroy process group, delete graphs
    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    # wait to read method and args from shared memory
    # execute the method with args
    # write results back to shared memory
    def loop(self):
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) # Unpack args when calling
            if method_name == 'exit':
                self.exit()
                break

    # will be called by both rank == 0 and rank != 0
    # given method name and args from shared memory
    # execute the method and return results
    def call(self, method_name: str, *args: dict):
        if self.world_size > 1 and self.rank == 0: # will be called in main engine
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    # cleanup memory
    # compute max number of sequence based on max token and max model length
    # run empty sequence to warm up the model
    # clear memory
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max(
            1,
            min(self.config["max_num_sequences"], max_tokens // max_model_length),
        )
        warmup_length = min(max_model_length, max_tokens // batch_size)
        seqs = [Sequence(token_ids=[0] * warmup_length) for _ in range(batch_size)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    # allocate kv cache memory blocks for model
    def allocate_kv_cache(self):
        # find all available memory
        free_mem, total_mem = torch.cuda.mem_get_info()
        # 预留显存和为了安全边际
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        # reserve some room for peak memory usage during model execution
        # 模型推理过程中可能需要的临时内存,必须预留这部分空间，避免运行时OOM
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # find parameters to compute kv cache size
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # check whether the current free memory can hold at least one block
        # compute the actual byte required of each block
        # Query是"当前"的，Key和Value是"历史"的
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        memory_limited_blocks = int(available_mem // block_bytes)
        self.num_available_kv_blocks = min(
            memory_limited_blocks, self.config["max_cached_blocks"]
        )
        assert self.num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'

        # allocate max possible kv cache for the model, instead for each sequence
        # this is the key for paged attention: one giant KV cache pool, divided into blocks
        # IMPORTANT: Use zeros() instead of empty() to avoid garbage values，避免垃圾值影响计算，便于发现未正确初始化的位置
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.num_available_kv_blocks, self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    # given seqs
    # prepare the data needed for a prefill forward pass
    # taking prefix cache into consideration: 
    # input_ids, positions, cu_seqlens_q/k, slot_mapping (where to write new KV values), block_tables (where to read KV values)
    # cu_seqlens_q = [0, 3, 5, 9]
    #               │  │  │  │
    #               │  │  │  └─ end of seq3 (position 9)
    #               │  │  └──── end of seq2 (position 5)
    #               │  └─────── end of seq1 (position 3)
    #               └────────── start (position 0)
    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        # length: sum of all input_ids after prefix cache
        input_ids = []
        # length: sum of all input_ids after prefix cache
        slot_mappings = []
        # length: num_seqs
        seqlens_q = []
        # length: num_seqs
        seqlens_k = []
        # length: num_seqs + 1
        cu_seqlens_q = [0]
        # length: num_seqs + 1
        cu_seqlens_k = [0]
        # block_tables: num_seqs x num_blocks (padded)
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = getattr(seq, self.cached_attr)
            input_ids.extend(token_ids[num_cached_tokens:])
            # K和V总是成对出现，使用相同的缓存结构，不需要分别处理。slot_mapping同时适用于K和V；block_tables同时管理K和V的块
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if getattr(seq, self.table_attr):
                table = getattr(seq, self.table_attr)
                num_cached_blocks = getattr(seq, self.cached_attr) // self.block_size
                for i, block_id in enumerate(table[num_cached_blocks:]):
                    if num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        # 张量的形状必须一致
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # pad block_tables
            all_block_tables = [getattr(seq, self.table_attr) for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                block_table = getattr(seq, self.table_attr) + [-1]*(max_num_blocks - len(getattr(seq, self.table_attr)))
                # 加到全局block_tables列表中
                block_tables.append(block_table)
        # 使用锁页内存，加速 CPU 到 GPU 传输
        # 异步传输，不阻塞 CPU
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        ''' k_v_tensor = [
        [seq1_token0], [seq1_token1], [seq1_token2],
        [seq2_token0], [seq2_token1], [seq2_token2], [seq2_token3], [seq2_token4],
        [seq3_token0], [seq3_token1], [seq3_token2], [seq3_token3]
        ] 
        @triton.jit
        def flash_attention_kernel(Q, K, V, O, cu_seqlens_q_ptr, ...):
            seq_idx = tl.program_id(2)  # 序列索引
            
            # 关键：使用cu_seqlens获取当前序列的边界
            seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)      # 序列起始位置
            seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)    # 序列结束位置
            seq_len = seq_end - seq_start                        # 序列长度
            
            # 计算内存偏移时使用seq_start
            q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + ...
        '''
        
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
            seqlens_q=seqlens_q,
            seqlens_k=seqlens_k,
        )
        return input_ids


    # prepare input data for decoding
    def _stage_1d(self, key: str, values: list, dtype) -> torch.Tensor:
        n = len(values)
        entry = self._stage.get(key)
        if entry is None or entry[0].numel() < n:
            cpu = torch.empty(max(n, 64), dtype=dtype, pin_memory=True)
            gpu = torch.empty(max(n, 64), dtype=dtype, device=f'cuda:{self.rank}')
            self._stage[key] = entry = (cpu, gpu)
        cpu, gpu = entry
        cpu[:n].copy_(torch.as_tensor(values, dtype=dtype))
        gpu[:n].copy_(cpu[:n], non_blocking=True)
        return gpu[:n]

    def _stage_2d(self, key: str, rows: list, dtype) -> torch.Tensor:
        r = len(rows)
        c = max(len(x) for x in rows) if rows else 1
        entry = self._stage.get(key)
        if entry is None or entry[0].shape[0] < r or entry[0].shape[1] < c:
            cpu = torch.empty(max(r, 8), max(c, 64), dtype=dtype, pin_memory=True)
            gpu = torch.empty(cpu.shape, dtype=dtype, device=f'cuda:{self.rank}')
            self._stage[key] = entry = (cpu, gpu)
        cpu, gpu = entry
        cpu[:r, :c].copy_(torch.tensor(rows, dtype=dtype))
        gpu[:r, :c].copy_(cpu[:r, :c], non_blocking=True)
        return gpu[:r, :c]

    def _tokens_to_list(self, gpu_tokens: torch.Tensor) -> list[int]:
        n = gpu_tokens.numel()
        entry = self._stage.get("tok_out")
        if entry is None or entry[0].numel() < n:
            cpu = torch.empty(max(n, 64), dtype=gpu_tokens.dtype, pin_memory=True)
            self._stage["tok_out"] = entry = (cpu, None)
        cpu = entry[0]
        cpu[:n].copy_(gpu_tokens)  # blocking D2H into pinned memory
        return cpu[:n].tolist()

    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        context_lens = []
        slot_mappings = []
        block_tables = []
        for seq in seqs:
            table = getattr(seq, self.table_attr)
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            slot_mappings.append(table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [getattr(seq, self.table_attr) for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            table = getattr(seq, self.table_attr)
            block_table = table + [-1]*(max_num_blocks - len(table))
            block_tables.append(block_table)
        input_ids = self._stage_1d("dec_in", input_ids, torch.long)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=self._stage_1d("dec_slot", slot_mappings, torch.long),
            context_lens=self._stage_1d("dec_ctx", context_lens, torch.long),
            block_tables=self._stage_2d("dec_bt", block_tables, torch.int32) if block_tables else None,
        )
        return input_ids

    # prepare the temperature
    def prepare_sample(self, seqs: list[Sequence]) -> None:
        return self._stage_1d("temps", [seq.temperature for seq in seqs], torch.float32)

    # prepare inputs for a speculative-decoding verification forward.
    # every sequence carries K_i draft tokens appended to token_ids; the
    # forward consumes [last_confirmed, d_1..d_{K_i}] and returns the logits
    # of all K_i+1 rows so the caller can accept/reject each drafted token.
    def prepare_spec_verify(self, seqs: list[Sequence]) -> torch.Tensor:
        input_ids = []
        slot_mappings = []
        positions = []
        seqlens_q = []
        seqlens_k = []
        cu_seqlens_q = [0]
        block_tables = []
        for seq in seqs:
            table = getattr(seq, self.table_attr)
            q_len = len(seq.draft_tokens) + 1
            k_len = len(seq)
            # inputs: positions n-1 .. n+K-1 where n = k_len - num_drafts
            for pos in range(k_len - q_len, k_len):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                block_idx = pos // self.block_size
                slot_mappings.append(table[block_idx] * self.block_size + pos % self.block_size)
            seqlens_q.append(q_len)
            seqlens_k.append(k_len)
            cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
        max_num_blocks = max(len(getattr(seq, self.table_attr)) for seq in seqs)
        for seq in seqs:
            table = getattr(seq, self.table_attr)
            block_tables.append(table + [-1] * (max_num_blocks - len(table)))

        set_context(
            is_prefill=True,   # multi-token forward semantics
            is_verify=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=None,
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(seqlens_k, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            seqlens_q=seqlens_q,
            seqlens_k=seqlens_k,
            positions=torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
        )
        return torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

    # verification forward: returns logits (sum(K_i + 1), vocab) in batch order
    # (impl next to the verify-graph capture at the end of this class)

    # when prefilling, directly compute model forward + logits
    # when decoding, use cuda graph execution to speed up
    # allocate input_ids, positions, slot_mapping, context_lens, block_tables, outputs
    # into graph_variable, and then replay the graph
    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
        temperatures: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if is_prefill or self.enforce_eager:
            # For varlen prefill, keep input_ids as 1D (concatenated tokens)
            # Do NOT unsqueeze - flash_attn_varlen_func expects 1D input with cu_seqlens
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            bs = input_ids.size(0)
            context = get_context()

            # finds smallest captured graph that fits the batch size; keys are
            # captured largest-first, so an unsorted scan would replay an
            # oversized graph whose stale tail rows write garbage KV into
            # slot 0 and corrupt live sequences
            graph = self.graphs[min(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars
            # copy input data into graph variables; anything beyond bs must
            # stay inert (slot -1 skips the KV store for padded rows)
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"].fill_(-1)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            vars["temperatures"][:bs].copy_(temperatures)
            # replay the graph
            graph.replay()
            if self.capture_logits:
                return vars['outputs'][:bs], vars['logits_out'][:bs]
            logits = vars['outputs'][:bs]

        return logits


    # prepare prefill
    # prepare sample
    # run model
    # sample logits
    # reset context
    def run(self, seqs: list[Sequence], is_prefill: bool, temperatures=None):
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        if temperatures is None:
            temperatures = self.prepare_sample(seqs)
        logits_or_tokens = self.run_model(input_ids, is_prefill, temperatures)
        # only sample when rank == 0; tokens must come back as plain CPU ints
        # so scheduler bookkeeping never touches CUDA tensors
        token_ids = None
        if self.rank == 0:
            if isinstance(logits_or_tokens, tuple):
                # (sampled_tokens, captured_logits) from a draft-model graph;
                # clone the logits — the static graph buffer is overwritten
                # by the next draft step, stored views would corrupt p_d
                sampled, logits = logits_or_tokens
                reset_context()
                return self._tokens_to_list(sampled), logits.clone()
            if is_prefill or self.enforce_eager:
                token_ids = self._tokens_to_list(
                    self.sampler(logits_or_tokens, temperatures))
            else:
                token_ids = self._tokens_to_list(logits_or_tokens)
        reset_context()
        return token_ids

    # capture the CUDA graph:
    # pre-allocation at maximum sizes: allocated onece and reuse for all graphs
    # capture for different common batch sizes: [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    # with torch.cuda.graph(graph, self.graph_pool):
    #        run model() and exact sequence of CUDA kernels for running self.model() will be captured
    # (later use graph.replay() to run the captured graph)
    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        max_bs = self.config['max_num_sequences']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        # for decoding, input is always of shape (batch_size, 1)
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # for paged attention
        # where to write new KV values in the cache
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # how many tokens each sequence has processed
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # where to read KV values in the cache
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        temperatures = torch.ones(max_bs, dtype=torch.float32, device=f'cuda:{self.rank}')
        outputs = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # optional: also capture pre-sampling logits (draft models need their
        # distribution for stochastic rejection sampling)
        logits_out = (
            torch.zeros(max_bs, self.config['vocab_size'], dtype=self.default_dtype, device=f'cuda:{self.rank}')
            if self.capture_logits else None
        )

        # graphs to be captured for different batch sizes
        batch_sizes = sorted(
            {size for size in [1, 2, 4, 8, *range(16, max_bs + 1, 16), max_bs] if size <= max_bs}
        )
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )
            hidden_states = self.model(input_ids[:batch_size])
            logits = self.model.compute_logits(hidden_states)
            if self.rank == 0:
                outputs[:batch_size] = self.sampler(logits, temperatures[:batch_size])

            with torch.cuda.graph(graph, pool=graph_pool):
                hidden_states = self.model(input_ids[:batch_size])
                logits = self.model.compute_logits(hidden_states)
                if self.rank == 0:
                    outputs[:batch_size] = self.sampler(logits, temperatures[:batch_size])
                    if logits_out is not None:
                        logits_out[:batch_size] = logits
            if graph_pool is None:
                graph_pool = graph.pool()
            # store the captured graph
            self.graphs[batch_size] = graph

            # make sure that the capture is done before resetting and next capture
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            temperatures=temperatures,
            outputs=outputs,
            logits_out=logits_out,
        )

    # capture verification graphs for speculative decoding: one graph per
    # batch bucket, verifying a fixed number of drafted tokens K per sequence
    # (all rows padded to K+1; pad rows carry position/slot -1 so they are
    # inert). The captured logits land in verify_vars['logits'].
    @torch.inference_mode()
    def capture_verify_graphs(self) -> None:
        max_bs = self.config['max_num_sequences']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)
        k = self.config['num_spec_tokens']
        rows = max_bs * (k + 1)

        v_input_ids = torch.zeros(rows, dtype=torch.long, device=f'cuda:{self.rank}')
        v_slot_mapping = torch.full((rows,), -1, dtype=torch.long, device=f'cuda:{self.rank}')
        v_context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        v_block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # absolute position per row; -1 marks an inert pad row
        v_positions = torch.full((rows,), -1, dtype=torch.long, device=f'cuda:{self.rank}')
        v_logits = torch.zeros(rows, self.config['vocab_size'], dtype=self.default_dtype, device=f'cuda:{self.rank}')
        # debug tap: capture the final hidden states too
        v_hidden_tap = torch.zeros(rows, self.config['hidden_size'], dtype=self.default_dtype, device=f'cuda:{self.rank}')

        self.verify_graphs = {}
        # kernel-argument tensors created per graph (e.g. cu_seqlens) must
        # outlive the capture: the graph records their raw device pointers,
        # and freeing the Python reference would leave the replay reading
        # reclaimed memory
        self._verify_graph_args = []
        # NOTE: each verify graph gets its OWN memory pool. Sharing a pool
        # across the verify graphs (as the decode graphs do) silently broke
        # the first-captured graph's replay on this torch build; verify
        # activations are tiny, so separate pools cost almost nothing.
        batch_sizes = sorted(
            {size for size in [1, 2, 4, 8, *range(16, max_bs + 1, 16), max_bs] if size <= max_bs},
            reverse=True,
        )
        for batch_size in batch_sizes:
            n_rows = batch_size * (k + 1)
            cu = torch.arange(0, n_rows + 1, k + 1, dtype=torch.int32, device=f'cuda:{self.rank}')
            graph = torch.cuda.CUDAGraph()

            def _forward():
                set_context(
                    is_prefill=True,
                    is_verify=True,
                    cu_seqlens_q=cu,
                    slot_mapping=v_slot_mapping[:n_rows],
                    context_lens=v_context_lens[:batch_size],
                    block_tables=v_block_tables[:batch_size],
                    seqlens_q=[k + 1] * batch_size,
                    seqlens_k=[1] * batch_size,
                    positions=v_positions[:n_rows],
                )
                hidden = self.model(v_input_ids[:n_rows])
                logits = self.model.compute_logits(hidden)
                v_hidden_tap[:n_rows] = hidden
                v_logits[:n_rows] = logits

            _forward()  # warmup/compile outside capture
            torch.cuda.synchronize()
            reset_context()
            with torch.cuda.graph(graph):  # own pool (see note above)
                _forward()
            torch.cuda.synchronize()
            reset_context()
            self.verify_graphs[batch_size] = graph
            self._verify_graph_args.append(cu)

        self.verify_vars = dict(
            input_ids=v_input_ids,
            slot_mapping=v_slot_mapping,
            context_lens=v_context_lens,
            block_tables=v_block_tables,
            positions=v_positions,
            logits=v_logits,
            hidden_tap=v_hidden_tap,
            k=k,
        )

    # verification forward through the captured graph when available
    @torch.inference_mode()
    def run_spec_verify(self, seqs: list[Sequence]) -> torch.Tensor:
        if getattr(self, 'verify_graphs', None):
            return self._run_spec_verify_graph(seqs)
        input_ids = self.prepare_spec_verify(seqs)
        hidden_states = self.model(input_ids)
        logits = self.model.compute_logits(hidden_states)
        reset_context()
        return logits

    def _run_spec_verify_graph(self, seqs: list[Sequence]) -> torch.Tensor:
        vars_ = self.verify_vars
        k = vars_['k']
        bs = len(seqs)
        graph = self.verify_graphs[min(b_ for b_ in self.verify_graphs if b_ >= bs)]

        input_ids, slot_mappings, positions, context_lens = [], [], [], []
        max_num_blocks = vars_['block_tables'].size(1)
        for seq in seqs:
            table = getattr(seq, self.table_attr)
            context_lens.append(len(seq))
            n = len(seq) - len(seq.draft_tokens)
            for pos in range(n - 1, len(seq)):
                input_ids.append(seq.token_ids[pos])
                positions.append(pos)
                block_idx = pos // self.block_size
                slot_mappings.append(
                    table[block_idx] * self.block_size + pos % self.block_size
                )
            # pad this sequence's rows up to the captured K+1
            for _ in range(len(seq.draft_tokens) + 1, k + 1):
                input_ids.append(0)
                positions.append(-1)
                slot_mappings.append(-1)

        block_tables = torch.full((bs, max_num_blocks), -1, dtype=torch.int32)
        for i, seq in enumerate(seqs):
            table = getattr(seq, self.table_attr)
            block_tables[i, : len(table)] = torch.as_tensor(table, dtype=torch.int32)

        rows = bs * (k + 1)
        vars_['input_ids'][:rows].copy_(torch.tensor(input_ids, dtype=torch.long))
        vars_['slot_mapping'].fill_(-1)
        vars_['slot_mapping'][:rows].copy_(torch.tensor(slot_mappings, dtype=torch.long))
        vars_['positions'].fill_(-1)
        vars_['positions'][:rows].copy_(torch.tensor(positions, dtype=torch.long))
        # padded batch rows must be inert when an oversized graph is replayed
        vars_['context_lens'].zero_()
        vars_['context_lens'][:bs].copy_(torch.tensor(context_lens, dtype=torch.long))
        vars_['block_tables'].fill_(-1)
        vars_['block_tables'][:bs, : block_tables.size(1)].copy_(block_tables)

        graph.replay()
        return vars_['logits'][:rows]

    # K draft steps fully GPU-resident: the only per-step unknown is the
    # sampled token, which stays on device (graph output -> graph input
    # device copy); positions/slots/contexts are deterministic and staged
    # upfront. Per-step tokens and logits land in device buffers; the host
    # synchronizes exactly once per round.
    @torch.inference_mode()
    def run_draft_steps_gpu(self, seqs: list[Sequence], k: int, temperatures):
        vars_ = self.graph_vars
        bs = len(seqs)
        graph = self.graphs[min(b_ for b_ in self.graphs if b_ >= bs)]

        # deterministic per-step metadata: step i feeds the token at
        # position (n-1+i) with context (n+i)
        ctx_rows, slot_rows, first_inputs = [], [], []
        for seq in seqs:
            first_inputs.append(seq.last_token)
        for i in range(k):
            ctx_rows.append([len(seq) + i + 1 for seq in seqs])
            slots = []
            for seq in seqs:
                table = getattr(seq, self.table_attr)
                pos = len(seq) - 1 + i
                block_idx = pos // self.block_size
                slots.append(table[block_idx] * self.block_size + pos % self.block_size)
            slot_rows.append(slots)

        dev = f'cuda:{self.rank}'
        for key, rows in (("draft_ctx", ctx_rows), ("draft_slot", slot_rows)):
            entry = self._stage.get(key)
            if entry is None or entry[1].shape[0] < k or entry[1].shape[1] < bs:
                cpu = torch.empty(max(k, 8), max(bs, 8), dtype=torch.long, pin_memory=True)
                gpu = torch.empty(cpu.shape, dtype=torch.long, device=dev)
                self._stage[key] = entry = (cpu, gpu)
            entry[0][:k, :bs].copy_(torch.tensor(rows, dtype=torch.long))
            entry[1][:k, :bs].copy_(entry[0][:k, :bs], non_blocking=True)
        ctx_dev = self._stage["draft_ctx"][1][:k, :bs]
        slot_dev = self._stage["draft_slot"][1][:k, :bs]

        if getattr(self, "_draft_logits_buf", None) is None or \
           self._draft_logits_buf.shape[0] < k or self._draft_logits_buf.shape[1] < bs:
            self._draft_logits_buf = torch.empty(
                max(k, 8), max(bs, 8), self.config['vocab_size'],
                dtype=self.default_dtype, device=dev)
        logit_buf = self._draft_logits_buf[:k, :bs]
        token_buf = self._stage_1d("draft_tokens", [0] * (k * bs), torch.long)[:k * bs].view(k, bs)

        tables = [getattr(seq, self.table_attr) for seq in seqs]
        width = max(len(t) for t in tables)
        bt = self._stage_2d(
            "dec_bt", [t + [-1] * (width - len(t)) for t in tables], torch.int32)

        vars_['input_ids'][:bs].copy_(torch.tensor(first_inputs, dtype=torch.long))
        vars_['temperatures'][:bs].copy_(temperatures[:bs])
        vars_['block_tables'].fill_(-1)
        vars_['block_tables'][:bs, :width].copy_(bt)

        for i in range(k):
            vars_['context_lens'][:bs].copy_(ctx_dev[i])
            vars_['slot_mapping'][:bs].copy_(slot_dev[i])
            graph.replay()
            token_buf[i].copy_(vars_['outputs'][:bs])
            logit_buf[i].copy_(vars_['logits_out'][:bs])
            if i + 1 < k:
                # next step's input is this step's sampled token: on device
                vars_['input_ids'][:bs].copy_(vars_['outputs'][:bs])

        tokens_host = token_buf.cpu()  # single blocking transfer for the round
        per_seq = [[int(tokens_host[i, j]) for i in range(k)] for j in range(bs)]
        return per_seq, logit_buf
