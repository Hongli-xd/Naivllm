# NaiVLLM 技术文档

本文面向想读懂这套引擎的工程师，按"数据流顺序 + 亮点深挖"组织：先整体后局部，
核心机制（分页缓存、Triton kernel、CUDA Graph、投机解码）详讲，常规组件（TP、
权重加载）略讲。所有小节都给出对应源码位置。项目定位与性能数字见 `README.md`，
本文不再重复。

```
                          ┌────────────────────────────────────────┐
 prompts ─► LLMEngine ───►│ Scheduler ◄── BlockManager (CPU 侧记账)│
           (tokenizer)    │  waiting/running/抢占                    │
                          └──────┬─────────────────────────────────┘
                                 │ schedule() → (seqs, is_prefill)
                          ┌──────▼─────────────────────────────────┐
                          │ ModelRunner                             │
                          │  prepare_prefill / prepare_decode       │
                          │  run() ── CUDA Graph 回放(decode/verify) │
                          │  ┌──────────────────────────────┐       │
                          │  │ Qwen3: Embedding→28×Layer→LM │       │
                          │  │ 每层 Attention 读写分页 KV 缓存 │       │
                          │  └──────────────────────────────┘       │
                          │  Sampler (贪心/Gumbel)                   │
                          └──────┬─────────────────────────────────┘
                                 │ token ids (CPU int)
                          Scheduler.postprocess → 追加/结束/释放块
```

投机解码在 `SpecLLMEngine` 中替换 decode 步骤：草稿(n-gram 或小模型) →
一次多 token 验证前向 → 拒绝采样 → 回滚，见第 5 节。

---

## 1. 一次请求的生命周期（`engine/llm_engine.py`）

`LLMEngine.generate(prompts)` → tokenize → `Sequence` 对象入
`Scheduler.waiting` → 循环 `step()` 直到全部完成：

- **prefill 阶段**：调度器从 waiting 取序列，`BlockManager.allocate()`
  分配块表（可能命中前缀缓存），`ModelRunner.run(seqs, is_prefill=True)`
  做一次变长前向，采样出首 token；
- **decode 阶段**：每步每序列只喂 1 个 token，走 CUDA Graph 回放；
- **postprocess**（`engine/scheduler.py`）：token 追加进序列；命中
  EOS / `max_tokens` / `max_model_length` 则置 FINISHED 并释放全部块。

`Sequence`（`engine/sequence.py`）是贯穿全链路的状态载体：`token_ids`、
`block_table`（物理块号列表）、投机解码用的 `draft_block_table` /
`draft_tokens`。它实现了 `__getstate__` 只序列化增量——这是为多卡
worker 进程 IPC 设计的。

## 2. Paged KV Cache 与 BlockManager（`engine/block_manager.py`）— 详

### 2.1 显存侧：一个巨大的块池

`ModelRunner.allocate_kv_cache()` 在权重加载后按剩余显存一次性分配：
形状 `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
（2 = K/V）。默认 `block_size=16`。把第 i 层的视图挂到每个 Attention
模块的 `k_cache/v_cache` 属性上——kernel 通过它直接读写。**CPU 侧的
BlockManager 只记账（哪个逻辑块在哪个物理块），不碰显存**。

### 2.2 逻辑侧：块表与前缀缓存

- 每个序列的 `block_table[i]` = 第 i 个逻辑块对应的物理块号；
- **前缀缓存**：以"链式 xxhash"为每个**满块**算内容指纹——
  `hash(block_i) = xxh64(hash(block_{i-1}) || tokens_i)`，父哈希参与
  子哈希，因此指纹天然携带整条前缀的因果性，不同上文下同句不会误命中；
  `hash_to_block_id` 表做查找，命中则 `_acquire_cached_block` 引用计数
  +1 直接复用，KV 无需重算；
- **pin_prefix**：对文档前缀的每个块加 `pin_count`，保证后续序列的
  抢占/释放永远不会把"知识库常驻前缀"挤出去——RAG 场景热前缀永远
  命中；
- **投机解码配套**：`ensure_capacity(seq, n)` 为多 token 前向扩表、
  `trim_to(seq, n)` 把块表裁剪回确认长度并清理半满块的哈希（未确认
  的投机尾巴不能进缓存索引）。

### 2.3 三条不变式（阅读代码时的锚点）

1. 块表中已确认的满块必有哈希；半满尾块哈希必为 -1；
2. `num_cached_tokens` ≤ `num_tokens - 1`（至少留 1 个 token 重算，
   否则 q_len=0 会导致输出错位——这是我们修过的真实 bug）;
3. 物理块的 `ref_count` 归零才可回空闲池；pin 的块永不为 0。

## 3. 调度器（`engine/scheduler.py`）— 中

两级队列 `waiting`/`running`，prefill 优先于 decode（吞吐与延迟的
常规取舍）。值得读的三个细节：

- **预算按新增 token 计**：`can_allocate` 通过后，prefill 预算扣的是
  `len(seq) - num_cached_tokens`——前缀缓存命中的请求几乎不占预算；
- **decode 抢占**：某序列需要新块而空闲块不足时，从 running **队尾**
  （最低优先级）抢占：`deallocate` 全部块、回到 waiting 头部，之后
  重新 prefill（recompute 式抢占，同 vLLM 的最简形式）。注意抢占
  目标可能是它自己——此时本轮放弃调度；
- **顺序恢复**：每步被调度的序列从 running 头部临时取出，步末用
  `extendleft(reversed(...))` 原序放回，保证公平性。

## 4. Attention 三条执行路径 + Triton Kernel（`layers/attention.py`）— 详

`Attention.forward` 根据 `utils/context.py` 的全局 Context 分派——
Context 携带本步批次的元数据（cu_seqlens / slot_mapping / block_tables /
positions…），层内随处可取，避免把参数穿透 28 层。

### 4.1 KV 写入：`store_kvcache_kernel`

grid `(num_tokens, num_kv_heads)`，每个 program 负责一个 token × 一个
KV 头的一行 128 维：从刚算出的 K/V（连续布局）读一行，按
`slot_mapping[t] = 物理块号 × block_size + 块内偏移` scatter 进缓存。
`slot == -1` 直接返回——这是 CUDA Graph 里 padding 行"静默跳过"的开关。

### 4.2 Prefill：变长 FlashAttention（`flash_attention_varlen_kernel`）

批内序列长度不一，用 `cu_seqlens=[0, l1, l1+l2, ...]` 前缀和描述边界。
grid `(cdiv(max_len,BLOCK_M), num_heads, num_seqs)`，program 内做标准
Flash 的 online softmax（`m_i/l_i/acc` 三状态）。因果掩码通过
`offs_m >= offs_n` 一次比较完成。GQA 复用：`kv_head = q_head // (H/Hkv)`。
**注意 prefill 有两条读路径**：无缓存时直接用本步新算的 K/V（Triton
kernel）；有前缀缓存时走 `paged_attention_prefill`（PyTorch 按块 gather，
正确性优先的实现——这是当前明确的性能妥协点）。

### 4.3 Decode：单查询 Paged Attention（`paged_attention_decode_kernel`）

grid `(batch, num_heads)`，每 program 一个 (序列, 头)：按 BLOCK_N 分块
遍历 `context_len` 内的 KV，块号经 block_tables 查物理块，online
softmax 归约。两个曾出过 bug 的细节，值得记住：

- **fp32 累加**：手工点积 `tl.sum(k*q)` 必须先 `.to(tl.float32)`——
  bf16 归约误差 ~1.0，会让平坦 top-2 的 argmax 翻转（`tl.dot` 路径
  自带 fp32 累加，无此问题）；
- -1 哨兵：block_tables 里未用到的列为 -1，加载时 `other=0` + 掩码。

### 4.4 Verify：多查询 Paged Attention（`paged_attention_verify_kernel`）

为投机解码新写：一次前向要同时算 K+1 个查询位置（见第 5 节）。与
decode kernel 的差别只有两点，却是整个投机解码能落地 CUDA Graph 的
前提：

1. **每 program 处理 BLOCK_M(≥16) 个查询行**，行号来自
   `cu_seqlens_q`（各序列行区间），用 `tl.dot(q, kᵀ)` 矩阵式算分数
   （tensor core，fp32 累加）；
2. **逐行因果掩码**：每行有自己的绝对位置 `q_pos`（从 positions 张量
   读），掩码 `q_pos[:,None] >= offs_n[None,:]`——不同查询行看到不同
   长度的历史。`q_pos = -1` 的 padding 行输出 NaN 但被丢弃，且其
   slot=-1 不写 KV，所以对图回放安全。

## 5. 投机解码（`spec_engine/`）— 详（本项目核心增量）

### 5.1 原理一页纸

小成本猜 K 个 token，大模型**一次前向**同时算出这 K 个位置的分布，
逐个与草稿对答案：接受则白赚，第一个被拒的位置用大模型自己的分布
重采一个"补偿 token"（数学上可证明最终分布严格等于大模型分布——
Leviathan et al. 2023）。全对还有第 K+1 个 logits 白送一个 bonus。
每轮净收益 = 接受 token 数 − 1 次验证前向的成本。

### 5.2 两个 Drafter（`spec_engine/drafters.py`）

- **NgramDrafter**：取输出末尾 n∈[2,8] 个 token 作"针"，在
  （prompt+已生成）全历史里找最近一次相同出现，把它后面的 token 抄
  作草稿；长 n 优先。实现上把历史编码为 4 字节大端的字节串，用
  `bytes.rfind`（C 实现）匹配——微秒级，否则 1200 token 上下文的
  Python 循环本身就要几毫秒，吃掉投机收益。无需 draft 模型，
  RAG 抽取式回答实测接受率 63–90%；
- **ModelDrafter**：一个小 Qwen3 跑 K 步自回归 decode。它有**自己的
  BlockManager**（管理 `seq.draft_block_table`）和**自己的 KV 池 +
  CUDA Graph**（`ModelRunner` 通过 `seq_table_attr` 配置参数化，
  两套缓存互不干扰）。

### 5.3 一轮的执行（`spec_engine/spec_llm_engine.py:_speculative_step`）

```
1. 草稿   Ngram: bytes.rfind 查找 / Model: K 次图回放，drafted 追加进 token_ids
2. 扩表   target 侧 ensure_capacity(seq, n+K)   # 覆盖验证要写的位置
3. 验证   run_spec_verify → CUDA Graph 回放     # logits: B×(K+1) 行
4. 采样   RejectionSampler.verify(...)          # 见 5.4
5. 提交   _commit: token_ids = base + 接受前缀(+补偿/bonus)
          block 表 trim_to(确认长度)；draft 侧同步 trim
```

### 5.4 拒绝采样器三种模式（`spec_engine/rejection_sampler.py`）

| 模式 | 条件 | 规则 | 无损性来源 |
|---|---|---|---|
| 贪心 | T=0 | 草稿==目标 argmax 则接受；被拒处取目标 argmax | 目标即 argmax |
| 随机+草稿分布 | Model drafter | u < min(1, p_t(d)/p_d(d))；被拒处从 norm(relu(p_t−p_d)) 重采 | Leviathan 定理 |
| 随机无分布 | Ngram drafter | 每位先从 p_t 抽 x，接受当且仅当 x==d | 输出恒为目标样本 |

贪心模式的输出与引擎自身普通解码**逐 token 一致**——这是
`scripts/check_hf_alignment.py` 的核心断言（对齐测试 PASS 的依据）。

### 5.5 验证为什么必须是 CUDA Graph（性能账）

eager 一次 28 层前向的 Python/launch 开销 ~26ms（GPU kernel 实际只
~5ms）；decode 早已图化，验证若走 eager，投机轮次反而比普通解码慢
（我们实测过 68 vs 301 tok/s 的倒挂）。`capture_verify_graphs()`
按 batch 桶 {1,2,4,8,…} 捕获"整轮验证前向"（模型 + 多查询 kernel +
logits 写回），回放一次 ~几 ms。图模式细节见第 6 节，投机特有的坑：
**每桶的 `cu_seqlens` 是捕获期创建的张量，必须挂在
`self._verify_graph_args` 上保活**——图记录的是裸指针，Python 引用
回收后回放会读到已释放显存（我们靠 hidden-tap 探针定位的这个 bug）。

## 6. CUDA Graph 工程（`engine/model_runner.py`）— 详

### 6.1 通用模式：静态变量 + 拷贝 + 回放

```python
# 捕获期（一次）：input_ids/slot/context_lens/block_tables/temperatures
# 全部是预分配的"静态张量"，kernel 捕获的是它们的地址
with torch.cuda.graph(graph, pool=graph_pool):
    outputs = sampler(model(input_ids))          # 采样也进图！

# 每步（重复）：把真实数据拷进静态张量 → 一次回放
vars['input_ids'][:bs].copy_(...) ; vars['slot_mapping'].fill_(-1) ; ...
graph.replay()
```

decode 图按桶 {1,2,4,8,16…max_bs} 捕获、共享内存池、**从大到小**捕获
（先定池的尺寸上界）；verify 图独立成池（详见 5.5 与 6.2）。

### 6.2 这个项目里踩过并修掉的四个图陷阱（很值得读）

1. **过大图回放**：桶查找用 `next()` 按插入序取到更大的图，未填充的
   行带着捕获期残留值参与计算；
2. **尾部 slot 污染**：`slot_mapping[:bs].fill_(-1)` 只清了前 bs 行，
   尾部残留 0——padding 行会把垃圾 KV 写进物理块 0，正好砸中序列
   自己的第 0 块 → 之后每步 argmax(NaN)=0，表现为"永远输出 token 0"；
   修法：**整张量** fill(-1) 再拷贝前 bs 行；
3. **悬垂参数**：捕获循环里创建的张量被 Python 回收后图仍在读它的
   显存地址（完整排查故事见 `INTERVIEW.md` D2，此处只记结论：所有
   捕获期张量必须显式保活，本仓库挂在 `_verify_graph_args`）；
4. **采样进图**：`torch.where(greedy, argmax, gumbel)` 全部用张量
   运算表达，贪心/温度两种路径都能被捕获，图外不做任何 Python 分支。

### 6.3 图外的确定性

`run()` 返回前把 GPU 采样结果 `.tolist()` 同步回 CPU——CPU 侧的
调度/记账永不持有 CUDA 张量（原版在这里把 GPU 标量塞进
`token_ids`，到块边界算哈希时崩溃）。

## 7. 模型与数值（`models/qwen3.py`, `layers/`, `utils/loader.py`）— 中

- Qwen3 结构照 HF 实现：QKV 合并投影 → **q/k 各自 RMSNorm（无条件）**
  → RoPE（半分式旋转，cos/sin 表缓存）→ 注意力 → 门控 MLP。残差用
  "融合 RMSNorm(x, residual)" 形式省一次读写；
- **数值对齐**：与 HF 贪心解码比，argmax 一致、logit 差 ~0.3-0.7
  （Triton flash 与 SDPA 的 bf16 舍入不同所致）；平坦 top-2（gap<0.1）
  上续写会分叉，这是自研 kernel 的正常水位，基准中所有后端同样面对；
- **权重加载**：合并 q/k/v 与 gate/up 到打包投影；分片逻辑收敛到每个
  Parameter 的 `weight_loader(param, shard, key)`（Column 切 0 维、
  Row 切 1 维、Vocab 切词表维），TP>1 时 loader 自动取本 rank 片段。

## 8. Tensor Parallel — 略

QKV 用 ColumnParallel（切输出维），attention 各卡独立算自己的头，
o_proj/down_proj 用 RowParallel + NCCL all_reduce（bias 只在 rank 0
加）。单卡场景不启用；与投机解码共用时限制 world_size=1。

## 9. 基准方法学（`benchmarks/rag_bench.py`）— 中

公平性设计：同权重（bf16/fp16）同 token 序列同 GPU 同贪心；vLLM 配
**完全对等**的 ngram 投机参数；Ollama 起专用实例绑定同一块 GPU
（fp16 + raw 模式防止双重 chat template + 受限 KV 池）；进程隔离、
3 次取中位、warm 前缀后计时。产出 JSON → `plot_results.py` 出图。

## 10. 目录速查

| 路径 | 内容 | 本文小节 |
|---|---|---|
| `engine/block_manager.py` | 分页缓存记账/前缀缓存/pin/trim | 2 |
| `engine/scheduler.py` | 队列/抢占/预算 | 3 |
| `engine/model_runner.py` | 数据准备/两张图/投机验证 | 4,6 |
| `layers/attention.py` | 四个 Triton kernel | 4 |
| `layers/sampler.py` | 贪心+Gumbel 采样 | 6.2 |
| `spec_engine/` | 引擎/drafter/拒绝采样 | 5 |
| `scripts/` | HF 对齐 + 数值/图排障探针 | 7,6.2 |
| `benchmarks/rag_bench.py` | 三方基准 | 9 |
