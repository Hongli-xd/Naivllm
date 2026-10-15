# NaiVLLM 面试问题手册

面向大模型推理优化 / Infra 岗位，按"考官视角"组织：每题给出**考点、
答题要点、常见追问链**。机制细节的完整解释在 `TECHNICAL.md` 对应
小节，本文不重复展开，只给"面试时要说出口的话"。按重要性排序：
★★★ = 本项目最有区分度、必被追问；★ = 常识核对。

---

## A. 项目动机与取舍（几乎必问）

**A1 ★★ 为什么自研引擎而不直接用 vLLM？垂直化怎么选的？**
要点：不做 vLLM 全功能替代；选 RAG 知识库问答这个单一负载——长共享
前缀 + 抽取式短回答 + 单卡小 batch，三项特征分别对应我们三个优化
（pinned prefix / ngram 投机 / 全链路 CUDA Graph）；用可复现的公平
基准定义"赢在哪、输在哪"。追问"为什么不选代码补全/Agent 流"：答
同构（都是"输出复述上下文"），但 RAG 的需求面最大、前缀共享最明显。

**A2 ★★★ 你的数字里哪些是赢的、哪些是输的？为什么输？**
要点（必须诚实背熟）：赢 Ollama 4.8–7×（它无 ngram 投机、无前缀
钉驻）；batch=1 与 vLLM 持平（452/441/463 tok/s 同噪声带内）；
冷启动 3.1s vs 44.9s（14.5×，scale-to-zero 场景关键）；**batch≥2
吞吐输 vLLM 1.4–2 倍**——vLLM V1 的异步调度 + 更成熟的批处理把 GPU
喂得更满，我们单线程 Python 调度每步开销固定，batch 大了被摊不平。
追问"怎么追"：调度器异步化（CPU 与 GPU 流水）、批内投机长度自适应。

**A3 ★ 3.5k 行能叫生产引擎吗？**
要点：定位是"可读的垂直引擎 + 实验平台"；诚实列 unsupported：
仅 Qwen3、前缀复用 prefill 走慢路径、无量化/前缀 disk cache。

## B. Paged KV Cache（基础功）

**B1 ★★★ 画一下 Paged KV Cache：逻辑块/物理块/块表的关系？解决什么问题？**
要点：KV 按固定 block_size(16) 分页；序列持逻辑→物理映射；外部碎片
消除、内部碎片 ≤1 块、前缀共享天然可行。对照 vLLM 论文的 block
table，我们能落到代码：`BlockManager.blocks[]` + `seq.block_table`。

**B2 ★★★ 前缀缓存怎么判断"可复用"？哈希为什么是链式的？**
要点：满块内容做 xxhash，且子块哈希混入父块哈希（`compute_hash`）
——指纹携带整条前缀因果，避免"同句不同上文"误命中；查表后还要
逐 token 比对防哈希碰撞（`token_ids` 复核）；半满块/被 trim 的块
哈希必须清成 -1（不变式见 TECHNICAL 2.3）。
追问"被抢占的缓存块数据还在吗"：在——空闲池块不清零，只要没被
重新写入就仍可"冷命中"，靠 token 复核保证安全。

**B3 ★★ pin_prefix 和普通 prefix caching 的差别？什么时候用？**
要点：普通缓存块 ref=0 即可被抢占回收，热前缀可能被冷请求挤掉；
pin 给文档前缀加 pin_count 保驻，RAG 换问题不换文档时 TTFT 稳定。

**B4 ★★ 抢占怎么做？recompute 与 swap 的取舍？**
要点：本引擎 recompute 式——释放全部块回 waiting，重新 prefill；
省显存实现简单，代价是重算；vLLM 还有 swap/标准实现分层。我们只从
running 队尾抢（最低优先级），抢的目标可能是申请者自己。

## C. 投机解码（本项目最核心，必深挖）

**C1 ★★★ 白板讲一遍投机解码的完整算法与无损性证明。**
要点：draft K 个 → target 一次前向出 K 个分布 → 逐位
`u < min(1, p_t(d)/p_d(d))` 接受 → 首拒位从 norm(relu(p_t−p_d)) 重采
→ 全对则第 K+1 个 logits 白送 bonus。证明思路：每步输出的分布 =
Σ_接受前缀 p_前缀 × [接受继续 + 拒绝残差]，代数展开恰等于 p_t
（能写出 p_t = min(p_t,p_d) + (p_t−p_d)^+ 这一步就赢了）。

**C2 ★★★ 你有草稿分布吗？n-gram drafter 没有 p_d 怎么保证无损？**
要点（答不出这题等于白做）：无 p_d 时不能做比值检验；改用"先从
p_t 抽 x，接受当且仅当 x==d_i"——输出要么是 d（其出现概率 ≥p_t(d)，
差额由后续 mismatch 分支补齐），要么直接是 x（目标样本），可证明
联合分布严格等于 p_t；工程上贪心模式退化为"草稿==argmax 才收"。
我们的验证：贪心 spec 输出与自身 plain 解码**逐 token 一致**。

**C3 ★★★ 验证前向的输入为什么是 K+1 个 token 而不是 K？**
要点：输入 `[x_n, d_1..d_{K-1}]` 的 logits 只覆盖验证 d_1..d_K；多喂
一个 d_K 才拿到第 K+1 行 logits 做 bonus——全接受时白赚一个 token，
这是吞吐上限从 K 提到 K+1 的关键。KV 写入覆盖 n−1..n+K−1 位置，被
拒后 trim，多余的 KV 会被后续轮次覆盖，不影响正确性。

**C4 ★★ 草稿被拒后 KV Cache 怎么办？**
要点：token_ids 截回 base + 接受前缀 + 补偿 token；target 侧
`trim_to`（块表裁剪 + 半满块哈希清理），model drafter 还有独立 draft
侧表同步 trim；块内"多写的脏 KV"无害——后续轮次在该位置覆盖写，
attention 长度由确认后的 context_len 决定不会读到。

**C5 ★★ 投机解码的加速上界和下界？什么时候反而亏？**
要点：上界 (K+1)/1（每轮全收）；下界 1/(K+1)（全拒，还多花了草稿
成本）；实际 = 接受 token 数 / (验证成本 + 草稿成本)。n-gram 草稿
成本≈0，接受率 63–90% 时稳赚；model drafter 若草稿模型与目标同量级
必亏（我们因此主推 ngram）。batch 变大：验证前向变宽、GPU 本来就
喂得满，收益摊薄——实测 batch=8 时 plain≈spec。

**C6 ★★ K 怎么选？batch 内序列接受长度不一怎么办？**
要点：K 是接受率与验证宽度的折中，我们基准扫了 K=5/8；当前实现
图按固定 K 桶 padding（-1 哨兵行静默跳过）；改进方向是 vLLM 式
动态/spec-num-tokens 调度。

**C7 ★ 你的多查询 verify kernel 和普通 decode kernel 差在哪？**
要点：BLOCK_M 个查询行共享同一份 KV 加载（`tl.dot(q,kᵀ)` 一次算
整列分数，tensor core + fp32 累加）；逐行因果掩码用各自绝对位置
`q_pos >= offs_n`；padding 行 q_pos=-1 输出 NaN 但被丢弃。

## D. CUDA Graph（第二核心）

**D1 ★★★ 讲讲你为什么需要 CUDA Graph，以及捕获/回放的机制。**
要点：小 batch decode 是 launch-bound——28 层 × 每层几十个 kernel，
eager 一步 ~26ms 而 GPU 实算 ~5ms；把"静态形状的一步"录成图，每步
只做"数据拷进静态张量 + graph.replay()"，一次提交整个 kernel 序列。
捕获要求：所有输入是预分配静态张量的**原地更新**，控制流在图内
必须无 Python 分支（我们的贪心/温度采样用 `torch.where` 张量化
就是为了进图）。

**D2 ★★★ 你们踩过的 CUDA Graph 坑，挑一个讲透。**
首选讲**悬垂指针**：按桶捕获时循环内创建的 cu_seqlens 张量被 Python
GC 回收，图记录的是裸地址，回放读已释放显存 → logits 全 NaN 且无
任何报错。定位：给图内加 hidden-tap（把中间 hidden 拷到静态缓冲），
NaN 在第一层就出现 → 排除数据准备 → 锁定图参数生命周期。修复：
`_verify_graph_args` 显式保活。备用坑：padding 行 slot 残留 0 污染
物理块 0（整表 fill(-1)）、`next()` 选到过大图（改 `min()`）。
这题展示的是**无报错故障的排查方法论**：二分（eager 对照）→ 打点
（tap）→ 假设验证。

**D3 ★★ 图怎么分桶？为什么不捕获所有 batch？**
要点：{1,2,4,8,16,…max_bs} 桶，bs 向上取整 padding；桶太多捕获慢
且费显存。从大到小捕获共享池（池尺寸由最大图定）；verify 图独立
成池（共享池时首图回放异常，独立池代价仅几 MB）。

**D4 ★ 采样放进图里有什么坑？**
要点：RNG——PyTorch 对图内随机数有 Philox 偏移机制，`exponential_`
可捕获；温度=0 的除零要在图外用 `torch.where` 换成安全值。

## E. Triton Kernel 与性能

**E1 ★★★ decode kernel 的精度坑：为什么 bf16 累加会翻转 argmax？**
要点：手工 `tl.sum(k*q)` 按输入精度归约，128 维 bf16 累加误差可到
~1.0；logit top-2 间隔常 <0.1（尤其小模型），噪声即翻车。修法：两
操作数 `.to(tl.float32)` 再乘加；对照 `tl.dot` 路径自带 fp32 累加。
定位手段可讲：构造 A/B（全量 prefill vs 缓存 decode）对比 logits，
max|diff|=1.05 vs 噪声基线 0.26 一眼异常。

**E2 ★★ online softmax 为什么必须用？写得出状态更新公式吗？**
要点：分块遍历 KV 不能两遍扫（scores 放不下显存）；维护
`m_i(块内最大)/l_i(归一化和)/acc`，新块来时 `α=exp(m_old−m_new)`，
`acc←α·acc+P·V`，`l←α·l+ΣP`。追问数值稳定性：减去行最大 m 防
exp 上溢。

**E3 ★★ GQA 在 kernel 里怎么落？**
要点：`kv_head = q_head // (num_heads/num_kv_heads)`；TP 下用每卡
本地头数计算比值；K/V 缓存只存 KV 头维度（省 (H/Hkv) 倍）。

**E4 ★ 你的 kernel 和 FlashAttention-2 / FlashInfer 的差距？**
要点：诚实——无 softmax 缩放融合优化、无 tile 级 pipelining 调优、
prefill 有缓存时走 PyTorch gather 慢路径；我们的价值在"可读 + 专
门的多查询 verify 形态"。加分：说出 FlashDecoding 的 split-KV 思路
（长上下文时按 KV 列切分再归约）。

## F. 调度与批处理

**F1 ★★ prefill 优先还是 decode 优先？TTFT 与 TPOT 怎么权衡？**
要点：我们 prefill 优先（新请求尽快出首 token）；代价是打断 decode
批的节奏。vLLM V1 有 chunked prefill 把长 prefill 切片穿插——我们
未实现，是 batch≥2 输给它的原因之一。

**F2 ★★ 你们的 prefill 预算怎么算？和 vLLM 有何不同？**
要点：按"新增 token"计费（len − cached），前缀命中的请求几乎零预算
占用；上限 = max_num_batched_tokens（一个调度步的 token 预算）。

**F3 ★ 连续批处理和 static batching 的区别一句话？**
要点：每步动态重组批（有人结束就走、有人来了就进），不需要等整批
齐/整批完；工程代价是每步重算 cu_seqlens/块表 padding。

## G. 数值与分布式

**G1 ★★ bf16 训推的数值水位？你的 HF 对齐结论？**
要点：与 HF 贪心比 argmax 一致、logit max|diff| 0.3–0.7（kernel
求和顺序不同的舍入）；平坦 top-2 上续写分叉是自研 kernel 正常水
位，投机无损性以"同引擎 spec vs plain 逐 token 一致"为准而非对 HF。

**G2 ★ TP 下 RowParallel 的 bias 有什么坑？**
要点：partial sum all_reduce 后 bias 被加了 world_size 次；只允许
rank 0 加。Qwen3 这些层无 bias，但代码修对了（潜在 bug 也要修）。

## H. 系统设计开放题（考迁移能力）

**H1 ★★★ 让你把这个引擎改造成 100 并发的在线服务，动哪几刀？**
答题框架（按收益排序）：① 调度器异步化，CPU 准备与 GPU 回放流水
（参考 vLLM V1 的 overlap scheduler）；② ITI-chunked prefill +
token-budget 调度防止长文档 prefill 饿死 decode；③ spec 桶按批内
最小接受长度自适应 K；④ 前缀 LRU + pin 混合淘汰策略；⑤ 多卡：
TP 已有，加 pipeline 并行不值得（模型小），数据并行多实例更优。

**H2 ★★ draft 模型投机解码（0.6B→1.7B）怎么做？瓶颈？**
要点：已有 ModelDrafter 全套（独立 KV/图/拒绝采样）；瓶颈在 draft
的 K 次串行图回放与 target 验证的串行和，改进：draft 也做投机/一次
前向多序列、或 Medusa/EAGLE 型单次多头草稿；提到我们 verify 图的
padding 方案可复用。

**H3 ★ 长上下文（>32k）会先在哪里崩？**
要点：RoPE cos/sin 表按 max_position 预算（qwen3 40k 可调）；块表
列数 = ceil(len/16) 膨胀（int32 表显存）；decode kernel 单 program
串行扫全 KV——需要 FlashDecoding split-KV；调度预算按 token 算仍成立。

**H4 ★ 为什么没做量化？加上要注意什么？**
要点：垂直场景瓶颈在 launch 开销不在算力，量化收益小；要加的话
KV Cache 量化（fp8）优先于权重（KV 是显存大头且影响批容量），
注意 decode kernel 的加载类型与 online softmax 精度、以及 lm_head
的采样对 logit 精度敏感（呼应 E1 的 bf16 翻转教训）。

## I. 反问与表达建议

- 主动引导：把话题引向 C1/C2（无损性）和 D2（悬垂指针排查）——
  这是项目里最"值钱"的两段；
- 每个数字都能说出测量条件（模型/卡/batch/贪心/3 次中位），被追问
  波动时坦承共享 GPU 的 ±15% 噪声与最终背靠背复测；
- 不贬低 vLLM/Ollama：强调"垂直负载 + 明确交叉点"的工程判断。

---

### 覆盖对照（自查表）

| 岗位 JD 关键词 | 对应题 |
|---|---|
| vLLM/SGLang 原理 | B1–B4, F1–F3 |
| 投机解码/草稿模型 | C1–C7 |
| CUDA Graph/Kernel 开发 | D1–D4, E1–E4 |
| 性能调优/Profiling | E1, D2, A2 |
| 分布式推理 | G2, H1 |
| 论文到工程落地 | C1–C3, 5.5 性能账 |
