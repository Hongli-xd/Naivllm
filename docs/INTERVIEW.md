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

**A4 ★★ 你的三方基准怎么保证公平？踩过哪些坑？**
考点：benchmark 方法学（简历第 3 条的直接考题）。要点：五同——
同权重精度（bf16/fp16）、同 token 序列（绕过各自 tokenizer 差异）、
同 GPU（Ollama 起专用 GPU 绑定实例，避免默认 Q4 量化与 CPU 回退）、
同解码（贪心，投机解码无损性下结果可比）、进程隔离 + 3 次取中位；
vLLM 配置**完全对等**的投机参数（K 与 n-gram 上下界一致）。能讲出
的真实坑：Ollama 默认再套一层 chat template 导致双重模板（要用
raw 模式）；共享 GPU 邻居负载造成 ±15% 波动（终测背靠背重跑）；
离线吞吐≠在线 goodput（见 L2）。追问"为什么不报 TTFT"：vLLM 离线
API 不暴露每请求首 token 时刻，宁缺毋假。

**A5 ★★ 统一采样器怎么设计的？为什么用 Gumbel-max？**
考点：简历"贪心与温度采样统一采样器"一句。要点：采样路径 =
`(probs / exponential噪声).argmax`——Gumbel-max 技巧，等价于从
categorical 分布采样，但**全程是张量运算**，无 `torch.multinomial`
的宿主同步语义，因此能直接进 CUDA Graph；贪心分支用
`torch.where(greedy, argmax, gumbel)` 在图内切换，温度=0 时先
`where` 成 1.0 防除零 NaN（E1 的老教训）。追问"Gumbel-max 为什么
对"：P(argmax(log p + G)=i)=p_i 是标准结论，工程优点是 kernel
友好（一次 elementwise + 一次 argmax，无需 scan）。

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
批的节奏。vLLM V1 用 chunked prefill 把两者混进统一 token 预算
（机制细节见 I4）——我们未实现，是 batch≥2 输给它的原因之一。

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

---

## I. vLLM 原理机制（对标产品必考）

> 简历里所有数字都以 vLLM 为参照系，考官必然检验你对它的理解
> 深度。答题技巧：每个机制先答 vLLM 的做法，再一句"我们引擎对
> 应的位置"，把知识变成对照而非背诵。

**I1 ★★★ vLLM PagedAttention 的核心机制？和操作系统分页的对应关系？**
要点：块表≈页表（逻辑块→物理块）、BlockManager≈分配器、抢占
≈swap out、前缀共享≈共享内存。论文三收益：消除外部碎片（KV 的
显存利用率 ~20%→90%+）、内部碎片 ≤1 块、并行采样分支可共享前缀。
我们对照：同样的三收益，但**只共享不可变的满块**——追加永远进新
块，从设计上回避了 vLLM 论文里的 copy-on-write 问题（见 I2）。

**I2 ★★ vLLM 为什么要 copy-on-write？你的实现怎么避开它？**
要点：vLLM 允许序列 fork（beam search / parallel sampling）共享
"还在增长的尾块"，两个序列都要往同一物理块写 token → 引用计数 >1
时谁写谁先复制整块。我们只缓存/共享**内容封死的满块**（哈希注册
过、永不再写），尾块永远是独占的新分配——用一点共享粒度换掉整个
CoW 路径；代价是"完全相同的两个活跃前缀"不能合并尾块，RAG 场景
无感。

**I3 ★★★ vLLM 的 prefix caching 怎么实现？演进过什么方案？**
要点：v0——满块内容哈希 + 引用计数（与我们同构）；V1——radix
tree（radix cache）：按 token 前缀建树，命中即沿树拿全部块，叶节点
LRU 淘汰，天然支持任意长度部分命中与淘汰顺序；默认开启。我们
对照：块哈希链等价于隐式 radix（父哈希=树边），但没有 LRU——
块释放即回池，靠 pin_prefix 显式保驻热前缀；改进方向就是 LRU 或
radix 化（诚实承认差距）。

**I4 ★★ vLLM 调度器的一步做什么？抢占的两种模式？**
要点：每步：调度批次（受 max_num_seqs / max_num_batched_tokens
双重限制）→ 前向 → 采样 → 输出处理；FCFS，prefill 优先（V1 用
chunked prefill 把长 prefill 切进统一 token 预算，prefill/decode
混批，防长请求饿死 decode——这是它 batch≥2 赢我们的结构性原因，
呼应 A2/K6）。抢占：recompute（释放全部块，回 waiting 重算，V1
默认，便宜简单）与 swap（KV 换出 CPU，换回，省重算费显存带宽）；
我们只实现 recompute（从队尾抢）。

**I5 ★★★ vLLM 的 CUDA Graph 怎么做？为什么是"piecewise"？**
要点：捕获的是**decoder 层堆叠**这一段形状固定的计算，attention
与采样留在图外——因为 attention 的块表/序列长度每步都变，自定义
kernel 难以静态化；输入按 batch 桶 padding，多个图共享显存池。
对照：我们把 attention（paged decode / 多查询 verify kernel）也
**放进了图**——块表/长度做成静态张量原地更新即可，连采样都用
Gumbel-max 图内化（D1/D4）；代价是 padding 行要 -1 哨兵防 KV
污染（D2 的坑 2）。这段"敢把 attention 进图"是差异化的好故事。

**I6 ★★ vLLM 的投机解码怎么组织？与我们差异？**
要点：组件化——proposer（ngram = prompt-lookup 式 top-k 匹配 /
draft model / EAGLE）+ verifier（target 一次多 token 前向）+ GPU
上批量 rejection sampler；V1 里投机批与普通批统一调度（spec-decode
worker 不再独立进程）。ngram 的 proposer 用最近 k token 在
prompt+output 里做多候选匹配（我们的单候选 + 右到左首个命中是
简化版）。追问"为什么我们单候选也够"：RAG 抽取式输出的复述比例
高，单链接受率已 63–90%，多候选树的收益要抵消验证变宽的成本。

**I7 ★ vLLM V1 相比 V0 的架构演进？**
要点（一句话每条）：async 前端（API server 与 engine core 分进程
解耦）、overlap scheduler（CPU 调度与 GPU 执行错步流水，K6）、
prefix caching 默认开 + radix、chunked prefill 默认开、piecewise
CUDA graph、多模态拆到 V2。要点不在背清单，而在把每条跟"我们
哪里对应/哪里没做"挂钩——这正是 A2/H1/K6 的答题素材。

**I8 ★ vLLM 为什么快？一句话总结它的性能观。**
要点：把 LLM 推理当作**显存布局与调度问题**而非算子问题——
PagedAttention 解决"放得下且共享"，continuous batching + chunked
prefill 解决"GPU 一直有活干"，CUDA graph 解决"派活的开销"。
我们全盘接受了这个框架，再在垂直负载上加投机解码与全图化验证。

## J. 超大规模集群调度（PD 分离 / 全局缓存 / 多租户）— 纵深章

> 与 H 的分工：H 是"单引擎在线化"；本章是**多实例/多机/集群**。
> 这是 infra 高级岗的区分题，答得出框架 + 权衡即领先。

**J1 ★★★ 千卡集群上，prefill 和 decode 为什么、怎么分开调度？**
考点：PD 分离（Disaggregated Serving，DistServe/Mooncake 路线）。
要点：①动机——两者资源形态相反：prefill 是 compute-bound（大 GEMM，
吃 MFU），decode 是 memory-bound（小 GEMM + 大 KV 读，吃带宽/批容量），
混部时互相干扰：长 prefill 卡住同批 decode 的 TPOT、decode 大批又让
prefill 的算力闲置；②架构——P/D 两组实例 + 请求路由层，prefill 完
成后把 KV 传给 decode 组；③收益——各自独立扩缩容、独立批策略、
SLO（TTFT 由 P 组保、TPOT 由 D 组保）解耦；④代价——KV 跨实例传
输（见 J3）与调度复杂度；⑤过渡形态——单机内 chunked prefill
（vLLM V1）是"部分分离"，先答它再升级到跨机分离，显得有层次。
追问"什么规模才值得 PD 分离"：KV 传输时间 < 干扰损失时才赚，小
模型/单卡不值（我们引擎正是后者，如实说）。

**J2 ★★★ 十万级并发的请求怎么路由到"对的"推理实例？**
考点：prefix-aware / cache-aware routing。要点：①朴素 LB（round-
robin/最少连接）会打散前缀复用——同一文档的请求分到无缓存的实例，
TTFT 翻倍；②做法：对 prompt 前缀块哈希（我们引擎的 xxhash 链式
指纹可直接复用为路由 key），一致性哈希把"同前缀请求"粘到同实例；
③SGLang 式 cache-aware LB：实例回报 radix-tree 命中率，路由器按
"预计命中长度"加权打分，兼顾负载均衡与命中率的权衡；④全局索引
的代价：路由表本身是一致性难题（实例下线/淘汰），Mooncake 用
中心 KV pool + 元数据服务解。追问"缓存抖动怎么办"：二级策略
（命中优先但过载则溢出到冷实例）。

**J3 ★★ KV Cache 跨实例传输，具体传什么、多大、怎么不挡路？**
要点：①只传本次会话的 KV（如 8k token × 每层），量级 =
2×layers×kv_heads×head_dim×2B×len（Qwen3-8B、8k 上下文 ≈ 1.2 GB）；
②与 prefill 流水——逐层传（layer i 算完即发），RDMA/NCCL P2P，
与剩余层的计算 overlap；
③decode 侧"先到先起跑"或等待齐全的取舍；④追问：为什么不重算？
——重算成本 = prefill FLOPs，通常高于一次网络传输（尤其 RDMA），
但跨机房就反转。能把我们"块 16 token 粒度"的 KV 布局说成"传输
天然分片单元"是加分项。

**J4 ★★ 怎么用排队论做容量规划与 SLO 保障？**
要点：①指标体系：TTFT P99 / TPOT / goodput（满足 SLO 的有效吞吐，
vLLM 对外常用）；②Little's law：在途请求数 = 到达率 × 停留时间，
反推并发上限与实例数；③decode 是准批服务（batch 周期 ~十 ms 级），
可用 M/G/c 近似估排队延迟，P 实例用 GPU 时间片估；④闭环：在线
扩缩容按队列长度 + SLO 违约率反馈。诚实说我们基准只有离线吞吐，
在线 SLO 是路线图——但能说出体系本身就过这题。

**J5 ★★ 多租户共享 GPU 怎么隔离？互相干扰哪个最疼？**
要点：①选项——MIG（硬件级切分，显存/SM 隔离最硬但碎片化）、
time-slicing（简单，尾延迟抖动大）、batch 内混跑（vLLM 多 LoRA
式，利用率最高但干扰最强）；②干扰源——decode 大批抢占带宽/
L2、prefill 的 SM 占满让邻居 decode 的 kernel 排队；③实践中
"关键租户独占 + 弹性租户混跑"分级。呼应：我们单引擎单租户，
TPOT 稳定性靠 CUDA Graph 的确定性执行时间保证。

**J6 ★ 弹性伸缩的冷启动问题怎么解？（我们能抢答）**
要点：scale-to-zero 后重启的代价 = 权重加载 + 图捕获 + KV 预热；
我们的数字：3.1s vs vLLM 44.9s——因为图捕获少（桶少）且无编译
栈；进一步方向：Mooncake 式 KV 持久化池（重启不丢前缀）、权重
mmap 常驻 page cache、模型预加载 sidecar。

**J7 ★ 序列热迁移 / 故障恢复怎么做？**
要点：序列状态 = token_ids + block_table + KV 数据；迁移 = 传 KV
（同 J3）+ 状态快照；故障恢复的两级——请求级重放（幂等，代价小）
vs 检查点续推（省算但工程重）；诚实：我们引擎未做，但 trim/rollback
机制说明"序列状态可重建"这一前提已经具备。

## K. 前沿架构与论文（考知识面与追踪能力）

> 与 C 的分工：C 讲我们实现的投机解码；K2 只做**横向对比**。

**K1 ★★★ MLA（DeepSeek）与 GQA/MQA 的本质区别？对推理栈的影响？**
要点：①MQA/GQA 压缩 KV 的"头"维（H/Hkv 倍），MLA 把 KV 投影到
低秩潜在空间（kv_lora_rank），K/V 不单独存、attention 时从 latent
升维；②收益——KV 再降一个量级（DeepSeek-V3 单 token KV 70KB 级），
decode 批容量大增；③代价——与 RoPE 的位置耦合要拆分（nope/rope
两部分），FlashAttention 形态要改（重量化 FA 的 support 是热点），
paged cache 存的是 latent 块；④答出"MLA 本质 = attention 版 LoRA
式低秩分解"这句就到位。

**K2 ★★★ 四代投机解码横向对比：n-gram / draft-model / Medusa / EAGLE / MTP**
要点：①n-gram：零模型成本，上限受"输出复述上下文比例"钳制
（我们实测 63–90% 就是这个比例）；②draft-model（我们已实现）：
通用但串行 K 次前向 + 双模型显存；③Medusa：目标模型加多个解码头
一次前出树，省串行但头要训练；④EAGLE：在特征层自回归（低维
hidden 而非 token），接受率最高，DeepSeek-V3 的 MTP 是其工程化
（训练时就带下一 token 预测头）；⑤选型结论：RAG 类抽取负载
n-gram 性价比无敌；通用 chat 才值得 EAGLE/MTP。
追问"树投机 vs 链投机"：树提高每轮期望接受数，代价验证 kernel
要支持树形因果掩码（我们 verify kernel 的逐行掩码是它的退化
直线形态）。

**K3 ★★ MoE 推理的通信瓶颈与 DeepEP？**
要点：①MoE 每 token 按路由发到专家——all-to-all 代替 allreduce
成为主通信；②EP（专家并行）下 all-to-all 通信量 ∝ 激活 token 数，
与 TP 的 per-layer 固定开销不同；③DeepEP 的两个 kernel：normal
（大 batch 高效 all-to-all）与 low-latency（小 batch 纯 RDMA），
对应 prefill/decode 两种形态——正好接 J1 的 PD 分离逻辑；④单机
MoE 也有专家装不下/负载倾斜问题（expert placement）。

**K4 ★★ KV 稀疏化/淘汰（H2O、SnapKV）与 paged cache 的关系？**
要点：①H2O 按注意力分数保留 top-k 历史 token（驱逐低分），SnapKV
在 prefill 末尾观察 attention pattern 预选 decode 要保留的列；
②与分页天然配合——驱逐以"块"为单位释放回池（我们 BlockManager
的引用计数回收直接复用）；③风险是召回损失（被驱逐 token 后文又
变重要），所以都是"软"压缩、有比例超参；④与投机解码的组合：
验证前向要一次读 K+1 行查询的完整 KV，稀疏化后接受率会掉。

**K5 ★ FP8/W8A8 推理落地注意什么？**
要点：per-tensor vs per-channel scale、dynamic vs static 校准；
KV fp8 对批容量的放大与 decode kernel 精度（呼应 E1——低精度累
加翻 argmax 的教训在 fp8 更明显）；DeepSeek-V3 是 fp8 训推一体的
标杆；量化的正确性验证方法 = 困惑度回归 + 逐层 cos-sim 对比。

**K6 ★ vLLM V1 / overlap scheduler 快在哪？**
要点：CPU 调度（python 选批、建表、拷数据）与 GPU 上一步的执行
重叠——batch i 在 GPU 跑时 CPU 已备好 batch i+1，消除了我们引擎
每步的同步间隙（我们 A2 里输掉的 1.4–2× 主要就在这）；配合
piecewise CUDA Graph（不同 batch 桶拼装）；说得出"我们 benchmark
测到的 gap 与该机制的因果"即是把项目与前沿对上号。

**K7 ★ Ring/Context Parallel 解决什么？**
要点：序列维切分，attention 分块环传 KV，解决单卡放不下超长序列
的激活/KV；与 TP 正交；代价是通信轮数 ∝ 序列长/块大小；我们 H3
答过的"decode 单 program 串行扫 KV"是它的单卡前哨问题。

## L. 生产化与性能工程

**L1 ★★★ 线上"慢了"，你的完整定位流程？**
要点（分层漏斗）：①SLO 面板定位层级——队列 vs prefill vs decode
（TPOT 分解）；②引擎级——每步 token 均时 + batch 曲线，判断是
launch-bound（图失效/回退 eager？）还是 memory-bound（KV 满了开始
抢占/重算？）；③kernel 级——nsys 看时间线空隙（间隙=CPU 瓶颈，
密集小 kernel=launch 瓶颈），ncu 看单 kernel 的带宽利用率（MBU）；
④判断 memory/compute bound 的算式：bytes_moved / FLOPs 与卡的
ROUT（如 4090 ~1TB/s : ~165TFLOP）比较。把我们"26ms eager vs 5ms
实算"的定位过程当案例讲。

**L2 ★★ 推理 SLO 的 goodput 怎么定义与测量？**
要点：有效吞吐 = 满足（TTFT,TPOT）双 SLO 的完成请求数/秒；测量
要固定到达过程（泊松注入）而非离线批打满——离线吞吐（我们基准
的口径）会高估在线 goodput；调参目标从"最大吞吐"变为"SLO 曲线
膝点"。

**L3 ★ $/M tokens 成本模型怎么算？怎么选型？**
要点：$/M = 实例时价 ×(SLO 违约下所需实例数)/有效吞吐；把 PD 分
离、投机、量化都折算进该式做选型（例：投机 +30% 吞吐 vs 多一份
draft 显存的机会成本）；我们垂直场景的卖点之一是"同卡冷启动快
→ 缩容到零的成本可行"（J6）。

**L4 ★ 性能回归怎么防劣化？**
要点：bench-in-CI——固定种子/固定 prompt 集/离线口径下限（我们
rag_bench 的进程隔离 + 3 次中位即雏形）+ 关键不变式单测（我们的
CPU 单测：块所有权/调度不变式）；防图退化：捕获耗时与桶数断言。

## M. 反问与表达建议（收尾）

- 主动引导：把话题引向 C1/C2（无损性）和 D2（悬垂指针排查）——
  这是项目里最"值钱"的两段；
- 每个数字都能说出测量条件（模型/卡/batch/贪心/3 次中位），被追问
  波动时坦承共享 GPU 的 ±15% 噪声与最终背靠背复测；
- 不贬低 vLLM/Ollama：强调"垂直负载 + 明确交叉点"的工程判断。

---

### 覆盖对照（自查表）

| 岗位 JD 关键词 | 对应题 |
|---|---|
| vLLM/SGLang 原理 | I1–I8, B1–B4, F1–F3, K6 |
| 投机解码/草稿模型 | C1–C7, I6, K2 |
| CUDA Graph/Kernel 开发 | D1–D4, I5, E1–E4 |
| 性能调优/Profiling | E1, D2, A2, L1, L4 |
| Benchmark 设计较 | A4–A5, L2 |
| 分布式推理（单机多卡） | G2, H1 |
| 大规模集群/PD 分离/服务化 | J1–J7, L2–L3 |
| 前沿架构（MLA/MoE/量化/长文本） | K1, K3–K5, K7 |
| SLO/容量规划/成本 | J4, L1–L4 |
| 论文到工程落地 | C1–C3, I1–I5, K1–K2 |
