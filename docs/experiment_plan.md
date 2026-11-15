# 实验方案：模型、基准、数据、SFT、RL、引擎、代码

2026-11-15 定稿第一版。前提：用户服务器上已经下载好 **Qwen3.5-2B / 4B / 9B**。

本文档把 [research_line.md](research_line.md) 和 [algorithm_design.md](algorithm_design.md) 落到可执行的层面，并**修正**了两处原设计：
- 动作接口从行内标签改成原生 tool call（§2）；
- 主实验模型从"纯 dense"改成 Qwen3.5 混合架构，另加一个 dense 对照（§1）。

---

## 0. 决策总表

| 问题 | 决定 | 理由 |
|---|---|---|
| 模型 | 2B 调试；**4B 主力 RL**；**9B 终版**，并和 CLM、ACM 对比；另加一个纯 dense 对照（Qwen3-8B，服务器能下载的话），只用于残留实验 K3 | 用户已有；CLM 的 RL 和 ACM 用的都是 Qwen3.5-9B |
| 动作接口 | 原生 tool call：`kv_out`、`kv_in`、`note`；对照条件里加 `kv_drop` | 模型本来就会调工具，零样本遵守率高；动作只发生在回合之间，引擎只需支持回合间的会话操作 |
| E0–E2、SFT 数据、小规模评测 | 自写的 HF 循环 | 不改任何引擎，最快拿到结论 |
| RL rollout、大规模评测、系统评测 | **SGLang monkeypatch**（仿照本仓库的 `suffix_cache_reuse/overlay.py`） | KV 按 token 粒度管理，没有对齐问题；已支持混合模型的循环状态；SCR 已经在 SGLang 上给混合模型做过同类改动 |
| vLLM | 第一版不用 | 混合模型下 attention 的 block 被放大到几百 token，换页太粗；还要改位置计数、会话保持、前缀哈希（§7.3） |
| 训练 | HF 模型 + FSDP2，自写 SFT 和 GRPO trainer；全注意力层用 FlexAttention 块掩码，GDN 层照常 | P3 保证一次前向就精确；rollout 和训练两端都要大改，用框架省不了多少事 |
| RL 算法 | GRPO（token 级损失、不除以标准差、clip-higher）+ CLM 双通道 advantage + TIS 校正 rollout 和训练的概率差 | §6 |
| 基准 | T1 仓库（自建，有 oracle）；T1b 查找（对相似度检索友好，作对照）；**T2 BrowseComp-Plus，用 ACM 的 680/150 划分**；T2b LongMemEval-S（可选）；T3 视频（后续，Qwen3.5 原生多模态） | §3 |

---

## 1. 模型：Qwen3.5 是混合架构，这改变了什么

### 1.1 事实 [S]

- 2B、4B、9B 都是混合架构：Gated DeltaNet（线性注意力）和全注意力按 **3:1** 交替。0.8B 的模型卡写的是 6 × (3 × GDN + 1 × 全注意力)；
- 原生多模态，视觉和语言从预训练开始就融合（early fusion）；
- 原生上下文 262,144 token。

**要在服务器上核对**：层数、`layer_types`（或同类字段）、KV 头数、head_dim、partial rotary（§12 给了命令）。

### 1.2 对设计的影响

| 方面 | 影响 |
|---|---|
| 换页的作用范围 | 只作用在**全注意力层**，约占 1/4 的层。GDN 层的循环状态始终包含整条日志 |
| P1–P3 | **仍然精确**。推理时 GDN 按顺序处理只追加的日志，训练时也是；全注意力层在两边用同一个驻留集 |
| P4（残留） | 多了一条通道：除了后续 token 的 KV，**GDN 状态本身就带着被换出页的信息**。K3 要拆成两条通道，所以需要 dense 对照 |
| 叙事 | 全注意力层是模型可控的**精确随机访问记忆**，GDN 是始终在线的**有损记忆**。模型决定哪些页享有精确回忆 |
| E0 的新看点 | 混合模型只保留最近的页可能也不太掉点，因为有 GDN 兜底。如果是这样，准确率上的空间就小，论文主打显存和成本 |
| 显存 | 只有全注意力层存 KV，每 token 的 KV 约为同尺寸 dense 模型的 1/4。预算 B 是研究设定，不是真的会 OOM；论文的动机要讲长上下文（262K）和批量服务 |
| 可比性 | CLM 的 RL 用 Qwen3.5-9B，BrowseComp-Plus 上 28.8% → 42.5%（[clm_notes.md](clm_notes.md) §8，[N]）；ACM 也用 Qwen3.5-9B，并公开了 checkpoint |
| 位置编码 | Qwen3.5 用 M-RoPE，`position_ids` 是三维的；纯文本 token 的三个分量相同。全局位置号照常使用 |

### 1.3 思考内容的处理

Qwen 系列的 chat template 会删掉前面回合的思考内容。我们的做法是：
- 每个回合的 `<think>…</think>` 单独成一页，回合结束时由系统默认换出。这和模板的分布一致；
- 模型可以用 `kv_in` 把它换回来；
- 这本身就形成一组干净的对比：系统统一剥掉思考 vs 由模型决定。

---

## 2. 动作接口（取代 algorithm_design §2.2 的行内标签）

### 2.1 工具

```json
{"name": "kv_out",  "description": "把页移出显存（可随时换回）。换出后看不到正文，只剩页头。",
 "parameters": {"pages": "int 列表或区间字符串，如 \"3,5-8\""}}
{"name": "kv_in",   "description": "把之前换出的页换回显存。",
 "parameters": {"pages": "同上"}}
{"name": "note",    "description": "写一条笔记。笔记会作为新的一页常驻，可以代替整页正文。",
 "parameters": {"text": "string"}}
{"name": "kv_drop", "description": "永久删除页（只在不可逆对照条件里提供）。",
 "parameters": {"pages": "同上"}}
```

- 一次回复里可以同时调用任务工具和 KV 工具，系统先执行 KV 工具；
- 每个工具结果末尾附一行仪表盘：
  `[KV] resident 21.3K/28.7K (74%) | on GPU: 1-3,9,12-14 | in host: 4-8,10-11`
- 笔记作为系统插入的新页做 prefill，带页头 `[[PAGE n note]]`。它是在源页还驻留时写的，所以 KV 里编码了源页的内容；
- **超预算**：新的工具输出放进来会超出 B 时，系统按 LRU 强制换出最久没用的非钉住页，并在仪表盘上说明。这计入代价。

### 2.2 一个回合的流程

```
系统：     [[PAGE 9 tool=search q=...]] <结果> [KV] ...     ← prefill，新页
模型：     <think>…</think>                                 ← 第 10 页（回合结束后系统默认换出）
           回复正文 + tool_calls: kv_out(4-8), search(...)   ← 第 11 页
引擎：     执行 kv_out → 改驻留集 → 执行 search → 下一回合
```

所有驻留集的改变都发生在回合之间，所以引擎只需要支持"会话的 KV 保留到下一回合"和"下一回合开始前改驻留集"。

### 2.3 以后的扩展

行内标签（推理中途换页）留到长推理任务时再做，需要引擎支持生成中途暂停。

---

## 3. 基准与测试床

| 编号 | 基准 | 数据来源 | 规模和划分 | 指标 | 用途 |
|---|---|---|---|---|---|
| **T1** | 多轮仓库环境 | `state_study/probes/warehouse.py` 改造，自动生成 | 训练种子 0–9999，测试种子 100000 以后，严格不重叠 | 决策准确率；错误按 `stale`、`occupied` 分类 | 有逐步真值和 Belady oracle；`delayed` 隔离通知和问题字面不相似，是相似度检索的盲区 |
| **T1b** | 多轮查找环境（新写） | 自动生成：很多页"实体 → 属性"的记录，问题按实体名查 | 同上 | 准确率 | 对相似度检索友好的对照。QuestPage 应该在这里接近 oracle |
| **T2** | **BrowseComp-Plus** | HF `hltcoe/BrowseComp-Plus`（语料、BM25 Lucene 索引、检索代码），题目划分用 ACM 仓库 `third_party/methods/acm/data/` | **训练 680 / 评测 150**（全集 830）[R] | 评判模型判对错 | 主基准。和 ACM（同一划分、同一模型）直接比；CLM 在同一划分上重跑 |
| T2b（可选） | LongMemEval-S | 用户自己下载（旧主线已接入 submodule） | 官方划分 | 准确率 | 非检索型的长记忆任务：会话按顺序到达、最后提问，是视频流的文本版 |
| T3（后续） | OVO-Bench 等 | — | — | — | Qwen3.5 原生多模态，同一个模型就能做视频扩展 |

### 3.1 T2 的本地化

全部可以离线运行，不依赖外部 API：

- **检索**：pyserini BM25（需要 JDK 21），沿用 ACM 的设置：`search` 返回片段，新结果 512 token、看过的 128 token；`get_document` 最多 8192 token [R]；
- **评判**：ACM 的 `src/evaluator.py` 用 OpenAI SDK 和官方 simple-evals 评判模板 [R]。把 base URL 指向本地 SGLang 服务的评判模型即可：
  - 用你手头最大的模型做评判；
  - 先在 200 个答案上和 exact match 对一致率；不一致的抽样人工看；
  - 结果里注明评判模型（ACM 用的是 GPT-5，数字不能直接混比）。
- **预算**：沿用 CLM 的 BCP 设置，驻留预算 B = 28,672 token（[clm_notes.md](clm_notes.md) §2.3）；
- **可以复用的代码**：ACM 是 MIT 协议，可以复制它的 BCP 数据加载、检索工具和评判代码，注明出处。

---

## 4. 数据采集

### 4.1 轨迹记录格式（所有数据共用）

**token 进、token 出**：直接记引擎生成的 token id，绝不重新分词。

```
Trajectory
  tokens:      int[]          # 整条日志
  role:        u8[]           # system / tool / assistant / think / note / gauge
  page_id:     int[]          # 每个 token 属于哪一页
  seg_id:      int[]          # 驻留集版本号（每次 kv_* 生效后 +1）
  resident:    bool[S, P]     # 每个版本下各页是否驻留
  loss_mask:   u8[]           # 1 = assistant 生成的 token
  ctrl_mask:   u8[]           # 1 = kv_out / kv_in / note 工具调用的 token
  logprobs:    float[]        # rollout 时引擎给的 logprob（RL 用）
  reward, cost: float; meta: {env, seed, budget, model, ...}
```

训练时由它直接生成掩码（§5.2）；推理时也按它记录。两边用同一个数据结构。

### 4.2 E0–E2 的数据

不需要训练数据：T1 和 T1b 的场景按种子生成，T2 从评测 150 题里抽 100 题。

### 4.3 SFT 数据：四个来源

| 来源 | 做法 | 规模 |
|---|---|---|
| **A. T1 程序 oracle + 事后补写理由** | 生成器给出每页的"活事实"和未来的使用时刻；按三种 oracle（Belady 换页、先写笔记再换出、混合）决定每回合的 KV 动作。理由不用模板，而是把 oracle 的动作作为提示，让 9B 写出自然语言的理由（STaR 的 rationalization），再过滤掉理由和动作不一致的样本 | 约 3000 条 |
| **B. 给 ACM 公开的轨迹补上 KV 动作** | ACM 在 HF 上公开了基座 Qwen3.5-9B 在训练 680 题上的 pass@4 轨迹（ReAct 和 memtool 两种模式，约 4.4 GB）[R]。取答对的 ReAct 轨迹，按事后信息判断每个工具输出后来有没有被用到（后续回合引用了它的 docid 或内容，或者答案出自它），再按 Belady 插入 `kv_out` / `kv_in`，然后按我们的协议重新渲染 | 视答对的条数而定，预计 1000 条以上 |
| **C. T2 拒绝采样** | 零样本 9B 在训练 680 题上、在预算压力下采样（每题 8 条，预算取自然长度的 25–50%），只留答对、协议合法、没有被强制换出的；每题最多留 2 条 | 约 500–1500 条 |
| **D. 蒸馏到小模型** | 用 9B 的 A、B、C 数据训练 4B 和 2B | — |

质量检查：
- **规则读者**：T1 上只看 oracle 驻留集里的文本就能 100% 答对（项目约定）；
- 协议合法率 100%；页号引用都存在；
- 按题目去重；训练集和评测集的题目、种子严格分开。

注意：B 来源依赖"事后判断哪些页有用"，标注规则本身会影响 SFT 学到的策略。论文里要写清楚，并做"只用 A+C"的消融。

---

## 5. SFT

### 5.1 设置

| 项 | 2B / 4B | 9B |
|---|---|---|
| 方式 | 全参数 | 全参数（显存够）或 LoRA r=64、alpha=128（挂在全部线性层） |
| 学习率 | 1e-5，cosine，warmup 3% | 全参数 5e-6；LoRA 1e-4 |
| epoch | 2 | 2 |
| 序列长度 | 32K，多条轨迹打包 | 32K |
| 损失 | 只算 assistant token（`loss_mask`） | 同左 |

### 5.2 注意力的实现

- **全注意力层**：把 HF 里 Qwen3.5 的全注意力层打补丁（monkeypatch），换成 FlexAttention 加 BlockMask：

```python
def make_mask_mod(doc, page, seg, resident):   # 全是 GPU 上的 int/bool 张量
    # resident[seg_global, page_global]：打包时 seg 和 page 都用全局编号
    def mask_mod(b, h, q, kv):
        same_doc = doc[q] == doc[kv]
        causal   = kv <= q
        own_page = page[q] == page[kv]
        visible  = resident[seg[q], page[kv]]
        return same_doc & causal & (own_page | visible)
    return mask_mod

block_mask = create_block_mask(make_mask_mod(doc, page, seg, resident),
                               B=None, H=None, Q_LEN=T, KV_LEN=T)
```

- **GDN 层**：照常按因果顺序处理，用 `cu_seqlens` 在打包的轨迹之间重置状态（fla 的 varlen kernel）。GDN 本来就看整条日志，和推理一致；
- **位置**：每条轨迹从 0 开始的全局位置，和推理时一样；
- **验收**：在 held-out 集上，协议合法率 > 95%，T1 准确率不低于零样本，强制换出率明显下降。

---

## 6. RL

### 6.1 算法

- **GRPO**，组大小 G = 8：
  - 优势只减组内均值，不除以标准差；
  - token 级损失；
  - clip 上下界 0.2 / 0.28（clip-higher）；
  - 不加 KL 项（视稳定性再加）。
- **奖励**：r ∈ {0, 1}，T1 判对错，T2 用评判模型。
- **代价** C：按 [algorithm_design.md](algorithm_design.md) §5 计算，κ 取 batch = 32 的值。Qwen3.5 只有全注意力层存 KV，所以 κ 要用实际 config 重新折算。强制换出另外每次加固定成本。
- **双通道 advantage**（沿用 CLM，[clm_notes.md](clm_notes.md) §5）：

```
A[i, t] = (r_i − mean_g r) + w_eff · a_eff_i · ctrl_mask_i[t]
a_eff_i = clip((C̄ − C_i) / C̄, −1, 1)    # 只在组内成功的轨迹之间比；成功少于 2 条的组为 0
w_eff = 0.25
```

- **TIS 校正**：rollout 引擎（SGLang）和训练（HF + FlexAttention）算出的概率会有差异，混合模型的 GDN kernel 差异可能更大。每个 token 乘截断重要性权重 `min(π_train / π_rollout, 2)`。训练前先测这个差异有多大（§7.2 的引擎一致性测试）。
- **课程**：
  - 预算 B 每条轨迹随机取自然长度的 {25%, 50%, 75%}；
  - 先只用 T1（短、便宜、奖励没有噪声），再混入 T2，T1 : T2 = 1 : 3；
  - 不可逆对照（K2）用同样的设置单独训一份，只提供 `kv_drop`、不提供 `kv_in`。

### 6.2 超参起点（4B）

| 项 | 值 |
|---|---|
| 每步题数 × G | 16 × 8 = 128 条轨迹 |
| 学习率 | 1e-6，constant |
| 每步更新 | 1 个 epoch、2 个 minibatch |
| 最大回合 | T1 30；T2 60 |
| 单回合最大生成 | 4096 |
| 温度 | 1.0 |
| 步数 | T1 约 200 步，再 T2 约 1 个 epoch（680 题） |
| 评测 | 每 20 步：T1 held-out 200 题 + BCP 评测 150 题中的 50 题 |

### 6.3 架构

```
┌─────────────┐  HTTP  ┌──────────────────────────────┐
│ 环境 worker │ ─────▶ │ SGLang + self-paging overlay │  ×N 个推理 GPU
│ T1/T2 + 协议 │ ◀───── │ 会话 KV、驻留集、全局位置     │
└─────┬───────┘        └──────────────▲───────────────┘
      │ Trajectory（token、掩码、logprob）│ update_weights_from_disk（先用简单的；之后换 NCCL）
      ▼                                │
┌───────────────────────────────────────┴──┐
│ trainer：HF Qwen3.5 + FSDP2 + FlexAttention │  ×M 个训练 GPU
│ 双通道 advantage、TIS、GRPO 损失             │
└───────────────────────────────────────────┘
```

先做同步版本：采完一步再训一步。稳定后再改成异步。

**监控**：准确率、协议合法率、平均和峰值驻留、换回率（反悔）、强制换出率、笔记 token 占比、代价 C、rollout 和训练的 log 概率差、每回合时间。

**为什么不直接用 verl 或 slime**：rollout 端需要会话级的换页引擎，训练端需要自定义掩码，两端都要大改。2B/4B 规模下，自写约 1500 行的循环更可控。规模上到 9B、需要异步或多机时，再迁移到 verl（FSDP、SGLang rollout、多轮 agent loop）。slime 是 CLM 用的框架，但它用 Megatron 训练，自定义掩码最难接。

---

## 7. 推理引擎

### 7.1 HF 循环（阶段 0–1）

```python
class LogStore:                       # 逻辑换页：全部 KV 留在 GPU，用下标选驻留部分
    K, V: dict[layer, Tensor]         # [1, kv_heads, log_len, head_dim]，只存全注意力层
    linear_state: object              # GDN 的 conv / recurrent 状态，代表整条日志
    pages, resident, log_len

def step(model, store, new_ids):
    idx = store.resident_token_index()                        # 排好序的全局位置
    cache = make_hybrid_cache(store.linear_state,             # 以所装 transformers 里
                              {l: (store.K[l][:, :, idx],     # Qwen3.5 的 cache 类为准
                                   store.V[l][:, :, idx]) for l in store.full_layers})
    n = len(new_ids); L = len(idx)
    out = model(input_ids=new_ids[None], past_key_values=cache, use_cache=True,
                position_ids=global_pos(store.log_len, n),    # M-RoPE：三个分量相同
                cache_position=torch.arange(L, L + n))
    store.append(new_ids, tail_kv(cache, L), cache.linear_state)
    return out.logits
```

- 每次前向前都按下标重新选一次，代码最简单；驻留集没变时，可以缓存上一次拼好的 cache 直接追加；
- 物理换页（真的挪到内存）以后再加，换出 = `K[l][:, :, span].to("cpu", non_blocking=True)`。

### 7.2 SGLang overlay（阶段 2 起）

**写法**：和 SCR 一样做 monkeypatch，不改 SGLang 源码。SCR 已经示范了需要碰的模块：`Scheduler`、`Req`、`ScheduleBatch`、`HybridReqToTokenPool`、`mamba_radix_cache`；以及做法：会话匹配、会话私有的 KV 槽、GDN 状态的保存和恢复（`_ssm_save` / `_ssm_restore`）、KV 拷贝（`_copy_kv`）、找出全注意力层（`_full_layer_ids`）。新代码按这些函数的写法来；具体函数名以你装的 SGLang 版本为准，SCR 是针对 0.5.16 写的。

| 组件 | 做什么 |
|---|---|
| 会话 | 每个会话保存：日志中每个 token 的 KV 槽号、各页的 token 区间、驻留区间列表、日志长度、GDN 状态快照。会话的 KV 不进 radix 树，也不参与淘汰（新版 SGLang 的 streaming session 提供了类似的会话槽 [S]，可以先看能不能直接用） |
| 新回合 | 请求只带新 token（工具结果 + 仪表盘）。把驻留 token 的槽号按日志顺序写进这个请求的 `req_to_token` 行，当作已缓存的前缀，长度记为 L_res；只 prefill 新 token |
| 位置 | 打补丁让新 token 的位置 = 日志长度 + i，而不是 L_res + i。prefill 和 decode 两条路径都要改；Qwen3.5 还要改 M-RoPE 的位置。做法是给每个请求加一个 `pos_offset = log_len − L_res` |
| GDN 层 | 回合开始时从会话快照恢复状态，回合结束时保存。换页不影响 GDN 状态 |
| 改驻留集 | 新接口 `POST /kv_set_resident {session_id, ranges}`。驻留集的逻辑（页、协议、LRU）全部放在客户端，引擎只认 token 区间，保持引擎补丁最小 |
| 逻辑模式 | 不释放槽，只是不写进 `req_to_token`。RL 用这个模式就够了 |
| 物理模式 | 换出：把全注意力层的 K/V 拷到锁页内存，释放槽；换入：申请槽、拷回、更新会话。系统评测时才用 |
| 共享前缀 | system prompt 和任务描述是在全部驻留的情况下写的，G 条 rollout 完全相同，可以照常走 radix 共享 |
| 记录 | 每回合返回 token id 和 logprob，供训练使用 |

**引擎一致性测试**（GPU 上，进 RL 之前必须过）：
- 同一条带换页的轨迹，SGLang 返回的 logprob 和 HF + FlexAttention 单次前向算出的 logprob 对比：bf16 下平均绝对差小于设定阈值，并记录最大差；
- 换出再换入以后，后续 logprob 和从没换出过的那条轨迹相同；
- 物理模式和逻辑模式的输出逐位相同。

### 7.3 vLLM：要改什么，为什么第一版不用

非要用 vLLM 的话，需要改五处：

1. **会话保持**：vLLM 没有原生的"请求跨回合保留 KV"。要么自己加会话管理，要么每回合发完整日志、靠前缀缓存命中；
2. **前缀哈希**：KV 不再只由 token 前缀决定，还取决于写入时的驻留集。不同 rollout 的 token 相同但换页历史不同，会互相命中，得到错误的 KV。要把"写入时驻留集的签名"加进每个 block 的哈希（extra key）；
3. **block table 过滤**：每个请求只把驻留 block 放进 block table，`seq_lens` 用驻留长度；
4. **位置和槽位分离**：`GPUModelRunner` 准备输入时，`positions` 和 `slot_mapping` 都从 `num_computed_tokens` 推出来，必须拆成"逻辑位置计数器"和"缓存下标"两套；
5. **对齐**：vLLM 按 block 管 KV，页边界必须落在 block 边界上。混合模型更糟：attention 的 block 被放大到和 mamba 状态对齐，有报告说 Qwen3.5 是 528 token，短于一个 block 的前缀根本命中不了 [S]。社区有把匹配粒度和物理 block 解耦的 PR（`prefix_match_unit`）[S]，但换页粒度仍然受物理 block 限制。

结论：vLLM 能改，但比 SGLang 侵入性大、粒度粗。等 SGLang 版本跑通、需要和 vLLM 生态对接时再说。

---

## 8. 代码结构（`self_paging/`，取代 algorithm_design §12）

```
self_paging/
  core/        pages.py        Page、PageTable、驻留集、LRU、不变量检查（和引擎无关）
               protocol.py     页头、转义、仪表盘渲染；kv_* 工具 schema；解析和合法性检查；超预算处理
               trajectory.py   §4.1 的记录格式；存取 JSONL / Arrow
               costs.py        C、峰值驻留、C_pcie；按实际 config 折算 κ
  hf/          logstore.py     §7.1 的 LogStore 和混合 cache 的拼装
               runner.py       HF agent 循环
               masks.py        Trajectory → mask_mod / BlockMask；测试用的稠密 4D 掩码
               patch_qwen35.py 给全注意力层打补丁接 BlockMask；GDN 层不动
  sglang_overlay/              仿 suffix_cache_reuse：session.py、positions.py、resident.py、serve.py、_site/sitecustomize.py
  envs/        base.py、t1_warehouse.py、t1b_lookup.py、t2_bcp.py（复用 ACM 的检索和评判，注明出处）
  policies/    full.py、recent.py、attn_evict.py、quest_page.py、belady.py
  data/        oracle_sft.py、rationalize.py、annotate_acm.py、reject_sample.py
  train/       sft.py、grpo.py、advantage.py、weight_sync.py
  experiments/ e0_headroom.py、e1_zeroshot.py、e2_residue.py、eval_t1.py、eval_bcp.py
  tests/       test_equivalence_hf.py   随机初始化的小 Qwen3.5 配置：拼 cache 推理 = 掩码单次前向（含 GDN 层）
               test_protocol.py、test_belady.py、test_rule_reader.py
               test_engine_parity.py    需要 GPU：SGLang vs HF
```

阶段 0 能在云端完成的：`core/`、`hf/`、`policies/`、`envs/t1*`、`data/oracle_sft.py` 和对应的测试。用随机初始化的小 Qwen3.5 配置跑，不需要下载权重。

---

## 9. 实验矩阵

### 9.1 继续/止损（阶段 1，只推理）

| 实验 | 设置 | 模型 | 规模 |
|---|---|---|---|
| E0 提升空间 | T1 三种关键事件 + T1b × 预算 {25, 50, 100}% × 策略 {Full, Recent, AttnEvict, QuestPage, Belady} | 9B、4B | 每格 200 个场景 |
| E1 零样本协议 | T1 多轮，预算 50%；T2 抽 50 题 | 2B、4B、9B | 每个模型 100 条轨迹 |
| E2 残留 | T1 `explicit`：FULL / OUT-LATE / OUT-EARLY / REWRITE × 剂量 {1K, 4K, 16K} | 9B + dense 对照 | 每格 200 个场景 |

判断规则见 [algorithm_design.md](algorithm_design.md) §9。混合模型上额外看一点：如果 Recent ≈ Full，说明 GDN 兜住了，准确率空间小，论文转向显存和成本。

### 9.2 主实验（阶段 3）

| 编号 | 内容 | 对应 |
|---|---|---|
| 表 1 | T1、T1b：零样本 / SFT / RL × 2B、4B、9B，对比 §7 的全部基线，预算 25% 和 50% | 主结果 |
| 表 2 | BCP 评测 150 题：本文 vs CLM（同模型重跑）vs ACM（公开的 9B checkpoint）vs SideQuest 式（只能删的 RL）vs QuestPage vs Recent vs Full | K1，主结果 |
| 图 1 | 准确率对代价 C 的 Pareto 曲线（batch = 1 和 32 两种折算） | K1 |
| 图 2 | 可逆 vs 不可逆 RL：驻留集大小随回合的变化、换回率、准确率 | K2 |
| 图 3 | 残留：E2 的剂量曲线，混合 vs dense；RL 前后对比 | K3 |
| 图 4 | T1 可预测性旋钮 vs 笔记 token 占比、换回次数 | K4 |
| 图 5 | 2B → 9B：零样本 vs 训练后的协议遵守率和准确率 | K5 |
| 表 3 | 系统：SGLang 物理模式下的峰值显存、batch = 32 时的吞吐、首字延迟，对比 CLM + SCR | 只在这里报墙钟 |

统计：同一批题目在各条件间配对；二值结果用 McNemar，区间用 bootstrap；RL 至少 2 个种子。

---

## 10. 算力估计（等用户给出 GPU 后再细化）

- **E0–E2**：9B bf16 推理需要约 20 GB 显存加上 KV，单张 40 GB 以上的卡就够。主要是 prefill 和短回答，1–2 天；
- **SFT**：9B 全参数需要 8 张 80 GB 卡做 FSDP；卡小就用 LoRA。几千条轨迹、32K 长度，每个 epoch 几个小时；
- **RL（4B，T1）**：每步 128 条轨迹，每条约 3K 生成 token，200 步约 7700 万生成 token。按单卡 SGLang 几千 token/秒估算，几张卡跑 1–2 天；
- **RL（T2）**：轨迹长得多（CLM 设置下单题生成可达 32K），成本主要在这里，需要先在 T1 上把流程调稳。

## 11. 时间线

| 周 | 内容 | 谁来做 |
|---|---|---|
| 1 | 阶段 0：`core/`、`hf/`、`policies/`、`envs/t1*`、`tests/`；SFT 数据来源 A 的生成器 | 云端（我） |
| 1 | 核对 config；在真实权重上跑 HF 等价性测试 | 用户服务器 |
| 2 | E0、E1、E2，做继续/止损判断 | 用户服务器（脚本由我写好） |
| 3–4 | SGLang overlay + 引擎一致性测试；SFT 数据 B、C；SFT | 两边 |
| 5–8 | RL：先 T1，再 T2；K2 不可逆对照 | 用户服务器 |
| 9+ | 系统评测、视频扩展、写作 | — |

## 12. 需要用户在服务器上确认的事

```bash
python - <<'EOF'
import json, sys
for m in ["Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B"]:
    c = json.load(open(f"/path/to/{m}/config.json")); t = c.get("text_config", c)
    keys = ["num_hidden_layers", "layer_types", "full_attention_interval", "num_attention_heads",
            "num_key_value_heads", "head_dim", "partial_rotary_factor", "max_position_embeddings"]
    print(m, {k: t.get(k) for k in keys})
EOF
```

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

```bash
pip list 2>/dev/null | grep -iE "^(torch|transformers|sglang|vllm|flash-linear-attention|flash_attn|flashinfer-python|verl) "
```

另外请确认：
- 服务器能否访问 HF：要下载 BrowseComp-Plus 语料和索引、ACM 公开的轨迹，以及 dense 对照模型 Qwen3-8B；
- 服务器上有没有 JDK 21（pyserini 需要）；
- 用哪个模型做 T2 的本地评判。
