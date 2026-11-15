# 实验方案（v2）：Qwen3-8B

2026-11-15。用户决定主模型改用 **Qwen3-8B**（纯 dense）。v1 是针对 Qwen3.5 混合架构写的，已被本版取代，可以在 git 历史里查到（提交 `2746535`）。

配套文档：
- [data_and_training.md](data_and_training.md)：SFT 和 RL 的数据从哪来、怎么训练（本版 §4、§6 只写概要）；
- [algorithm_design.md](algorithm_design.md)：形式化、掩码、代价口径、E0–E2 协议；
- `self_paging/README.md`：已经实现并测试过的代码。

---

## 0. 决策总表

| 问题 | 决定 |
|---|---|
| 模型 | **Qwen3-8B**，唯一的策略模型。Qwen3.5-9B 已经下载好，用作本地评判模型，以及特权提示生成时的备用老师（§4） |
| 换页作用范围 | 全部 36 层。P1–P4 都在所有层上成立，残留只有一条通道：后续 token 的 KV |
| 位置编码 | 用全局位置号。日志超过 32K 的基准（BCP、LongMemEval）对所有方法统一开 YaRN（factor 4，到 131K）；T1 把日志控制在 32K 以内，不开 YaRN（§1.2） |
| 动作接口 | 原生 tool call：`kv_out`、`kv_in`、`note`，对照条件加 `kv_drop`；驻留集只在回合之间改变 |
| 推理参考实现 | `self_paging/hf/logstore.py` 的 `PagedKV`，**已实现并通过等价性测试** |
| RL rollout 引擎 | SGLang monkeypatch（逻辑换页）。vLLM 在 dense 模型上也可行，作为备选（§7） |
| 训练 | HF + FSDP2；`attn_implementation="flex_attention"` 直接接收我们的 BlockMask，**已验证不用改模型代码** |
| 基准 | T1 仓库、T1b 查找（合成）；**T2-A BrowseComp-Plus**（680/150）；**T2-B LongMemEval-S**（问题最后才给出，流式记忆） |
| 评判 | 本地 Qwen3.5-9B + 官方评判模板，先做一致性校验（§3.5） |

---

## 1. Qwen3-8B：事实和影响

### 1.1 事实 [K]（凭已有知识，要用 config 核对）

| 项 | 值 |
|---|---|
| 层数 / hidden | 36 / 4096 |
| 注意力 | 32 个 Q 头、8 个 KV 头（GQA）、head_dim 128；每层都是全注意力 |
| 原生上下文 | 32,768（`max_position_embeddings` 40,960）；YaRN factor 4 可到 131,072 |
| RoPE | theta = 1,000,000 |
| KV 大小 | 2 × 36 × 8 × 128 × 2 B ≈ **144 KB/token**（bf16）：28K token ≈ 4.0 GB，128K ≈ 18 GB |
| 权重 | 约 16.4 GB（bf16） |
| 思考模式 | 同一个模型可以思考或不思考（`enable_thinking`，或在消息里写 `/think`、`/no_think`） |
| 工具调用 | Hermes 风格：assistant 输出 `<tool_call>{"name":…,"arguments":…}</tool_call>`，工具结果放在 user 轮的 `<tool_response>` 里 |
| 推荐采样 | 思考模式 temperature 0.6、top_p 0.95、top_k 20；非思考模式 0.7、0.8、20 |

### 1.2 对设计的影响

| 方面 | 影响 |
|---|---|
| 精确性 | 等价性测试已经在 Qwen3 的模型结构上通过（`self_paging/README.md`）。全部 36 层都用同一个驻留集，不再有混合架构的 GDN 通道 |
| 残留（K3） | 只剩一条通道：被换出页驻留期间写下的后续 token 的 KV。机制更干净，E2 不再需要单独的 dense 对照 |
| **位置号会超出原生窗口** | 全局位置号随日志增长，和驻留了多少无关。例如驻留只有 28K，但日志已经 100K，新 token 的位置就是 100K。所以：①T1 把单个 episode 的日志控制在 32K 以内；②BCP 和 LongMemEval 对**所有方法**统一开 YaRN factor 4，日志上限定为 120K，到上限时系统提示模型立即作答；③E0 加一个 E0b，只在推理时比较"全局位置（有空洞）"和"压紧位置（SCR 那样重新旋转 key）"，看空洞本身伤不伤害。训练时模型一直看到的是带空洞的位置，SFT 和 RL 本身就会让它适应 |
| 思考内容 | Qwen3 的模板在一次用户提问之内的工具调用循环里会保留思考内容 [K]，所以只追加日志在一个 episode 内和模板分布一致。思考仍然单独成页，模型可以把旧的思考换出去，这是一个很自然的用途 |
| 可比性 | CLM 和 ACM 公开的数字都是 Qwen3.5-9B 上的，不能直接比。改为在 Qwen3-8B 上**重跑**它们的零样本版本（CLM harness、ACM 的 memtool 模式），和我们同模型对比 |
| BCP 的难度 | Qwen3-8B 的检索 agent 能力比 Qwen3.5-9B 弱，在 BCP 上准确率可能很低，RL 组内会几乎全是 0。E1 先测训练 680 题上的 pass@8，不够就启用备选方案（§3.3） |

### 1.3 为什么不用 Qwen3.5（留档）

Qwen3.5 的 2B/4B/9B 是 Gated DeltaNet 和全注意力 3:1 的混合架构 [S]：换页只能作用在 1/4 的层上，GDN 的循环状态始终带着全部历史，残留多出一条去不掉的通道；vLLM 上 attention block 还会被放大到几百 token [S]。用户据此决定改用 dense 的 Qwen3-8B。

---

## 2. 动作接口

和 v1 相同，适配 Qwen3 的工具格式：

```json
{"name": "kv_out",  "parameters": {"pages": "如 \"3,5-8\""}}
{"name": "kv_in",   "parameters": {"pages": "同上"}}
{"name": "note",    "parameters": {"text": "string"}}
{"name": "kv_drop", "parameters": {"pages": "同上"}}
```

- 页：system 和任务描述（钉住）、每个工具结果、每个 assistant 回合的思考和正文（分成两页）、每条笔记；
- 页头 `[[PAGE n kind …]]` 由系统渲染，常驻并计入预算；工具输出里出现的 `[[PAGE` 一律转义；
- 每个工具结果末尾附一行仪表盘：`[KV] resident 21.3K/28.7K (74%) | on GPU: 1-3,9,12-14 | in host: 4-8,10-11`；
- 超预算时**截断新的工具输出**，而不是让系统按 LRU 换出（理由见 [data_and_training.md](data_and_training.md) §3.1）；
- **渲染必须逐 token 和 Qwen3 的 chat template 一致**：增量渲染每一段的 token，再写测试和 `tokenizer.apply_chat_template` 对整段对话的结果逐 token 比较（需要 Qwen3-8B 的 tokenizer 文件，在用户服务器上跑）。

---

## 3. 基准和测评怎么测

### 3.1 总表

| 编号 | 基准 | 任务形态 | 预算 B | 日志长度 | 训练 / 评测 |
|---|---|---|---|---|---|
| T1 | 多轮仓库（合成） | 事件逐页到达，多个决策点 | 自然长度的 25 / 40 / 60% | ≤ 32K | 训练种子 0–99999；测试种子从 100000 起；另有分布外测试集 |
| T1b | 多轮查找（合成） | 实体记录逐页到达，按实体名提问 | 同上 | ≤ 32K | 同上 |
| **T2-A** | **BrowseComp-Plus** | 检索 agent，问题一开始就给出 | 28,672 / 20,000 / 14,000 | ≤ 120K（YaRN） | 训练 680 / **评测 150**（ACM 的划分） |
| **T2-B** | **LongMemEval-S** | 约 50 个会话逐个到达，**问题最后才给出** | 8K / 16K | 约 115K（YaRN） | 500 题只做评测（零样本迁移）；可选按题目对半分出训练集 |

### 3.2 T2-B 为什么重要

LongMemEval-S 的历史约 115K token，远超 Qwen3-8B 原生的 32K，**必须管理**。而且问题最后才出现，读的时候不知道以后会问什么：
- 只能删（不可逆）时，模型只能猜哪些会话以后有用；
- 可以换出再换回（可逆）时，可以先换出，等问题出现后再按页头和笔记换回。

这正是 K2（可逆性）和 K4（写笔记 vs 换页）最能显出差别的地方，也是文本版的"流式视频"，和用户最初的 proposal 是同一个问题。数据集给出了证据会话（`answer_session_ids`），可以用来定义"需要"和做分析。

**协议**：每个会话作为一页到达，之后模型有一个短回合（可以 `kv_out`、`note`，也可以什么都不做）；最后给出问题，模型可以 `kv_in`，然后作答。另有一个"问题先给出"的变体作为对照。

### 3.3 BCP 的备选方案

E1 如果显示 Qwen3-8B 在训练 680 题上 pass@8 > 0 的题少于 15%：
1. 先把检索器换成 Qwen3-Embedding-8B 稠密检索。BCP 报告检索器对准确率影响很大 [S]，ACM 的代码支持 `RETRIEVER=qwen3_8b_embedding` [R]；
2. 还不够，就把 T2 的 RL 换成更容易的多跳问答（MuSiQue、2WikiMultihopQA，本地维基检索）。让检索返回较长的全文段落、预算设紧，这样换页才有意义。BCP 只用于评测。

### 3.4 基线和指标

**基线**（同一个 Qwen3-8B、同一个预算）：

| 类别 | 基线 |
|---|---|
| 参照 | Full（不设预算，需要时开 YaRN） |
| 系统 · 不可逆 | Recent（钉住页 + 最近的页）；AttnEvict（页级 SnapKV / H2O） |
| 系统 · 可逆 | QuestPage（每页存一个代表 key，每回合用 query 选前几页）；LongMemEval 上加 BM25 会话检索 |
| 模型 · 文本 | CLM 零样本（文本编辑）；ACM memtool 零样本（摘要、存盘、查回）；每页读完写摘要再删原文 |
| 模型 · KV | 本文：零样本 / SFT / RL；只能删的 RL 变体（SideQuest 式） |
| 上界 | Belady oracle（T1、T1b） |

**指标**：
- 任务：准确率。T1 按决策点；BCP 由评判模型判；LongMemEval 按官方题型分开报告（单会话、多会话、时间推理、知识更新、拒答）；
- 代价：C（batch = 1 和 32 两种折算，[algorithm_design.md](algorithm_design.md) §5）、峰值驻留、平均驻留、PCIe 字节数；
- 行为：协议合法率、换出和换回次数、换回率、截断率、笔记 token 占比、回合数。

### 3.5 测评协议

- **采样**：用 Qwen3 推荐的采样参数，每题跑 3 次，报告均值和 bootstrap 95% 置信区间；E0 和 E2 这种机制实验用贪心解码；
- **配对**：同一批题目、同一批种子在各条件间配对；二值结果用 McNemar 检验；
- **评判**：
  - 本地 Qwen3.5-9B，用 BCP 官方的 simple-evals 模板和 LongMemEval 官方的评判提示；
  - 先在 200 个答案上校验：和 exact match 对比，不一致的抽 50 个人工看，报告一致率；
  - 评判模型和策略模型不同，可以减少自我偏好；
  - 论文里注明评判模型。ACM 用的是 GPT-5，数字不能直接混比；
- **防泄漏**：BCP 评测 150 题和 LongMemEval 不进入任何训练数据，包括 SFT 的拒绝采样；T1 的训练种子和测试种子不重叠；
- **墙钟**：只在 SGLang 物理换页模式下报告（峰值显存、batch = 32 的吞吐、首字延迟）；HF 原型只报告解析代价。

---

## 4. 数据（概要，详见 [data_and_training.md](data_and_training.md) §1–§2）

| 来源 | 内容 | 本版的变化 |
|---|---|---|
| A. T1 特权提示生成 | 老师 = Qwen3-8B 加特权提示；连续 4 次不按计划输出时，换 Qwen3.5-9B 当老师，再不行用模板兜底 | 老师换了 |
| B. T2-A 轨迹重标注 | 原料**优先用 Qwen3-8B 自己**在完整上下文下答对的轨迹。ACM 公开的是 Qwen3.5-9B 的轨迹，分布不同，只作补充 | 原料换了 |
| C. 拒绝采样 | Qwen3-8B 在预算压力下采样，只留成功且合法的 | — |
| 所有来源 | 加"**可实现性**"过滤（[data_and_training.md](data_and_training.md) §2.6）：oracle 用到未来信息的动作，只保留观察者也能推断出来的那部分 | 新增 |

---

## 5. SFT（Qwen3-8B）

| 项 | 设置 |
|---|---|
| 方式 | 有 8 张 80 GB 卡就全参数（FSDP2 + 激活检查点）；卡少就 LoRA r=64、alpha=128，挂在全部线性层 |
| 学习率 | 全参数 5e-6；LoRA 1e-4；cosine，warmup 3% |
| epoch / 长度 | 1–2 个 epoch；32K，多条轨迹打包 |
| 掩码 | `attn_implementation="flex_attention"`，把 `make_mask_mod` 生成的 BlockMask 直接作为 `attention_mask` 传入（**已在 CPU 上验证**）；打包时传 `doc_of`，页号和版本号用全局编号 |
| 损失 | 只算 assistant token |
| 验收 | held-out 上协议合法率 > 95%；T1 准确率不低于零样本；截断率下降 |

E1 显示零样本协议合法率 ≥ 80% 时，可以跳过 SFT 直接 RL（[data_and_training.md](data_and_training.md) §2.7）。

---

## 6. RL（概要，详见 [data_and_training.md](data_and_training.md) §3）

- GRPO，G = 8，优势只减均值，token 级损失，clip 0.2 / 0.28；CLM 的双通道效率项只加在 KV 工具调用的 token 上；TIS 校正；动态采样；
- 阶段：RL-1 只有 T1（非思考模式，便宜）→ RL-2 是 T1 加 T2-A（思考模式），或者 §3.3 的备选任务；
- 变体：只能删、去掉效率项、去掉笔记、LRU 兜底、跳过 SFT；
- 起点超参（8B）：每步 16 题 × 8 条，学习率 1e-6（全参数）或 1e-5（LoRA），T1 约 200 步。

---

## 7. 引擎

### 7.1 参考实现：`PagedKV`（已完成）

`self_paging/hf/logstore.py`：每次前向前把驻留页的 KV 按日志顺序拼成 `DynamicCache`，新 token 用全局位置，并传一个显式的加性 4D 掩码。支持逻辑换页和物理换页。13 个测试全部通过。它有两个用途：
- E0–E2 和 SFT 数据生成直接用它；
- 作为 SGLang 版本的"标准答案"：一致性测试拿两者的 logprob 对比。

性能上，每次前向都重新拼一次 cache 是 O(驻留长度) 的拷贝，原型阶段可以接受；之后改成只在驻留集变化时重拼。

### 7.2 RL rollout：SGLang overlay（推荐）

dense 模型上更简单，不用处理 GDN 状态：
- **会话**：每个 episode 是一个会话，引擎保存日志中每个 token 的 KV 槽号、驻留的 token 区间和日志长度；会话的 KV 不进 radix 树、不参与淘汰（可以先看新版 SGLang 的 streaming session 能不能直接用 [S]）；
- **新回合**：请求只带新 token。把驻留 token 的槽号按日志顺序写进这个请求的 `req_to_token` 行，当作已缓存的前缀；
- **位置**：每个请求加 `pos_offset = log_len − L_res`，prefill 和 decode 两条路径都要改；
- **接口**：`POST /kv_set_resident {session_id, ranges}`。页、协议、LRU 的逻辑都放在客户端，引擎只认 token 区间；
- **写法**：仿照 `suffix_cache_reuse/overlay.py` 做 monkeypatch，不改 SGLang 源码；
- **一致性测试**：同一条轨迹，SGLang 返回的 logprob 和 `PagedKV` 的对比；换出再换回以后输出不变；逻辑模式和物理模式一致。

### 7.3 vLLM：dense 模型上可行，作为备选

v1 里最大的障碍（混合模型下 block 被放大到几百 token）在 Qwen3-8B 上不存在，block 是 16 token。仍然要改：

1. 会话保持：跨回合保留请求的 block；
2. 前缀哈希：把"写入时驻留集的签名"放进 block 哈希的 extra key，否则不同 rollout 会互相命中错误的 KV；
3. block table 过滤：每个请求只放驻留的 block，`seq_lens` 用驻留长度；
4. 位置和槽位分离：`GPUModelRunner` 准备输入时，`positions` 和 `slot_mapping` 要拆成两套计数器；
5. 页对齐：页边界必须落在 16 token 的边界上。系统插入的内容（工具结果、页头、仪表盘）前面用换行补齐；生成过程中的边界（思考和正文之间）补不了，所以 vLLM 版本里一个 assistant 回合只能是一页。

结论：能做，但比 SGLang 改动多，粒度也粗一些。SGLang 版本先跑通；如果服务器上只有 vLLM 的环境，再按这五条改。

---

## 8. 代码

### 8.1 已完成

| 文件 | 内容 | 测试 |
|---|---|---|
| `self_paging/core/pages.py` | 页表、驻留集版本、换出 / 换回 / 删除、钉住页、预算 | `test_illegal_actions` |
| `self_paging/hf/logstore.py` | `PagedKV` 参考引擎 | 等价性、物理换页、换回逐位相同 |
| `self_paging/hf/masks.py` | 可见性矩阵、加性掩码、FlexAttention `mask_mod` | mask_mod 和稠密矩阵一致；HF flex_attention 直接可用 |

### 8.2 接下来（按顺序）

| 文件 | 内容 | 能否在云端完成 |
|---|---|---|
| `core/protocol.py` | 页头、转义、仪表盘渲染；`kv_*` 工具的 schema；解析 Qwen3 的 `<tool_call>`；合法性检查；超预算截断 | 能（逐 token 比对模板的测试要在服务器上跑） |
| `core/trajectory.py` | 轨迹记录（token 进、token 出），存取 | 能 |
| `core/costs.py` | 代价 C 的两种折算 | 能 |
| `envs/t1_warehouse.py`、`envs/t1b_lookup.py` | 多轮环境、真值、规则读者、最小充分集（[data_and_training.md](data_and_training.md) §1.1） | 能 |
| `policies/` | Full、Recent、AttnEvict、QuestPage、Belady | 能 |
| `hf/runner.py` | agent 循环：采样解码、遇到 `<|im_end|>` 停、解析工具调用、执行 KV 工具和环境工具 | 能（用小模型测流程） |
| `experiments/e0_headroom.py`、`e1_zeroshot.py`、`e2_residue.py` | 三个继续/止损实验的驱动和打分 | 能写，要在服务器上跑 |
| `data/oracle_sft.py`、`data/relabel.py`、`data/reject_sample.py` | SFT 数据的三个来源 | 能写，要在服务器上跑 |
| `envs/t2_bcp.py`、`envs/t2_longmemeval.py` | 复用 ACM 的检索和评判（MIT，注明出处）；LongMemEval 的流式协议 | 能写，要在服务器上跑 |
| `train/sft.py`、`train/grpo.py`、`train/advantage.py` | 训练 | 能写，要在服务器上跑 |
| `sglang_overlay/` | §7.2 | 只能在 GPU 上测 |

---

## 9. 实验矩阵

| 实验 | 内容 | 对应 |
|---|---|---|
| **E0** | T1、T1b × 预算 {25, 50, 100}% × {Full, Recent, AttnEvict, QuestPage, Belady}，每格 200 个场景 | 提升空间 |
| **E0b** | BCP 和 LongMemEval 上，Full 用全局位置（有空洞）还是压紧位置 | 位置空洞的影响 |
| **E1** | 零样本协议：T1 100 条；BCP 训练 680 题 pass@8；LongMemEval 100 题 | 协议遵守；BCP 是否可行 |
| **E2** | T1 `explicit`：FULL / OUT-LATE / OUT-EARLY / REWRITE × 剂量 {1K, 4K, 16K} | K3 残留 |
| 表 1 | T1、T1b：零样本 / SFT / RL 对比全部基线 | 主结果 |
| 表 2 | BCP 评测 150 题 + LongMemEval-S：同上 | 主结果，K1 |
| 图 1 | 准确率对代价的 Pareto 曲线 | K1 |
| 图 2 | 可逆 vs 只能删：驻留大小、换回率、准确率；LongMemEval 上最明显 | K2 |
| 图 3 | 残留的剂量曲线，RL 前后 | K3 |
| 图 4 | 可预测性旋钮 vs 笔记占比和换回次数；LongMemEval 上笔记 vs 换回 | K4 |
| 表 3 | SGLang 物理换页：显存、吞吐、首字延迟，对比 CLM + SCR | 系统 |

K5（规模）需要其他尺寸（Qwen3-1.7B / 4B / 14B），作为可选项。

---

## 10. 算力（Qwen3-8B）

| 步骤 | 粗估 |
|---|---|
| E0–E2 | 单张 48 GB 以上的卡；T1 的日志 ≤ 32K，KV ≤ 4.6 GB。1–2 天 |
| E1 的 BCP pass@8（680 × 8） | rollout 是大头，几张卡 1–2 天 |
| SFT（约 5000 条 × 2 万 token × 2 epoch ≈ 2 亿 token） | 6 × 8.2e9 × 2e8 ≈ 1e19 FLOPs，按每卡 400 TFLOPS 有效算力约 7 个 GPU 小时（全参数还需要显存够） |
| RL-1（T1，200 步 × 128 条 × 约 3K 生成 token ≈ 7700 万 token） | 几张卡 1–2 天 |
| RL-2（BCP） | 最贵，T1 调稳后再做 |

## 11. 时间线

| 周 | 内容 | 在哪里 |
|---|---|---|
| 1 | §8.2 的前六项（协议、轨迹、代价、T1/T1b、基线和 Belady、HF 循环）+ E0–E2 脚本 | 云端 |
| 1 | 下载 Qwen3-8B；核对 config；跑模板逐 token 比对和真实权重上的等价性测试 | 用户服务器 |
| 2 | E0、E0b、E1、E2；做继续/止损判断 | 用户服务器 |
| 3–4 | SFT 数据和 SFT；SGLang overlay + 一致性测试 | 两边 |
| 5–8 | RL-1、RL-2、变体 | 用户服务器 |
| 9+ | LongMemEval 评测、系统评测、写作 | — |

## 12. 需要用户在服务器上做的事

1. 下载 Qwen3-8B，并核对 config：

```bash
python -c "import json;c=json.load(open('/path/to/Qwen3-8B/config.json'));print({k:c.get(k) for k in ['num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','max_position_embeddings','rope_theta','rope_scaling']})"
```

2. 查 GPU 型号和数量：

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

3. 查软件版本：

```bash
pip list 2>/dev/null | grep -iE "^(torch|transformers|sglang|vllm|flash_attn|flashinfer-python) "
```

4. 在服务器上跑一遍 `self_paging/tests`，确认 GPU 和服务器上的 transformers 版本下也能通过；
5. 确认能否下载 BCP 的语料、索引和 evidence 标注，以及 LongMemEval-S；服务器上有没有 JDK 21（pyserini 需要）。
