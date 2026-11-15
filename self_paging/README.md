# self_paging：代码文档

新主线"让模型自己管理 KV 缓存"的全部代码。本文说明**每个文件做什么、主要接口、输入输出、数据格式、测试状态**。

- 要跑实验，读 [`docs/runbook.md`](../docs/runbook.md)：准备什么、实验顺序、每步的验收标准、剩下的开发任务；
- 方法设计见 `docs/research_line.md`、`docs/experiment_plan.md`（v2，Qwen3-8B）、`docs/data_and_training.md`、`docs/algorithm_design.md`。

---

## 1. 核心概念（读代码前先看）

| 概念 | 含义 | 在代码里 |
|---|---|---|
| **日志** | 所有 token 按顺序只追加；每个 token 的 KV 只在写入时算一次，之后不重算 | `TokenLog.tokens`；token 的全局位置 = 它在日志里的下标 |
| **页** | 日志被切成页：system、任务、每个工具结果的正文、每个 assistant 回合、每条笔记、页头/页尾、仪表盘、回执 | `core/pages.py` 的 `Page` |
| **钉住** | system、任务、页头、页尾、块的开闭标记永远驻留，不能换出 | `Page.pinned` |
| **驻留集 R** | 当前在"显存"里、注意力能看到的页。只有它会变 | `PageTable.snapshots[-1]` |
| **版本** | 每次换出 / 换回 / 删除，驻留集就生成一个新版本；每个 token 记下它写入时的版本号 | `TokenLog.seg_of` |
| **可见性规则** | token i 能看 token j，当且仅当 j ≤ i，并且 j 和 i 在同一页，或者 j 所在的页在 i 写入时的驻留集里 | `hf/masks.py` 的 `visibility` |
| **P3（核心性质）** | 推理时逐步换页的结果，等于训练时整条日志加上面这个掩码一次前向的结果 | `tests/test_equivalence.py`、`tests/test_pipeline.py` |
| **预算 B** | 驻留 token 数的上限（含钉住部分） | `EpisodeConfig.budget` |
| **超预算** | 新的工具输出被截断；只有连一个最小的 assistant 回合都放不下时，系统才按"最早的页"强制换出，并计数 | `agent/episode.py` |

动作接口是原生 tool call：`kv_out(pages)`、`kv_in(pages)`、`note(text)`，不可逆对照里换成 `kv_drop(pages)`。所有 KV 动作在回合之间生效：同一回合里先执行 KV 工具，再执行环境工具。

---

## 2. 目录和每个文件

### 2.1 `core/`：不依赖模型的部分

| 文件 | 做什么 | 主要接口 |
|---|---|---|
| `pages.py` | 页表：页的元数据、驻留集的版本历史、换出 / 换回 / 删除的合法性检查（钉住页、正在写的页、位置不对、超预算会抛 `PagingError`） | `PageTable.open / grow / page_out / page_in / drop / resident / resident_tokens / in_order`；`KINDS` 是合法的页类型 |
| `log.py` | `TokenLog`：记账。记录每个 token 的 id、页号、版本号和峰值驻留量；导出可 JSON 序列化的记录。造 SFT 数据时用它，不需要模型 | `open / feed / page_out / page_in / drop / record() / trajectory()` |
| `protocol.py` | Qwen3 格式的渲染和解析：system 加工具签名、用户轮、assistant 前缀（非思考模式会加空的 `<think>` 块）、工具结果块的各个片段、页头 `[[PAGE n …]]`、仪表盘 `[KV] …`、转义（防止工具输出伪造页头或对话结构）、`<tool_call>` 解析、页号解析（"3,5-8"）；`Tok` 分词器适配器 | `render_system / render_user / assistant_prefix / block_open / response_header / response_trailer / block_close / gauge / escape / parse_tool_calls / parse_pages / kv_tool_schemas / Tok` |
| `costs.py` | 每个 token 能看到多少个 key（O(T)），以及代价 C = Σ(κ + 可见 key 数)；κ 按服务时的 batch 折算（Qwen3-8B：batch 1 约 11 万，batch 32 约 3.5K） | `visible_counts(rec) / episode_cost(rec, batch) / kappa(batch)` |

### 2.2 `hf/`：在 HuggingFace 模型上的实现

| 文件 | 做什么 | 主要接口 |
|---|---|---|
| `logstore.py` | `PagedKV(TokenLog)`：**参考推理引擎**。每次前向时，把驻留页的 KV 按日志顺序拼成 `DynamicCache`（驻留集不变就复用上一次的），新 token 用全局位置，并传一个显式的 4D 掩码。`page_out(physical=True)` 会把 KV 真的挪到内存（CUDA 上用锁页内存） | `feed(pid, ids) -> logits[n, vocab]`；其余接口同 `TokenLog` |
| `masks.py` | 训练用的掩码：驻留矩阵、可见性矩阵、HF 用的加性 4D 掩码、FlexAttention 的 `mask_mod`（支持多条轨迹打包） | `resident_matrix / visibility / additive_mask / make_mask_mod` |

### 2.3 `envs/`：环境

所有环境都满足同一个协议：属性 `system`、`task`、`tools`、`done`；`step(name, args) -> (text, meta, done)`，`meta["env_page"]` 标记这段文本是一个环境页；`score() -> float`，作为奖励。

| 文件 | 做什么 |
|---|---|
| `t1_warehouse.py` | **T1 多轮仓库环境**。复用 `state_study/probes/warehouse.py` 的世界模拟器生成事件流，切成页，其中埋几个决策点。每个决策都依赖更早的一个关键事件：审计更正、盘点、或者后来才生效的隔离通知。工具有两个：`next_events()` 取下一页，`store(shelf)` 回答当前决策。环境总是按规则正确地存货，所以每一页的内容由种子唯一确定。<br>还包括：`rule_read(texts)` 是规则读者，只看文本算出答案；`minimal_sufficient(...)` 求最小充分集，反复删页直到删不动为止；`state_note(texts)` 生成规则读者能读懂的状态笔记；`generate(seed, cfg)` 生成 `Episode`（含 `pages`、`decisions`、`need`）；`T1Config` 是难度参数 |
| `t2_bcp.py` | **BrowseComp-Plus 环境**。工具 `search(query)`、`get_document(docid)`、`answer(answer)`；片段和文档长度沿用 ACM 的设置；列出多个候选答案直接判错。检索器和评判是注入的：`PyseriniSearcher(index_path)` 跑 BM25（需要 pyserini 和 JDK 21）；`OpenAIJudge(base_url, model)` 调本地 OpenAI 兼容服务，评判模板逐字取自 ACM（MIT）；`load_split(path)` 读 ACM 的划分文件 |

### 2.4 `oracle/`：生成监督信号

| 文件 | 做什么 |
|---|---|
| `t1_oracle.py` | `T1Oracle(mode)`：T1 上的 oracle 策略，返回每一回合的完整文本（理由 + 工具调用）。作答前逐个货架列出状态；换出时按页的内容写明原因。实测质量见 `docs/data_and_training.md` §2.8。<br>取下一页之前，用按页大小加权的在线 Belady 腾出空间：先换出以后再也用不到的页，再换出下次使用最远的页；assistant 回合排在环境页前面换出。决策前，把规则读者需要的页换回来。<br>四种模式：`paging`；`notes`（先写状态笔记再换出）；`eager`（每回合都换出当前状态用不到的页，需要时再换回，专门产出 `kv_in` 样本）；`mixed`。换出时优先选不改变当前状态读数的页；决策时如果可见集合会误导，就收缩到最小充分集。<br>每次换出都在 meta 里标注是"可观察"的（从已看到的内容就能判断）还是"依赖未来信息"的；答案在可见文本里找不到依据时记 `oracle_unsupported` |
| `hinted_teacher.py` | `HintedTeacher(model, tok, oracle)`：动作完全照抄 oracle，前面的理由由真实模型写。模型能看到驻留页的文本和一段它不该提到的提示；理由泄漏提示或为空时退回模板，记 `teacher_fallbacks` |

### 2.5 `agent/`：episode 循环

| 文件 | 做什么 |
|---|---|
| `episode.py` | `run_episode(env, log, policy, tok, cfg, pager=None) -> record`：造数据、RL 采样、评测共用这一个循环。每回合依次是：系统换页器（如果有）→ 最小空间检查 → assistant 页（前缀 + 策略生成 + `<|im_end|>\n`）→ 解析工具调用并标出 KV 调用的 token → 先执行 KV 工具、再执行环境工具 → 写工具结果块（页头钉住、正文可换出、页尾钉住；回执和旧仪表盘由系统换出；正文超预算就截断）。<br>`EpisodeConfig`：预算、最大回合数、单回合最大生成长度、是否思考、可逆/不可逆、是否有笔记工具、是否提供 KV 工具、最多允许几次非法调用等。<br>`ModelPolicy`：逐 token 采样，记录行为 log-prob。`ScriptedPolicy(fn)`：强制输出 `fn(state)` 返回的文本；日志带模型时，也记录这些 token 的 log-prob |
| `pagers.py` | 系统换页基线，运行时不提供 KV 工具，模型只做任务。`RecentPager`：只留最近的页。`BeladyPager`：用未来信息换页，是 T1 上的上界 |

### 2.6 `data/`：造训练数据

| 文件 | 做什么 |
|---|---|
| `build_t1_sft.py` | 用 oracle 造 T1 的 SFT 数据。每个种子在多个预算下各跑一次（1.0 表示不设预算，就是"不换页"样本），按换页、笔记两种模式。质量门槛：答对；没有非法调用、截断、强制换出、KV 报错；不超预算；答案有依据。可选 `--max-future-share` 过滤掉主要靠未来信息的 episode；`--teacher` 用真实模型写理由。`t1_budget(env, tok, ratio)` 定义预算 |
| `reject_sample.py` | 预算压力下的拒绝采样：用当前模型采样，只留成功且干净的，每个种子最多留几条。输出格式和 oracle 数据相同 |

### 2.7 `train/`：训练

| 文件 | 做什么 |
|---|---|
| `batch.py` | `pack(records, advantages)`：多条轨迹拼成一条序列，页号和版本号改成全局编号，驻留矩阵按块对角拼接，用 `doc_of` 隔开轨迹，位置号每条轨迹从 0 开始。`attention_mask(...)`：CPU 上用稠密加性掩码，GPU 上用 FlexAttention 的 BlockMask。`token_logprobs(model, packed, attn)`：一次前向算出每个 token 的 log-prob；LM head 和 log-softmax 分块并做 checkpoint，显存随序列长度线性增长 |
| `common.py` | 加载模型（指定注意力实现；可选 LoRA、gradient checkpointing、YaRN；可以直接读 LoRA 目录）、FSDP2 切分、读写 JSONL、cosine 学习率、加载分词器（`toy` 表示测试分词器）、`tiny_qwen3()` 随机小模型 |
| `sft.py` | SFT：按 token 数切 micro-batch，只对 assistant 生成的 token 算损失。单卡直接跑；torchrun 下全参数用 FSDP2，LoRA 每卡一份副本、对 adapter 梯度做 all-reduce。保存为 HF 格式 |
| `advantage.py` | 组内优势 `r − mean`（不除以标准差）；CLM 的效率项只在组内成功的轨迹之间按代价比较，只加在 KV 工具调用的 token 上；`informative()` 用于动态采样，奖励全相同的组丢弃 |
| `grpo.py` | GRPO 主循环：采题（种子 + 预算比例）→ 每题 G 次 rollout（`PagedKV` + `ModelPolicy`，rollout 时切到 sdpa）→ 算奖励和代价 → 算优势 → 丢掉无信息的组 → 切回 flex，算 token 级裁剪损失，乘截断重要性权重 → 更新。torchrun 下每卡一份副本（8B 用 LoRA），梯度 all-reduce。`--save-every K` 保存中间 checkpoint。`rollout_fn` 参数留给以后接 SGLang |

### 2.8 `eval/`

| 文件 | 做什么 |
|---|---|
| `t1_eval.py` | T1 评测，五种方法：`model`（模型自己管 KV）、`full`（不设预算）、`recent`、`belady`（系统换页）、`oracle`（健全性检查，应该是 1.0）。每个 episode 一行 JSONL（奖励、代价、峰值驻留、各种计数），结束时打印按"方法 @ 预算"汇总的结果 |

### 2.9 `testing/` 和 `tests/`

| 文件 | 做什么 |
|---|---|
| `testing/toy_tokenizer.py` | 字符级分词器，带 Qwen3 的特殊 token；云端测试和冒烟运行用 |
| `tests/test_equivalence.py` | P1–P3：在 Qwen3 结构上，逐步换页推理等于加掩码的一次前向；物理换页逐位相同；FlexAttention 掩码一致；非法动作会报错 |
| `tests/test_pipeline.py` | 协议解析；T1 环境（规则读者、最小充分集）；oracle 数据；代价和可见 key 数一致；**rollout log-prob 等于训练端 log-prob**；SFT 能过拟合；GRPO 的更新方向正确；Belady 换页器 |
| `tests/test_t2_and_teacher.py` | BCP 环境（假检索器、假评判）；特权提示老师 |
| `tests/test_template_server.py` | 我们的渲染和官方 `apply_chat_template` 逐字比较，并检查分段分词等于整段分词。需要设置 `QWEN3_TOKENIZER`，**在服务器上跑** |

---

## 3. 数据格式：episode 记录（JSONL 一行一条）

SFT 数据、拒绝采样数据、RL rollout、评测都用这一个格式，由 `run_episode` 产生：

| 字段 | 类型 | 含义 |
|---|---|---|
| `tokens` | int[T] | 日志里的全部 token id，**按喂入时的原样记录，绝不重新分词** |
| `page_of` | int[T] | 每个 token 的页号 |
| `seg_of` | int[T] | 每个 token 写入时的驻留集版本号 |
| `snapshots` | int[][] | 第 v 个版本的驻留页号列表 |
| `pages` | obj[] | `{pid, kind, start, length, pinned}` |
| `loss_mask` | 0/1[T] | 1 = 策略生成的 token（训练目标） |
| `ctrl_mask` | 0/1[T] | 1 = KV 工具调用（`kv_out` / `kv_in` / `note` / `kv_drop`）里的 token |
| `logprobs` | float[T] | 生成 token 的行为 log-prob（RL 用；oracle 数据里是 0） |
| `reward` | float | 环境的 `score()` |
| `budget` | int / null | 预算 B |
| `peak_resident` | int | 峰值驻留 token 数 |
| `env_page` | {str: int} | 环境页号 → 日志页号 |
| `meta` | obj | 计数：`turns`、`invalid`、`truncated_tokens`、`forced_evictions`、`kv_errors`、`kv_out`、`kv_in`、`notes`、`over_budget`、`observable_evictions`、`future_informed_evictions`、`oracle_unsupported`、`teacher_fallbacks`；以及 `seed`、`ratio`、`mode`、`source` |
| `cost` | float | 只在 GRPO 的 rollout 里有：`episode_cost` 算出的代价 |

---

## 4. 测试

```bash
uv venv -p 3.11 <venv> && . <venv>/bin/activate
uv pip install torch          # 服务器装 CUDA 版；云端用 --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest transformers peft
python -m pytest -q -p no:cacheprovider self_paging/tests
QWEN3_TOKENIZER=/path/to/Qwen3-8B python -m pytest -q self_paging/tests/test_template_server.py
```

2026-11-15 在云端（torch 2.15.0+cpu、transformers 5.23.0）：35 个通过、1 个跳过（模板比对）。

**版本要求**：代码用了 transformers 5 的接口：`DynamicCache.layers[l].keys` / `.values`、`model.set_attn_implementation()`，以及前向不再传 `cache_position`。服务器上建议直接装 5.23；4.x 的旧版本大概率不兼容。FlexAttention 需要 torch ≥ 2.5。

---

## 5. 已知限制

- **rollout 慢**：`PagedKV` 逐 token 解码，一次只跑一条序列。T1 规模可以接受；BCP 规模要等 SGLang overlay（runbook 的开发任务 D6）；
- **多卡路径没在 GPU 上跑过**：SFT 的 FSDP2、SFT 和 GRPO 的 LoRA 梯度 all-reduce；
- **模板没在真实分词器上核对过**：`test_template_server.py` 必须先过；
- **`t1_budget` 的固定开销是按字符级分词器调的**，对 Qwen3 分词器偏保守（预算偏大），可以按实际 token 数调；
- **YaRN 参数**写的是 `config.rope_scaling = {"rope_type": "yarn", "factor": …}`。transformers 5 可能改用别的字段名；第一次用 `--yarn` 时要确认 `model.config` 里真的生效了。T1 用不到 YaRN；
- **代价常数**（权重 16.4 GB、KV 144 KB/token）是凭已有知识写的 [K]，要按 Qwen3-8B 的实际 config 核对。
