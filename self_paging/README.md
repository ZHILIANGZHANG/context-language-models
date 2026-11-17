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
| `costs.py` | 每个 token 能看到多少个 key（O(T)；token 级基线扣掉隐藏的 token），以及代价 C = Σ(κ + 可见 key 数)；κ 按服务时的 batch 折算（Qwen3-8B：batch 1 约 11 万，batch 32 约 3.5K） | `visible_counts(rec) / episode_cost(rec, batch) / kappa(batch)` |

### 2.2 `hf/`：在 HuggingFace 模型上的实现

| 文件 | 做什么 | 主要接口 |
|---|---|---|
| `logstore.py` | `PagedKV(TokenLog)`：**参考推理引擎**。每次前向时，把驻留页的 KV 按日志顺序拼成 `DynamicCache`（驻留集不变就复用上一次的），新 token 用全局位置，并传一个显式的 4D 掩码。`page_out(physical=True)` 会把 KV 真的挪到内存（CUDA 上用锁页内存）。**token 级保留掩码（D10）**：`set_keep(keep[层, KV 头, 位置])`，token 可见 = 所在页驻留且该层该头保留；每个（层, 头）保留数必须相同，这样拼出的 cache 形状一致；页级换入换出让各头不等时，系统隐藏多出的最旧 token（`keep_equalized`）；`resident_tokens()` 只算保留的；`record_keep=True` 时可用 `visibility_parts` / `visibility_rows` / `token_visibility` 重建逐层逐头可见性；记录里的 `hidden_pages` 供代价计算 | `feed(pid, ids) -> logits[n, vocab]`；`set_keep`、`keep_view`、`visible_positions`；其余接口同 `TokenLog` |
| `masks.py` | 训练用的掩码：驻留矩阵、可见性矩阵、HF 用的加性 4D 掩码、FlexAttention 的 `mask_mod`（支持多条轨迹打包） | `resident_matrix / visibility / additive_mask / make_mask_mod` |
| `token_probe.py` | `ProbedPagedKV(PagedKV)`：token 级基线用的观测，不改任何注意力输出。记录每层最近 `window` 个 token 的 query（过了 norm 和 RoPE）；`accumulate=True` 时把每个 query 对每个可见 key 的注意力累加到 `acc[层, KV 头, 位置]`（H2O）；`window_attention(layer)`：窗口 query 对现在可见的 key 的注意力（TOVA、SnapKV）；`all_keys(layer)`：所有驻留 token 的 key（Quest 的块上下界）。测试里和单次前向的注意力逐项对上 | `window_attention / visible_keys / all_keys / acc` |

### 2.3 `envs/`：环境

所有环境都满足同一个协议：属性 `system`、`task`、`tools`、`done`；`step(name, args) -> (text, meta, done)`，`meta["env_page"]` 标记这段文本是一个环境页；`score() -> float`，作为奖励。

| 文件 | 做什么 |
|---|---|
| `t1_warehouse.py` | **T1 多轮仓库环境**。页头摘要：`page_digest(text)` 生成 `steps a-b | quarantine S5 | store S9 …`，`T1Config.digest`（默认开）控制是否放进页头；`digest_shelves` 用于判断换回是否可实现（`docs/rl_design.md` §2）。复用 `state_study/probes/warehouse.py` 的世界模拟器生成事件流，切成页，其中埋几个决策点。每个决策都依赖更早的一个关键事件：审计更正、盘点、或者后来才生效的隔离通知。工具有两个：`next_events()` 取下一页，`store(shelf)` 回答当前决策。环境总是按规则正确地存货，所以每一页的内容由种子唯一确定。<br>还包括：`rule_read(texts)` 是规则读者，只看文本算出答案；`minimal_sufficient(...)` 求最小充分集，反复删页直到删不动为止；`state_note(texts)` 生成规则读者能读懂的状态笔记；`generate(seed, cfg)` 生成 `Episode`（含 `pages`、`decisions`、`need`）；`T1Config` 是难度参数 |
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
| `pagers.py` | 系统换页基线，运行时不提供 KV 工具，模型只做任务。`RecentPager`：只留最近的页。`BeladyPager`：用未来信息换页，是 T1 上的上界；和 oracle 一样不会拆散成对的记录。`rule_task_policy`：用规则读者作答、从不碰 KV，和换页器配合，可以在不用模型的情况下判断题目可不可行、换页敏不敏感 |
| `token_pagers.py` | **token 级系统基线（D11）**，和页级换页器一样在回合之间调用、严格守预算：`StreamingLLMPager`（sink + 最近）、`H2OPager`（累计注意力 + 最近窗口，逐层逐头）、`TOVAPager`（最新 query 的注意力，逐层，头平均）、`SnapKVPager`（观察窗口投票 + 一维池化，逐层逐头）、`QuestPager`（16-token 块的 key 上下界，逐层逐头，**可逆**，每次重选）。打分函数是纯函数，可以单独测。`protect_pinned=True`（默认）保留钉住的结构页，和我们的预算口径一致，是可调超参。和论文不同的地方写在文件开头。`make_token_log(method, model, budget)` 给出对应的 `PagedKV`。**注意**：`rule_task_policy` 按页读文本，看不到 token 级隐藏，所以 token 级基线只能用模型评测 |

### 2.6 `data/`：造训练数据

| 文件 | 做什么 |
|---|---|
| `build_t1_sft.py` | 用 oracle 造 T1 的 SFT 数据。默认：四种模式（paging、notes、eager、mixed）；每个种子随机难度（`--vary`）；以 15% 的概率加入失误后恢复的样本（`--recovery`，失误回合不算损失）。每个种子在多个预算下各跑一次（1.0 表示不设预算，就是"不换页"样本），按换页、笔记两种模式。质量门槛：答对；没有非法调用、截断、强制换出、KV 报错；不超预算；答案有依据。可选 `--max-future-share` 过滤掉主要靠未来信息的 episode；`--teacher` 用真实模型写理由。`t1_budget(env, tok, ratio)` 定义预算 |
| `build_rl_pool.py` | RL 题库。`build`：不需要模型，丢掉预算不可行的题，换页敏感的题加权；`profile`：用策略模型测通过率，只留 0.1–0.9 之间的；`eval`：固定的分布内、分布外评测题库。格式：每行 `{env, seed, ratio, t1_config, budget, n_pages, n_decisions, rule_belady, rule_recent, sensitivity, oracle_ok, pass_rate, weight}` |
| `rl_preparation.py` | 服务器 agent 的 CPU 题库准备（划分、每个世界的难度、参考方法判可行、按预算 × 决策点数选 5000 题）。OOD 用 `ood_config_v2`（7–9 个决策点，放得进 32K；`config(seed, True, ood_version=1)` 是旧版） |
| `rl_strata.py` | 按决策类型分层（不需要模型和分词器）：`annotate` 给题目加 `n_delayed`、`n_long_delayed`、`max_gap_pages`、`stratum`（long_delayed / delayed / no_delayed）和每个决策的关键事件距离；`select_stratified` 在原有配额上保证 delayed 决策 ≥ 40%、远距离题 ≥ 35%；`select_profile_stratified` 选按"预算 × 分层"均衡的 64 题画像集。命令行读 `cpu-profile.jsonl`，见 runbook v2.5 |
| `inspect_sft.py` | 数据质量和多样性报告：按预算 × 模式统计换出、换回、反复换回的页、换出多久后换回；动作组合种类；episode 形态；恢复样本数（`docs/data_and_training.md` §2.8–§2.9） |
| `reject_sample.py` | 预算压力下的拒绝采样：用当前模型采样，只留成功且干净的，每个种子最多留几条。输出格式和 oracle 数据相同 |

### 2.7 `train/`：训练

| 文件 | 做什么 |
|---|---|
| `batch.py` | `pack(records, advantages)`：多条轨迹拼成一条序列，页号和版本号改成全局编号，驻留矩阵按块对角拼接，用 `doc_of` 隔开轨迹，位置号每条轨迹从 0 开始。`attention_mask(...)`：CPU 上用稠密加性掩码，GPU 上用 FlexAttention 的 BlockMask。`token_logprobs(model, packed, attn)`：一次前向算出每个 token 的 log-prob；LM head 和 log-softmax 分块并做 checkpoint，显存随序列长度线性增长 |
| `common.py` | 加载模型（指定注意力实现；可选 LoRA、gradient checkpointing、YaRN；可以直接读 LoRA 目录）、FSDP2 切分、读写 JSONL、cosine 学习率、加载分词器（`toy` 表示测试分词器）、`tiny_qwen3()` 随机小模型 |
| `sft.py` | SFT：按 token 数切 micro-batch，只对 assistant 生成的 token 算损失。单卡直接跑；torchrun 下全参数用 FSDP2，LoRA 每卡一份副本、对 adapter 梯度做 all-reduce。保存为 HF 格式 |
| `advantage.py` | **reward v2**：`hygiene_penalty`（强制换出、截断页、非法调用、KV 错误的小额惩罚，总上限 −0.25）、`counterfactual_credit`（用规则读者找出导致答错的那次 `kv_out` 和起作用的 `kv_in`，把 ±c 加在那一回合的控制 token 上）、`advantages_v2`、`has_signal`（`docs/rl_design.md` §4）。旧的： `r − mean`（不除以标准差）；CLM 的效率项只在组内成功的轨迹之间按代价比较，只加在 KV 工具调用的 token 上；`informative()` 用于动态采样，奖励全相同的组丢弃 |
| `grpo.py` | GRPO 主循环：采题（种子 + 预算比例）→ 每题 G 次 rollout（`PagedKV` + `ModelPolicy`，rollout 时切到 sdpa）→ 算奖励和代价 → 算优势 → 丢掉无信息的组 → 切回 flex，算 token 级裁剪损失，乘截断重要性权重 → 更新。torchrun 下每卡一份副本（8B 用 LoRA），梯度 all-reduce。`--save-every K` 保存中间 checkpoint。`--pool` 时由 `sampler.py` 的 `PromptSampler` 抽题；`--ratio-start 0.25:30` 是预算课程；`--no-adaptive`、`--explore` 控制自适应。`rollout_fn` 参数留给以后接 SGLang。reward v2 相关参数：`--reward v2`（默认）、`--w-credit`、`--w-eff-warmup`、`--w-eff-ramp`；另有 `--pool`（每行含 seed、ratio、t1_config）、`--no-digest` |
| `sampler.py` | GRPO 的抽题器：题库权重 × 桶的信号（桶 = 预算, 决策点数, delayed 数；信号 = 组内"任务分 + 卫生"标准差的滑动平均，只有信用或效率差的组算弱信号）；10% 的抽题只看题库权重；`ratio_start` 课程。旧题库缺 `n_delayed` 时启动时自动标注 |

### 2.8 `eval/`

| 文件 | 做什么 |
|---|---|
| `t1_eval.py` | T1 评测，五种方法：`model`（模型自己管 KV）、`full`（不设预算）、`recent`、`belady`（系统换页）、`oracle`（健全性检查，应该是 1.0）。每个 episode 一行 JSONL（奖励、代价、峰值驻留、各种计数），结束时打印按"方法 @ 预算"汇总的结果。`--pool` 直接读评测题库 |
| `control_eval.py` | 正式的 E0 系统基线评测（服务器 agent 写的，贪心解码，按世界存档、可续跑）。方法：`full`、`recent`、`belady`、页级近似 `quest` 和 `attn_evict`、`model`，以及 token 级的 `streaming`、`h2o`、`tova`、`snapkv`、`quest_token`。token 级超参用 `--pager-kwargs '{"snapkv": {"kernel": 7}}'`，观测窗口用 `--window`；只在开发集上调参 |
| `token_sanity.py` | D11 的验收：单轮大海捞针（问题在最后）。`oneshot` 是论文设定：整段 prefill，以问题为观察窗口压缩一次，再解码，预期 SnapKV、TOVA、Quest 远好于 StreamingLLM，H2O 居中；`stream` 是我们的协议：预算严格，逐页压缩，预期不可逆方法跌到“针还在”的比例，Quest 能恢复 |

### 2.9 `testing/` 和 `tests/`

| 文件 | 做什么 |
|---|---|
| `testing/toy_tokenizer.py` | 字符级分词器，带 Qwen3 的特殊 token；云端测试和冒烟运行用 |
| `tests/test_equivalence.py` | P1–P3：在 Qwen3 结构上，逐步换页推理等于加掩码的一次前向；物理换页逐位相同；FlexAttention 掩码一致；非法动作会报错 |
| `tests/test_pipeline.py` | 协议解析；T1 环境（规则读者、最小充分集）；oracle 数据；代价和可见 key 数一致；**rollout log-prob 等于训练端 log-prob**；SFT 能过拟合；GRPO 的更新方向正确；Belady 换页器 |
| `tests/test_t2_and_teacher.py` | BCP 环境（假检索器、假评判）；特权提示老师 |
| `tests/test_template_server.py` | 我们的渲染和官方 `apply_chat_template` 逐字比较，并检查分段分词等于整段分词。需要设置 `QWEN3_TOKENIZER`，**在服务器上跑** |
| `tests/test_rl_sampling_v2.py` | RL 抽题：决策类型标注、分层选题的占比和配额、无法达标时如实报告、自适应抽题器的降权和 10% 探索、课程、GRPO 抽题循环、OOD v2 比 v1 短 |
| `tests/test_token_baselines.py` | D10：全真掩码逐位相同；任意逐层逐头掩码（加上页级换入换出）下，逐步推理等于单次前向；校验和自动拉平。观测：H2O 的累计注意力、窗口注意力等于参考前向的注意力。D11：各打分函数用手工构造的张量检查；五种方法在 T1 回合里守住预算，记录的 log-prob 等于单次前向；只有 Quest 会换回；各方法选的 token 不同，粒度符合设计；代价只算可见 token；`token_sanity` 两种模式能跑 |
| `tests/test_rl_preparation.py` | 服务器 agent 的题库准备；OOD v2 的完整参考在 32K 以内。需要 `QWEN3_TOKENIZER`（2026-11-17 已在云端用 ModelScope 下载的分词器跑通） |

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

2026-11-16 在云端（torch 2.15.0+cpu、transformers 5.23.0）：38 个通过、1 个跳过（模板比对）。

**版本要求**：代码用了 transformers 5 的接口：`DynamicCache.layers[l].keys` / `.values`、`model.set_attn_implementation()`，以及前向不再传 `cache_position`。服务器上建议直接装 5.23；4.x 的旧版本大概率不兼容。FlexAttention 需要 torch ≥ 2.5。

---

## 5. 已知限制

- **rollout 慢**：`PagedKV` 逐 token 解码，一次只跑一条序列。T1 规模可以接受；BCP 规模要等 SGLang overlay（runbook 的开发任务 D6）；
- **多卡路径没在 GPU 上跑过**：SFT 的 FSDP2、SFT 和 GRPO 的 LoRA 梯度 all-reduce；
- **模板没在真实分词器上核对过**：`test_template_server.py` 必须先过；
- **`t1_budget` 的固定开销是按字符级分词器调的**，对 Qwen3 分词器偏保守（预算偏大），可以按实际 token 数调；
- **YaRN 参数**写的是 `config.rope_scaling = {"rope_type": "yarn", "factor": …}`。transformers 5 可能改用别的字段名；第一次用 `--yarn` 时要确认 `model.config` 里真的生效了。T1 用不到 YaRN；
- **代价常数**（权重 16.4 GB、KV 144 KB/token）是凭已有知识写的 [K]，要按 Qwen3-8B 的实际 config 核对。
