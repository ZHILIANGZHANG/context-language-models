# 研究日志与已验证事实

本文档按时间记录这个课题的决策过程，并汇总所有已验证的事实、纠正过的错误、代码和基准的可用性、冒烟测试结果和环境限制。目的是让新开的会话不用重新调研。

- 主线（2026-11-15 起）：[research_line.md](research_line.md)，让模型自己管理 KV 缓存
- 算法与实验设计：[algorithm_design.md](algorithm_design.md)
- 旧主线"状态 vs 历史"已归档：[`archive/state_vs_history/`](archive/state_vs_history/README.md)。本文 §2–§5 和 §7.2 主要是旧主线阶段的记录，事实仍然有效
- 相关工作全集：[related_work.md](related_work.md)
- CLM 笔记：[clm_notes.md](clm_notes.md)
- idea 库：[idea_bank.md](idea_bank.md)

---

## 1. 时间线

### 2026-10-02

| 步骤 | 用户的问题 | 结论 |
|---|---|---|
| 1 | 这个仓库和论文在做什么、必须训练吗、怎么 follow | CLM 把上下文变成文件让模型用 bash 改；不必训练（zero-shot → ICL → RL 三层递进）。见 [clm_notes.md](clm_notes.md) |
| 2 | 这是纯 agent 任务还是新风格的 harness | 本质是新风格 harness + 让模型学会用它的训练方法；harness 只管机制，模型定策略。名字叫"Language Model"是想讲一种模型能力，RL 那一步才真正把它训进权重 |
| 3 | 想 follow 发一篇，有哪些有 insight 的切入点（RSI、记忆细粒度管理、安全审计、特殊场景） | 提出 A1–A6：安全（数据洗成模型想法，已复现角色头伪造）、回归先验、策略也是文件、存储引擎视角、监督、其他。见 [idea_bank.md](idea_bank.md) A 组 |
| 4 | 想 ambitious 一点，打 general bench | 提出 B1"上下文管理是 test-time scaling 的第三条轴"。下载 Harbor 注册表，80 个数据集 |
| 5 | 想用迭代后的 CLM harness 在传统 harness 做不好的 bench 上赢 | 提出 B2"CLM v2 五个原语"，按传统 harness 的失败模式选 bench |
| 6 | 转来另一份"工作记忆"分析，要求先确认相邻工作和可用基准 | 核实那份分析（见 §2.1）；结论是"工作记忆"已非常挤；提出 B4"Forget the trace, keep the lesson" |
| 7 | 系统梳理 CLM 和相关工作 | 五把尺子 + 三个竞争假说（动作空间太窄 / 模型看不见自己 / 策略必须学 / 策略应由外循环搜） |
| 8 | 聚焦"状态派"那一波（Scroll、SKILL.state、PoS、VISTA…）和趋势 | 趋势：从管理历史到管理状态；记录和视图分离；状态写成程序；不确定性；校验与恢复；harness 变薄 |
| 9 | 这个方向能快速占坑的切入点 | 推荐"早绑定 vs 晚绑定"（C1）。当时的引子"状态 + 历史并存最差"后来发现是误读（§2.2） |

### 2026-10-03

| 步骤 | 用户的问题 | 结论 |
|---|---|---|
| 10 | 哪些开源、哪些 bench 能用、哪些 idea 真实可推进 | 四个调研子 agent 核查了代码和基准（§4）；发现 delayed-relevance 复现项目抢先测了延迟相关和事后更正；首推改为"可验证、事务化的状态"（C4） |
| 11 | 提炼核心趋势 | 记忆单位从对话记录换成状态；四个新问题：应该知道什么、放在哪、谁来写、写错了怎么办 |
| 12 | 基于已有 paper 和能用的 bench，想几个切实可行的 idea | C4 撞车（MemTX、PatchOptic、PatchBoard 等），降级；提出 I1–I4，**用户确认** |
| 13 | 把仓库改成这个研究方向的完整工作区，写详细 proposal | 15 个 submodule 固定版本接入 `third_party/`；写 `docs/proposal.md`、`docs/literature_map.md`。提交 `b7d5ca0` |
| 14 | 这些都能在云端验证吗 | 不能全部：没有模型 API 密钥，多数模型服务域名被拦（§6） |
| 15 | 没有 API，可以调用自己的低端模型跑验证吗 | 可以用 Claude 子 agent 中转做冒烟测试，但不能当实验数据；**不会**拿本会话自己的凭证去调 API |
| 16 | 把能跑的验证跑了，给一个验证可行、值得赶紧做完的 idea | 实现仓库探针 + 7 种条件 + 测试，跑 Haiku 冒烟测试（36 题）。推荐"作废标记 vs 状态"（实为 I3）。提交 `a264d57` |
| 17 | 之前确认的那几个 idea 呢？验证的不是它们 | 澄清"作废标记"就是 I3；补做 I1、I2、I4 的验证：PoS 装好、ALFWorld 真值、召回条件、三档模型。提交 `e2543bf` |
| 18 | I3 跑起来扎实吗？怎么撑满一篇论文 | 按现在的形态最多 workshop 短文；给出扩成完整论文的方案（三种机制、三领域、相图、真实基准、闭环、成本、机制分析） |
| 19 | 需要扎实充实有 insight 的论文，找最 promising 的完整研究主线 | **确定主线**："写入时付代价，还是读取时付代价？"四个 idea 全部并入。写 `docs/research_line.md` |
| 20 | 把全部算法设计、调研结果、related work 详细写进文档，方便下次新会话 | 本文档及配套的 algorithm_design、related_work、clm_notes、idea_bank、根目录 CLAUDE.md |

（这之后用户把 CLM 论文原文上传到仓库根目录 `clm.pdf`，提交 `8540e6b`。[clm_notes.md](clm_notes.md) 里标 [N] 的论文细节，现在可以对照原文核实。）

### 2026-11-15：换主线

| 步骤 | 用户的问题 | 结论 |
|---|---|---|
| 21 | 上传视频 KV 换页 proposal，问可行性 | 工程上能做，三处硬伤：缩略图超预算、效率论证反了、块间隔离改变了注意力拓扑。StreamScout 撞车，SimpleStream 是底线。建议先做 oracle 换页上界等三个只推理的实验。详见 [idea_bank.md](idea_bank.md) F1 |
| 22 | 澄清核心 idea：像 CLM 那样，把 KV 调度管理直接交给模型；不必是长视频 | 提出 Self-Paging 形式化（只追加日志 + 驻留集）、P1–P4、受控对比表、K1–K4、E0–E2。新一轮检索发现 SideQuest（模型自己删 KV）、Leyline、KVEraser 等（[related_work.md](related_work.md) §0） |
| 23 | "忘掉之前的主线，现在模型自己管 KV 就是我们的主线" | 旧主线文档归档到 `docs/archive/state_vs_history/`；重写 `research_line.md`、`algorithm_design.md`；更新 related_work §0、idea_bank F 组、本日志、README、CLAUDE.md。提交 `fd5e2dc` |
| 24 | 具体实验怎么做：bench、数据、SFT、RL、模型（服务器上已有 Qwen3.5-2B/4B/9B）、代码、vLLM 怎么改 | 写 `docs/experiment_plan.md`。三处修正：① Qwen3.5 是混合架构，换页只作用在全注意力层，加 dense 对照；② 动作接口改成原生 tool call；③ 引擎选 SGLang monkeypatch（仿 SCR），vLLM 第一版不改。T2 用 ACM 的 BrowseComp-Plus 680/150 划分，复用 ACM 的检索、评判和公开轨迹 |

---

## 2. 纠正过的错误（务必不要再犯）

### 2.1 对用户转来的"工作记忆"分析的核实

| 原说法 | 核实结果 |
|---|---|
| TerminalBench 2.1 上只打平 | **对**：准确率持平，FLOPs 是 Summary 的 70% |
| RL 在小模型上只打平 | **对**：42.5 vs 42.1；训练前 CLM 更差（28.8 vs 34.7） |
| 技能演化只在合成任务上 | **对**：ContextBench KV Store 38.3 → 74.2 |
| 多 agent 没开源 | **对**：Python 版删了 subagent 参数；Pi 插件也没有 |
| "编码任务上只打平" | **不准确**：TBLite 上 CLM 73.7 vs Summary 67.0 |
| "所有结果都在 32K 下" | **不准确**：EdgeBench 也跑了 128K，集群是 272K |
| "可变状态基本没被开发" | **说重了**：ContextBench 的 Sudoku Sketchpad、KV Store 就是可变状态任务；集群 orchestrator 用 163 次原地编辑维护计分板，上下文 6–8K，这也反驳了"预算放开后模型基本不会主动编辑" |
| "状态块"是新东西 | **已有雏形**：Pi 插件允许插入 `role=notes` 块（示例是 TASK TRACKER）；默认预算是整个窗口；大段工具输出自动换成指向文件的说明 |
| （遗漏） | 漏了最直接的先例 MemGPT/Letta 的核心记忆块 |

### 2.2 SKILL.state 的 InterCode CTF 数字：我的误读

- **我原来的说法**："状态和历史都保留的版本反而最差（41.8）"，并以此作为"早绑定 vs 晚绑定"的引子。
- **核实后**（搜索摘要 + caiboyang/ML-learning PR #22、zaebee/p-e #272 两份读书笔记）：
  - 三个数 54.2 / 46.4 / 41.8 本身是对的；
  - 46.4 是带**记忆摘要**的 ReAct，不是完整历史；
  - 41.8 是 **LangGraph** 实现的 Stateful agent；
  - 论文**没有**测"状态 + 完整历史"。
- **影响**："两者并存最差"不能当已知结论写，作为前提放弃。剩下的真问题（历史里新旧并存，每步重新解决矛盾）变成了 I3。
- **同一来源发现的 SKILL.state 弱点**：
  - SkillExecBench 上的 "accuracy" 是逐动作正确率，不是任务成功率；只有 T≥50 时差异显著；
  - 算法先提交状态更新再执行动作，失败的工具调用会让状态显示"已成功"。

### 2.3 "可验证状态"从首推降级

我在 10 月 3 日一度把"可验证、事务化的状态"（C4）列为首推，随后查到 MemTX、MemTxn、PatchOptic、PatchBoard、State-Bound Evidence、2606.22030，已经很挤，降级为方法的一个部件。

### 2.4 I3 改名造成的混淆

我把 I3 改名为"作废标记 vs 状态"单独推荐，而且只验证了它，用户以为是新 idea。以后**沿用用户确认过的编号 I1–I4**。

### 2.5 编号混淆：Nous 与 2606.22030

之前文档把 2606.22030 写成 "Nous"。实际上 2606.22030 是 *When Does Belief-Based Agent Memory Help? Reliability-Conditional Updating and Provenance-Capped Poisoning Defense*；Nous 是另一篇（2610.00094，*Nous: Learning and Certifying Memory Decisions Before Source Calibration*）。两者都只核对到搜索摘要。

### 2.6 BEAM-10M 上 Scroll 的数字

两份搜索摘要分别写 73.1 和 91.2，互相矛盾，**没有核实**。

---

## 3. 已验证的事实与证据

标记同 [related_work.md](related_work.md)：[R] 读过仓库或文档；[N] 基于全文的读书笔记；[S] 只看过搜索摘要。

### 3.1 支撑旧主线"统一解释"的三组证据

| 现象 | 数字 | 主线的解释 | 来源 |
|---|---|---|---|
| 程序性任务上状态赢 | delayed-relevance：事后更正时显式状态 93/93，完整历史 18/82；PoS：ALFWorld 88.81 vs Raw 62.69 vs 最强基线 72.39 | 以后需要什么可预测 → 写入误差小；回合长 → 读取误差大 | [R] |
| 需求不可预测时完整上下文赢 | Supersede：LongMemEval-KU 上完整上下文 82/91/92，300 字符笔记 63/64/77（gpt-4.1-mini / gpt-4.1 / gpt-5.4）；给更多空间也恢复不了 | 问题在读完之后才出现 → 写入误差大 | [R] |
| 排名随时长反转 | Ground Truth First（2607.21962）：记忆架构排名随使用时长反转，96% → 72% | 两类误差随长度增长的速度不同 → 必然有交叉点 | [S] |

### 3.2 delayed-relevance 复现项目的关键数字 [R]

完整表格见 [related_work.md](related_work.md) §6.1。要点：

- 成本：SKILL.state 的优势按 token 是 7.54 倍，按带缓存计费只剩 1.39 倍；ReAct 缓存节省 82%，所有会改写前缀的方法节省 0%；同样内容、可变状态放在历史前面，成本是放在后面的 5.7 倍；
- 延迟相关（k=40，Haiku）：完整历史 2/12 = 17%；没有字段 0/24；原文照搬提醒 16/24 = 67%；提炼后的提醒 24/24 = 100%；notes 字段约 21%（9–40%）；
- 事后更正：Haiku ReAct 3/44，SKILL.state 44/44；Sonnet ReAct 15/38，SKILL.state 49/49，但有 66 个不合 schema 的补丁、10 步无动作、21 个其他错误；
- Sonnet 上显式状态的主要失败：符合 schema、但语义抄错的补丁；
- 方法论：先测噪声底，至少四个结论在测噪声后被推翻；按依赖步数选种子，信号多 4 倍；
- 作者：SKILL.state 的作者是 Badhe、Tiwari、Chung；复现者是 Javier Aguilar（JaviMaligno）。

### 3.3 PoS 仓库 [R]

- Qwen3.7-Plus：ALFWorld Raw 62.69 → PoS 88.81（最强记忆基线 72.39）；LOCA（7 个长度 × 75 例汇总）Raw 43.62 → PoS 56.38；
- 三个模型的范围：ALFWorld Raw 62.7/81.3/93.3 → PoS 88.8/94.0/97.0；LOCA Raw 43.6–69.7 → PoS 56.4–74.3；
- RCA-100 上总 token 是原始轨迹的 5.06 倍；
- `ContextProvider` 接口只有 4 个方法；`contexts/raw.py` 45 行；
- 默认用 DashScope 的 OpenAI 兼容接口（`DASHSCOPE_API_KEY`）。

### 3.4 CLM 论文 [N]

见 [clm_notes.md](clm_notes.md) §8。最常引用的：BrowseComp-Plus 59.4%（相对 +11.4%）；TerminalBench 2.1 打平，FLOPs 70%；TBLite 73.7 vs 67.0；RL 42.5 vs 42.1（1.34 vs 2.19 PFLOPs）；9B zero-shot 28.8 vs 34.7；ICL KV Store 38.3 → 74.2；只有 CLM 享有的三项待遇。

### 3.5 状态派其他数字 [S]

- Scroll（Qwen3.8-Max）：LongMemEval-S 94.8；LOCA-256K 86.7；BEAM-10M 73.1 或 91.2（矛盾）；
- VISTA：LOCA 上 Gemini-3-Flash 22.7 → 50.7；38/75 vs ReAct 17/75；
- NOOA：253 行，SWE-bench Verified 82.2%（GPT-5.5），OpenCode 78.6%，Pi 78.2%；
- SKILL.state：仓库任务 200 步 0.94；100 步时累计 token 减 16.2 倍；prompt 恒定 1.7K–1.9K token；Table 1（Gemini-3-Flash）ReAct 0.90 → 0.74，SKILL.state 1.00 → 0.94；
- ContextRender：AppWorld 6K 预算接近完整历史，成本降 10–32%；
- ContextPipe：SWE-bench Pro 子集 token 降 31%；
- Schema：ARC-AGI-3 58.7 → 99.2；
- Why Retrying Fails：污染比 7.1；
- Lost in Compaction：压缩器只保留 17% 的会话约束；
- CompactionRL：GLM-4.5-Air SWE-bench Verified 66.8%、Terminal-Bench 2.0 24.5%；
- LOCA：部分模型 8K 时 >70%，256K 时跌到 2.7%；
- MemoryAgentBench：所有方法在多跳 CR 上 ≤6–7%。

### 3.6 支持和反对"在上下文里放状态"的证据

**支持**：StateAct 保留完整历史仍 +10%（ALFWorld）；Sculptor 在干扰问题上 22.5 → 99.4；PoS 相对 +22.68%；InfiAgent 的消融；Chroma Context Rot；AdaCoM（越强的模型越偏好保真度高的上下文）。

**反对或有风险**：自我条件化（2509.09677）和 FSM 执行研究（2511.14777）表明早期错误会传播，写错的状态放在注意力最强的尾部可能放大错误；Claude Code 文档说较新模型不需要书面清单就能跟踪多步工作，TodoWrite 默认关闭。

### 3.7 CLM 代码里的事实 [R]

- 角色折叠：编辑后非 assistant 角色一律变成 user；
- 角色头伪造：已用最小例子复现；
- SCR fork 模式下删除不等于遗忘；
- `_REMOVED_KWARGS` 暴露了作者试过的方向（`research_ledger`、`ctx_archive`、`subagents`、`gauge_*`、`shadow_*`、`unlimited_ctx_turns` 等）。

详见 [clm_notes.md](clm_notes.md) §2.4、§7。

### 3.8 新主线用到的数字（2026-11-15）

**估算**（按 Qwen3-4B/8B 公开配置：36 层、8 个 KV 头、head_dim 128、bf16；要对照实际 config 核对）：
- KV 约 2 × 36 × 8 × 128 × 2 B = 147,456 B，约 **144 KB/token**；4B 和 8B 一样；
- 1K token 的一页约 150 MB；PCIe 4.0 x16 按 20–25 GB/s 算，搬一页约 **6–7 ms**；
- 8B 模型 bf16 权重约 16 GB；A6000 显存带宽约 768 GB/s，所以 batch = 1 时解码一个 token 至少约 21 ms，实际约 20–30 ms；
- 权重读取折算成 KV token：batch = 1 时约 11 万 token，batch = 32 时约 3.4K token（[algorithm_design.md](algorithm_design.md) §5）。

**Qwen3.5 和引擎**：
- Qwen3.5-2B/4B/9B 是混合架构，Gated DeltaNet 和全注意力 3 : 1；原生多模态（early fusion）；原生上下文 262,144 [S]。层数和字段名待用 config 核对；
- vLLM 跑混合模型时，前缀缓存用 `align` 模式，仍是实验特性；有 issue 报告 Qwen3.5 的 attention block 被放大到 528 token，短于一个 block 的前缀命中不了；有把匹配粒度和物理 block 解耦的 PR（`prefix_match_unit`）[S]；
- SGLang 新版有 streaming session（`--enable-streaming-session`，会话槽不参与淘汰，来自 PR #21875）[S]；SGLang 的 KV 按 token 粒度管理（`req_to_token`），SCR 已在 0.5.16 上给混合模型做过会话级 KV 改动 [R]。

**ACM 仓库（`third_party/methods/acm`，MIT）[R]**：
- BrowseComp-Plus 划分：`bcp_train_680.json`、`bcp_eval_150.json`、`bcp_full.json`（830），每条只有 id、question、answer；
- 语料、BM25 Lucene 索引和检索代码来自 HF `hltcoe/BrowseComp-Plus`，pyserini 需要 JDK；可选 Qwen3-Embedding-8B 稠密检索；
- 工具：`search` 返回片段（新结果 512 token、看过的 128 token），`get_document` 最多 8192 token；
- 评判：OpenAI SDK + 官方 simple-evals 模板，默认 GPT-5；可以把 base URL 换成本地服务；
- HF 上公开了基座 Qwen3.5-9B 在训练 680 题上的 pass@4 轨迹（ReAct 和 memtool，约 4.4 GB），以及 OPD 三轮的 checkpoint。

**检索到的关键数字** [S]：
- SideQuest：215 条训练样本；峰值 token 降低最多 65%；
- Declarative Attention：注意的 token 降 52.0% / 31.1%，准确率掉 1.27 / 2.75 个点；小模型选区成功率 58%；
- Online KV Compaction（2608.00902）：立刻压缩会掉点，延后到问题可见时再压缩能找回大半；淘汰 80% 的 KV 仍保留大部分准确率；
- SimpleStream：最近 4 帧，OVO-Bench 67.7、StreamingBench 80.59；
- StreamScout：Qwen3-VL-8B 在 OVO-Bench 上 +14.65 分，token 少 59%。

---

## 4. 代码和基准的可用性（2026-10-03 核查）

方法：`git ls-remote` + depth-1 clone，或 raw.githubusercontent 拉取 README/LICENSE。

### 4.1 方法

| 档次 | 方法 |
|---|---|
| API 模型能直接跑、不训练、带评测脚本 | PoS（MIT）、RLM（MIT）、SelfCompact（MIT）、VISTA（**无 license**） |
| 有方法代码，评测要自己写 | Scroll（Apache-2.0，评测适配器在私有仓库）、BB-WM（无 license）、ACM 的 memtool 模式（MIT）、SKILL.state 的非官方复现（WUWeifeng710，MIT） |
| 要训练或自托管模型 | Agent-BRACE、ContextPilot、CompactionRL、From History to State、Context-as-a-Tool |
| 产品化运行时 | InfiAgent（GPL-3.0）、ContextPipe/Astra、TokenPilot/LightRSI、ESAA |
| 没有代码 | BeliefMem、ContextRender、Schema、ContextEvo、SKILL.state 官方 |

### 4.2 基准

| 结论 | 基准 |
|---|---|
| 用 | 仓库探针（自建）、LOCA-bench、ALFWorld、LongMemEval-S + KU 子集（含 Supersede） |
| 备选 | BEAM 128K/1M、STATE-Bench、InterCode CTF、τ-bench、AppWorld、RCA-100、Prosus Vending Bench、TextQuests、SlopCodeBench、dict_sum |
| 不用 | StateMemBench、WorldLines、BeliefShift、SkillExecBench（未发布）；DreamBench-SWE（评分不公开）；MemoryAgentBench（数据只在 HF）；CooperBench（harness 固定）；ClinDiag；LongMemEval-V2；AMA-Bench |

详细理由见 [related_work.md](related_work.md) §8。

### 4.3 接入本仓库的 15 个 submodule

见 [`third_party/README.md`](../third_party/README.md)。都用 `git update-index --add --cacheinfo 160000,<sha>,<path>` 注册，`.gitmodules` 里设了 `shallow = true`。

| 路径 | 上游 | commit | license |
|---|---|---|---|
| methods/pos | luoyu100/PoS | d6acb43 | MIT |
| methods/scroll | niceIrene/QwenPaw（scroll-research 分支） | 3db60c5 | Apache-2.0 |
| methods/skill-state-runtime | WUWeifeng710/skill-state-runtime | 84b9804 | MIT |
| methods/vista | binyxu/VISTA | 647a93d | 无 |
| methods/rlm | alexzhang13/rlm | d04208a | MIT |
| methods/selfcompact | tianjianl/selfcompact | 1295817 | MIT |
| methods/acm | lixiaochuan2020/agentic-context-management | f06f90e | MIT |
| methods/belief-world-models | skumar-ml/belief-world-models（master） | ac5bf2b | 无 |
| references/delayed-relevance | JaviMaligno/delayed-relevance | 53ceaf4 | 无 |
| benchmarks/loca-bench | hkust-nlp/LOCA-bench | 8b6fac4 | MIT |
| benchmarks/alfworld | alfworld/alfworld（master） | aaba687 | MIT |
| benchmarks/longmemeval | xiaowu0162/LongMemEval | 9e0b455 | MIT |
| benchmarks/supersede | Vrin-cloud/supersede | 677993d | Apache-2.0 |
| benchmarks/state-bench | microsoft/STATE-Bench | 5644b18 | MIT |
| benchmarks/beam | mohammadtavakoli78/BEAM（约 4 GB） | b2da22e | MIT |

---

## 5. 冒烟测试结果

**全部只说明方向，不能当证据**：每格 n=3；Claude 子 agent 带着 Claude Code 自己的系统提示，温度不可控，拿不到 token 用量。

### 5.1 第一轮：I3（Haiku，36 题）

记录：[`state_study/pilots/2026-10-03_haiku_smoke`](../state_study/pilots/2026-10-03_haiku_smoke/README.md)。设置：种子 0/1/2，gap 60，warm-up 200，决策步在第 270–290 步。

| 场景 / 条件 | 正确 | 说明 |
|---|---|---|
| delayed / raw | 2/3 | 错误是选了已被占用的货架（重建错误），不是忽略隔离 |
| delayed / state | 3/3 | |
| delayed / state_noq | **0/3** | 全部选了被隔离的货架 |
| explicit / raw | 2/3 | 选了 S1，重建错误 |
| explicit / tomb_inline | 2/3 | 同样是 S1 |
| explicit / tomb_tail、delete、state、state_log | 各 3/3 | |
| implicit / raw | 2/3 | 1 次选了作废前的旧答案 |
| implicit / tomb_tail、state | 各 3/3 | |

要点：
- 缺字段会稳定失败；
- `raw` 的 3 个错误里只有 1 个是用了旧值，2 个是重建错误。这直接导致把 I3 拆成 M1 干扰和 M2 重建负担。

### 5.2 第二轮：I1 + I4（Haiku / Sonnet / Opus，36 题）

记录：[`state_study/pilots/2026-10-03_i1_i4_scale`](../state_study/pilots/2026-10-03_i1_i4_scale/README.md)。Haiku 在 `explicit/raw` 和 `explicit/state` 上沿用第一轮结果（生成是确定性的）。

**I4**（Haiku）：

| 场景 | 陈旧状态 | 只召回候选 | 召回候选 + 更低编号货架 |
|---|---|---|---|
| delayed（漏了约束） | 0/3 | **3/3** | 2/3（1 个 missed_lower） |
| explicit（漏了机会） | 0/3 | **0/3** | **3/3** |

**I1**（explicit 场景）：

| 模型 | raw | state | self_schema |
|---|---|---|---|
| Haiku | 2/3（1 个 occupied） | 3/3 | 3/3 |
| Sonnet | 3/3 | 3/3 | 3/3 |
| Opus | 3/3 | 3/3 | 3/3 |

结论：
- I4 发现了"约束 vs 机会"的不对称，写进了方法设计；
- I1 在单题探针上测不出规模效应，需要多步回合或更难的探针；
- 账本能自动区分 stale 和 occupied。

### 5.3 不调模型的验证

- **探针测试**：20 个全部通过（`python -m pytest state_study/tests`）；
- **PoS 离线检查**：`check_behavior_equivalence.py --self-test` 和 `check_method.py` 输出 "Passed: Raw, full PoS, both ablations, and diagnostic PoS"；28 个单元测试通过；
- **ALFWorld 真值**：valid-unseen 134 局；重放 12 局、6 种任务类型、0 违规；clean/hot/cool 属性正确；
- **submodule**：所有固定 commit 都用 `git ls-remote` 解析过；抽 3 个实际拉取，都检出到对应 commit；`setup_third_party.sh` 遇到未知分组时退出码 2；
- **文档链接**：相对链接全部能找到目标。

---

## 6. 环境限制与搭建命令

### 6.1 云环境实测（2026-10-03）

| 项目 | 情况 |
|---|---|
| 硬件 | 4 核 CPU、15 GB 内存、**没有 GPU**（无 `nvidia-smi`） |
| Docker | 有命令，但守护进程没在运行 → CLM 原版基于 Harbor 的运行方式用不了 |
| 语言 | Python 3.11 和 3.12、Node 22、npm、uv |
| 能访问 | PyPI、npm registry、github.com、raw.githubusercontent.com |
| 被拦截（代理 403） | arxiv.org、huggingface.co、alphaxiv、emergentmind、themoonlight、academy.dair.ai、manus.im、letta.com、modelscope、hf-mirror、ollama、OpenAI、DashScope、OpenRouter、Google Drive、openreview、semanticscholar、*.github.io、medium、dev.to |
| 被拦截的影响 | 读不了论文原文；LongMemEval 数据下载不了；Qwen（PoS 默认）、GPT 接口用不了；本地模型权重下载不了 |
| api.github.com | 被拦截，改用 `git ls-remote` / `git clone` |
| 网络上可达但没有密钥 | api.anthropic.com、Gemini 接口 |
| 模型 API 密钥 | 环境变量里没有任何模型服务的 key |
| tiktoken | 词表下载被拦，token 数按字符数 ÷ 4 估算 |
| 生命周期 | 容器闲置后会被回收；不适合跑几天的实验矩阵 |

**原则**：
- 不会把本会话自己的登录凭证取出来调 API；
- 不要求用户把密钥贴到聊天里。

**2026-11-15 补充**：
- WebSearch 能用，能拿到 arXiv 论文的搜索摘要，但论文正文仍然读不到；
- 新主线不需要模型 API，但需要 GPU 和开源权重：云端没有 GPU，huggingface.co 被拦，所以实验在用户机器上跑；云端只写代码、在 CPU 上跑小模型测试；
- PyPI 能用，所以可以在 scratchpad 的虚拟环境里装 CPU 版 torch 和 transformers。

### 6.2 解除阻碍需要用户做的事

入口是会话标题栏的云环境菜单 → Edit：

1. **Network access**：加上要用的模型服务域名（`dashscope.aliyuncs.com` 或 `api.openai.com`），以及 `arxiv.org`、`huggingface.co`；
2. **环境变量 / API credentials**：加 `DASHSCOPE_API_KEY`（PoS 默认）、`OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`。新开的会话才会读到新设置；
3. **数据**：LongMemEval 需要用户自己下载（HF 或 Google Drive）；
4. **长时间实验**：放在用户自己的机器或服务器上跑。

### 6.3 搭建命令（已在云端跑通）

```bash
# 第三方代码（core 组：PoS、LOCA、ALFWorld、SKILL.state 运行时、Supersede、LongMemEval）
scripts/setup_third_party.sh core          # 其他分组：methods | benchmarks | references | optional | all

# 探针测试（只需标准库 + pytest）
python3 -m pip install -q pytest
python3 -m pytest -q state_study/tests

# PoS + ALFWorld 环境（放在 scratchpad 里，避免污染仓库）
V=<scratchpad>/venv-pos
uv venv -q -p 3.11 $V && . $V/bin/activate
cd third_party/methods/pos
uv pip install -q -r requirements/alfworld.txt
python scripts/check_behavior_equivalence.py --self-test   # 离线检查，不调模型
python scripts/check_method.py
python scripts/prepare_benchmarks.py --benchmark alfworld   # 下载 ALFWorld 数据（GitHub release，可用）
cd -

# ALFWorld 逐步真值：重放专家
python -m state_study.groundtruth.alfworld_facts --data third_party/methods/pos/benchmarks/ALFWorld --games 12 --stride 11

# 探针出题与打分
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 --plan explicit:raw,state
python -m state_study.probes.build_items score --out OUT

# 只下载 Harbor wheel 查看注册表（不安装）
pip download harbor==0.16.1 --no-deps --python-version 3.12 --only-binary=:all: -d harbor_pkg
```

注意：
- PoS 数据准备脚本会把 ALFWorld 数据放到 `third_party/methods/pos/benchmarks/ALFWorld/`，这个目录在 submodule 里，**不要提交**；
- 跑完测试后清理 `__pycache__` 和 `.pytest_cache` 再提交。

---

## 7. 待核实清单

### 7.1 新主线

- [ ] **SideQuest 全文**：有没有换回（从内存恢复 KV）；训练是 SFT 还是有 RL；怎么定义"段"；评测任务和基线。这决定我们的贡献边界
- [ ] Declarative Attention 全文：状态机实现、能不能接训练、有没有容量上限的实验
- [ ] Leyline、KVEraser、PatchKV、KVP、2608.00902 全文：和 P4（残留）、K2（可逆性）相关的实验
- [ ] Qwen3-4B/8B 的实际 config（层数、KV 头数、head_dim），核对 §3.8 的估算
- [ ] HF transformers 里 Qwen3 写 cache 的 key 是否已经加过 RoPE（[algorithm_design.md](algorithm_design.md) §3.2 的精确性依赖这一点；阶段 0 的等价性测试会直接验证）
- [ ] `clm.pdf`：对照原文核实 [clm_notes.md](clm_notes.md) 里标 [N] 的数字，尤其是 RL 配置和 SCR 结果
- [ ] BrowseComp-Plus：语料、检索器、评判能否全部本地化
- [ ] 用户的实际 GPU 硬件（原 proposal 写的是 4×RTX A6000）
- [ ] Qwen3.5-2B/4B/9B 的 config：层数、`layer_types`、KV 头数、head_dim、partial rotary（命令见 [experiment_plan.md](experiment_plan.md) §12）
- [ ] 用户服务器上 torch、transformers、SGLang、fla 的版本；transformers 里 Qwen3.5 的 cache 类名和 GDN 状态字段
- [ ] SGLang streaming session（PR #21875）在用户的版本里是否可用，能不能直接当会话层
- [ ] 本地评判模型和 GPT-5 评判的一致率（T2 的数字才能和 ACM 比）
- [ ] 服务器能否访问 HF：BrowseComp-Plus 语料和索引、ACM 公开轨迹、Qwen3-8B
- [ ] 原 proposal 里检索不到的名字：WRWS、MemStream、NGC、StreamingTOM、EgoProbe、MemInStream

### 7.2 旧主线（已归档，暂不处理）

写作前必须完成（来自[归档的 proposal.md](archive/state_vs_history/proposal.md) §12，并补充）：

- [ ] SKILL.state 原文：InterCode CTF 三个基线的确切定义和数字；SkillExecBench 的 accuracy 定义；"先提交后执行"的算法细节
- [ ] Scroll 原文：BEAM-10M 的数字（73.1 还是 91.2）；LOCA-256K 的设置
- [ ] VISTA 原文：LOCA 上的确切设置（22.7 → 50.7）
- [ ] PoS：LOCA 每个长度的分项结果
- [ ] Ground Truth First 原文：96% → 72% 的具体设置，"tenure" 的定义
- [ ] Diagnosing Retrieval vs. Utilization、Rate–Distortion、Know It Act on It、When Does Memory Help、LAM、MD5 状态跟踪、Long-Horizon Task Mirage、Memory Trust Gap：读原文，确认和我们的误差分解有多少重叠
- [ ] ALFWorld：在 PoS adapter 里打开 facts 后能否稳定拿到逐步真值（专家重放已验证，真实 agent 轨迹未验证）
- [ ] LOCA：各任务家族的模拟服务能否逐步拍快照
- [ ] LongMemEval-KU：能否从 `has_answer` 和 `answer_session_ids` 推出"哪一轮是被取代的旧值"
- [ ] license：联系 VISTA 和 delayed-relevance 的作者
- [ ] 每篇论文动笔前一周：重新检索撞车（关键词见 [related_work.md](related_work.md) §10）

---

## 8. 下一步

按 [experiment_plan.md](experiment_plan.md) §11：

1. **云端现在就能做的（阶段 0，只需 CPU）**：
   - 按 experiment_plan §8 建 `self_paging/`：`core/`（页、协议、轨迹格式、代价）、`hf/`（LogStore、runner、掩码、Qwen3.5 全注意力层补丁）、`policies/`（含 Belady）、`envs/t1_warehouse.py`、`envs/t1b_lookup.py`、`data/oracle_sft.py`；
   - 用随机初始化的小 Qwen3.5 配置跑等价性测试，包括 GDN 层（需要在 scratchpad 的虚拟环境里装 CPU 版 torch 和支持 Qwen3.5 的 transformers；fla 的 kernel 可能需要 GPU，CPU 上用 transformers 自带的 torch 实现）；
   - 写好 E0、E1、E2 的驱动和打分脚本。
2. **需要用户服务器的**：
   - 回复 experiment_plan §12 的信息（config、GPU、软件版本、HF 访问、JDK、评判模型）；
   - 在真实权重上跑等价性测试，然后跑 E0、E1、E2，做继续/止损判断。
3. **之后**：SGLang overlay 和引擎一致性测试 → SFT 数据（四个来源）→ SFT → RL。
4. **动笔前**：先读 SideQuest 全文（§7.1）。
