# 研究日志与已验证事实

本文档按时间记录这个课题的决策过程，并汇总所有已验证的事实、纠正过的错误、代码和基准的可用性、冒烟测试结果和环境限制。目的是让新开的会话不用重新调研。

- 主线：[research_line.md](research_line.md)
- 算法与实验设计：[algorithm_design.md](algorithm_design.md)
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
| 21 | 把这些 paper 引领的趋势捋清楚：是不是从 memory 往 state 走？最好的切入点是什么 | 趋势确实在往状态走，但证据比宣称的弱；提出"显式状态是用干扰换持久"（C1 干扰律、C2 持久律、C3 交叉点可预测、C4 规模） |
| 22 | 先验证 paper 路径可行、方法初步可靠，再交付 | C1 不成立、C2 一半成立、C3 无法检验；发现 delayed-relevance 的 L2 事后更正探针自相矛盾（§2.7）。修正为"决定成败的是到达时的写入，不是表示方式"。见 [validation_2026-10-03.md](validation_2026-10-03.md)，**主线待用户决定是否改写** |

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

### 2.7 "事后更正时显式状态 93/93、完整历史 18/82"主要是探针伪影

- **原说法**（§3.1、§3.2、proposal、literature_map、related_work、idea_bank 都引用过）：事实被更正后，显式状态 93/93 次用对，完整历史只对 18/82 次；解释是"历史要在每一步重新解决矛盾"。I3 的动机也来自这里。
- **核实后**（运行 delayed-relevance 自己的环境，`state_study/validation/dr_coherence.py`）：
  - 更正通知总是针对第 0 步的入库（货架 0），但那托货早已发出；
  - 它 L2 用过的 6 个种子全部如此，其中 5 个种子的货架 0 上已经放了另一托货，环境仍把货架 0 清空；
  - 种子 0–39 中 36 个通知与历史矛盾。
- **影响**：
  - 历史 agent 按时间线推理（"货架 0 上是第 8 步放的货"），被判错；状态 agent 看不到矛盾，照写，被判对；
  - 只改通知里的 `corrects_step` 一个字段，Haiku 子 agent 重放从 4/12 升到 10/12（方向性，代理校准不好）。
- **结论**：这组数字不能再当"状态在更正上胜出"的证据。I3 的真问题变成"通知与历史冲突时谁来裁决"，探针必须加一致性检查。详见 [validation_2026-10-03.md](validation_2026-10-03.md) §2.2–§2.3。

---

## 3. 已验证的事实与证据

标记同 [related_work.md](related_work.md)：[R] 读过仓库或文档；[N] 基于全文的读书笔记；[S] 只看过搜索摘要。

### 3.1 支撑主线"统一解释"的三组证据

| 现象 | 数字 | 主线的解释 | 来源 |
|---|---|---|---|
| 程序性任务上状态赢 | delayed-relevance：事后更正时显式状态 93/93，完整历史 18/82（⚠️ 主要是探针伪影，见 §2.7）；PoS：ALFWorld 88.81 vs Raw 62.69 vs 最强基线 72.39 | 以后需要什么可预测 → 写入误差小；回合长 → 读取误差大 | [R] |
| 需求不可预测时完整上下文赢 | Supersede：LongMemEval-KU 上完整上下文 82/91/92，300 字符笔记 63/64/77（gpt-4.1-mini / gpt-4.1 / gpt-5.4）；给更多空间也恢复不了 | 问题在读完之后才出现 → 写入误差大 | [R] |
| 排名随时长反转 | Ground Truth First（2607.21962）：记忆架构排名随使用时长反转，96% → 72% | 两类误差随长度增长的速度不同 → 必然有交叉点 | [S] |

### 3.2 delayed-relevance 复现项目的关键数字 [R]

完整表格见 [related_work.md](related_work.md) §6.1。要点：

- 成本：SKILL.state 的优势按 token 是 7.54 倍，按带缓存计费只剩 1.39 倍；ReAct 缓存节省 82%，所有会改写前缀的方法节省 0%；同样内容、可变状态放在历史前面，成本是放在后面的 5.7 倍；
- 延迟相关（k=40，Haiku）：完整历史 2/12 = 17%（这是较早版本；v3 轨迹重新统计是 0/24，见 [validation_2026-10-03.md](validation_2026-10-03.md) §2.4）；没有字段 0/24；原文照搬提醒 16/24 = 67%；提炼后的提醒 24/24 = 100%；notes 字段约 21%（9–40%）；
- 事后更正（⚠️ 探针自相矛盾，见 §2.7）：Haiku ReAct 3/44，SKILL.state 44/44；Sonnet ReAct 15/38，SKILL.state 49/49，但有 66 个不合 schema 的补丁、10 步无动作、21 个其他错误；
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

### 5.3 第三轮：验证"干扰换持久"（Haiku 子 agent + delayed-relevance 轨迹重新统计）

记录：[`state_study/pilots/2026-10-03_validation`](../state_study/pilots/2026-10-03_validation/README.md)；结论：[validation_2026-10-03.md](validation_2026-10-03.md)。要点：

- L2 事后更正探针自相矛盾（§2.7）；一字段修正后历史 4/12 → 10/12；
- L1 延迟约束（相隔 40 步）：历史 0/24、状态无字段 0/24（原 API 轨迹）；状态的自由字段到达时写下 → 10/11 用对，没写下 → 0/13；
- 重放：通用回看指令 0/12，改写成占用事实 0/12，决策时提醒 11/12，agent 自己钉住 12/12；
- 短上下文下历史对更正很稳（24/24）；版本深度到 25 个版本看不出影响（18/20）；
- 显式状态的错误信念中位持续 24 步，被拒后修好 8/78。

### 5.4 不调模型的验证

- **探针测试**：40 个全部通过（`python -m pytest state_study/tests`；其中 20 个是第三轮新加的规则读者测试）；
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

# delayed-relevance 轨迹重放和分析（只运行、只读，不复制进仓库）
uv venv -q -p 3.11 <scratchpad>/venv-dr && <scratchpad>/venv-dr/bin/pip install -q anthropic google-genai numpy scipy pandas
git -C third_party/references/delayed-relevance fetch --depth 1 origin runs/table1-gemini-3-flash-preview-vertex
git -C third_party/references/delayed-relevance archive FETCH_HEAD results/ | tar -x -C <scratchpad>/drtraces
# 具体命令见 state_study/validation/README.md

# 只下载 Harbor wheel 查看注册表（不安装）
pip download harbor==0.16.1 --no-deps --python-version 3.12 --only-binary=:all: -d harbor_pkg
```

注意：
- PoS 数据准备脚本会把 ALFWorld 数据放到 `third_party/methods/pos/benchmarks/ALFWorld/`，这个目录在 submodule 里，**不要提交**；
- 跑完测试后清理 `__pycache__` 和 `.pytest_cache` 再提交。

---

## 7. 待核实清单

写作前必须完成（来自 [proposal.md](proposal.md) §12，并补充）：

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
- [ ] 读原文：*Delivery, Not Storage*（2607.20972，最接近）、PIS（2609.01272）、PM-Bench（2607.12385）、TriggerBench（2606.23459，回顾性记忆的定义）、2609.37125；确认是否已经讨论"显式状态 vs 历史"
- [ ] SKILL.state 原文的"静默漂移"实验：是否也有 L2 那样的一致性问题
- [ ] 人工复核 `dr_tables.py` 里"自由字段到达时写下"的启发式判断

---

## 8. 下一步

**2026-10-03 更新**：第三轮验证改变了优先级，按 [validation_2026-10-03.md](validation_2026-10-03.md) §5 执行：

1. 读最接近的五篇原文（需要放行 arxiv.org 或用户提供 PDF）；
2. 用 API 重做全部重放（Haiku 4.5 / Sonnet 5.5 / Opus 5.5，每格 n=24，约 30 美元）；
3. 在自写仓库环境上做闭环实验，因子为更正是否一致、通知是否过时、延迟约束 / 延迟事实；
4. 第二个领域（ALFWorld 注入延迟约束）；
5. 用户决定是否把主线改写成"到达时绑定"。

以下是第三轮之前的计划，保留备查：

按主线阶段 1（[research_line.md](research_line.md) §8）：

1. **不需要 API、现在就能做的**：
   - H1 旋钮（`n_state_events`、`n_telemetry`、`n_voided`）和 `control` 场景，配测试（[algorithm_design.md](algorithm_design.md) §1.5、§1.6）；
   - 新条件 `state_head`、`recall_raw`、`recall_all`、`ask_then_act`；
   - API 调用器（温度 0、记录 token 和缓存命中、断点续跑）和噪声底脚本；
   - 领域 2、领域 3 的生成器；
   - 闭环驱动器。
2. **需要 API 之后**：噪声底 → H1、H5（3 个以上规模）→ 止损判断。
3. **需要用户决定的**：模型家族和 API 预算；是否联系 VISTA 和 delayed-relevance 的作者。
