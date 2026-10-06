# 相关工作全集（截至 2026-10-03）

本文档把这个课题调研过的所有工作汇总在一处：做法、代码和 license、关键数字、和我们的关系。要看简明版的谱系和趋势，读 [literature_map.md](literature_map.md)；要看"我们和谁最近、怎么区分"，读本文 §1 和 [research_line.md](research_line.md) §7。

## 核实程度标记

本环境的网络策略拦截了 arxiv.org、huggingface.co、alphaxiv 等，**论文正文几乎都没读到**。每条信息都标了来源：

| 标记 | 含义 |
|---|---|
| **[R]** | 实际拉取并读过 GitHub 仓库（README、代码、LICENSE），或读过官方文档页面 |
| **[N]** | 第三方基于论文全文写的读书笔记（GitHub 上的笔记） |
| **[S]** | 只看过搜索引擎摘要，**引用前必须核对原文** |
| **[T]** | 只看到标题 |

日期：搜索结果里有具体日期的写到日；否则按 arXiv 编号推算到月（例如 2608 = 2026 年 8 月）。

---

## 1. 和主线最接近的工作（必须正面区分）

| 工作 | 编号 / 日期 | 做了什么 | 我们的区别 | 标记 |
|---|---|---|---|---|
| **delayed-relevance**（SKILL.state 的独立复现） | GitHub，2026-10-02 | 在重新实现的仓库环境上，用 Haiku 4.5 和 Sonnet 5 测了延迟相关、事后更正和带缓存的成本（数字见 §6.1） | 只有单一环境、两个模型；没有作废标记的对照；没有把干扰和重建拆开；没有跨真实基准的预测 | [R] |
| What to Keep, What to Forget：记忆压缩的率失真视角 | 2607.08032 | 用信息论给压缩的误差下界；提出 COMPACT-Bench | 不涉及"从历史重建"的读取误差；没有 agent 状态跟踪真值上的干预实验 | [S] |
| Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory | 2603.02473 | 把记忆问答的失败分成写入、检索、使用三层，发现检索是瓶颈 | 针对检索式问答；我们针对**可变状态**的 agent，以及上下文内的重建和干扰 | [S] |
| Ground Truth First：纵向记忆评估与 tenure crossover（Quentin Spencer） | 2607.21962 | 先由带随机种子的"人生剧本"采样器生成事实（带有效区间、易变等级、来源渠道），再生成文本；发现记忆架构的排名随使用时长反转（96% → 72%） | 它报告了反转现象；我们给出机制，并预测交叉点 | [S] |
| Know It, Act on It：个性化中的记忆利用 | 2607.29433 | 研究"知道但没用上" | 只对应我们读取误差里的"用上失败"子项 | [S] |
| StateMemBench：Can Agent Memory Systems Track Evolving State? | 2608.19652 | 234 个多会话场景，专门检测 agent 是否依据已被取代的事实行动；用确定性事件程序算真值 | 指出了"状态漂移"现象；我们解释漂移由哪项误差造成。未找到代码和数据 | [S] |
| Supersede：Diagnosing and Training the Memory-Update Gap | 2606.27472 | LongMemEval 知识更新子集上，自维护的有界笔记输给完整上下文；提供 RL 训练环境 | 现象 + 训练修复；我们给出机制。它的数字是我们 H3 的一个锚点 | [R] 代码，[S] 论文 |
| When Does Memory Help? Cost-Aware Evaluation | 2609.05441 | 工具型 agent 长期记忆的带成本评估 | 成本维度的近邻；不做误差分解 | [T] |
| LAM：Lossy Agent Memory，带检索分数误差界 | 2609.32256 | 有损记忆框架，给出检索分数的误差界 | 理论近邻，针对检索 | [T] |
| Long-Horizon State Tracking：用一长串依赖工具调用执行 MD5 | 2609.00012 | 长程状态跟踪的压力测试 | 测的是纯状态跟踪能力；我们关心表示方式的选择 | [T] |
| The Long-Horizon Task Mirage? 诊断 agent 系统在哪、为什么崩 | 2604.11978 | 长程任务的失败诊断 | 诊断框架的近邻，需要读原文确认是否涉及状态与历史 | [T] |
| STALE：Can LLM Agents Know When Their Memories Are No Longer Valid? | 2605.06527 | 记忆失效的识别 | 跨会话记忆系统；我们在单回合上下文里做因果对照 | [S] |
| Temporal Validity in Retrieval Memory | 2606.26511 | 一个确定性的"取代层"，消除检索记忆里的过时事实错误 | 检索记忆的工程方案；对应我们的作废标记，但不在上下文内 | [T] |
| When Memory Updates but Behavior Does Not | 2608.01619 | 修复个性化回复里隐含的过时依赖 | 对应"用上失败" | [T] |
| The Memory Trust Gap：持久记忆 agent 中依赖能力的失败 | 2609.01852 | 失败随模型能力变化 | 和 H5（规模的不对称）相关，需读原文 | [T] |

---

## 2. CLM 与上下文管理这一派

### 2.1 CLM 本身

| 项 | 内容 |
|---|---|
| 编号 / 日期 | 2609.37725，2026-09-29 |
| 作者 | Rulin Shao（一作；UW / Meta Superintelligence Labs）、Shannon Zejiang Shen、Junjie Oscar Yin、Yuetai Li、Minheng Wang、Hamish Ivison、Radha Poovendran、Nathan Lambert、Teng Xiao、Mike Lewis、Wen-tau Yih、Luke Zettlemoyer、Pang Wei Koh |
| 做法 | 把上下文镜像成文件，模型用 bash/python 任意改写；`C_{t+1} = f(C_t, y_t)` |
| 代码 | 本仓库 `clm/`，CC BY-NC 4.0 [R] |
| 详细笔记 | [clm_notes.md](clm_notes.md) |

### 2.2 harness 定时压缩（何时压缩不由模型决定）

| 工作 | 编号 / 日期 | 做法 | 代码 | 标记 |
|---|---|---|---|---|
| Codex、Claude Code、Cursor、Terminus-2 | 产品 | 接近上限时把历史压成一份摘要。Codex 约在 90% 触发，`/responses/compact` 返回一个加密的笔记项放在最后（第三方 gist，2026-09-26） | — | [R] 文档 / gist |
| MEM1 | 2506.15841，2025-06-18 | 用 RL 训练模型每轮重写一份内部状态，旧上下文丢弃；目标是恒定内存 | 有 GitHub | [S] |
| MemAgent | 2507.02259 | 分块读长文档，用 RL 训练模型覆盖写一个定长记忆 | — | [S] |
| ReSum | 2509.13313，2025-09-16 | 网页搜索 agent 周期性写摘要，并用 RL 训练 | — | [S] |
| SUPO | 2510.06727 | 摘要式压缩 + RL | — | [S] |
| MemAct | 2510.12635 | 压缩家族 | — | [T] |
| CompactionRL | 2607.05378，2026-07-06 | 接近上限时由策略写摘要，加最近几轮继续；用 PPO 和任务一起训练。GLM-4.5-Air：SWE-bench Verified 66.8%，Terminal-Bench 2.0 24.5% | 无官方代码；非官方：RainyFields/Comp_Rubric（Apache-2.0，Context-Folding 的 fork，含 `agents/compaction_agent.py`） | [S] |
| Focus | 2601.07190，2026-01-12 | 压缩家族 | — | [T] |
| AdaCoM | 2605.30785，2026-05-29 | 发现越强的模型越偏好保真度高（压缩少）的上下文 | — | [S] |
| ReadAgent | 2402.09727，2024-02-15 | 要点记忆页 + 查找，用于读长文档 | — | [S] |

### 2.3 模型自己决定何时管理，但只能用给定工具

| 工作 | 编号 / 日期 | 做法 | 策略来源 | 代码 | 标记 |
|---|---|---|---|---|---|
| Context-as-a-Tool（CAT） | 2512.22087，2025-12-26 | 上下文管理作为和改文件同级的工具；工作区分三块：稳定的任务语义、压缩后的长期记忆、保真的近期交互 | 训练（32B SWE-Compressor） | 未找到 | [S] |
| SelfCompact | 2606.23525 | 一个压缩工具 + 一份"何时压、何时别压"的规则说明（子任务完成时等）；策略有 none / fixed / selfcompact 三种 | 不训练 | tianjianl/selfcompact，MIT，2026-06-25；LiteLLM；需要 Serper key | [R] |
| ACM | 2607.23809，2026-07-26 | 摘要、卸载、查询三类记忆工具 | 教师模型生成"何时该管"的示范数据，蒸馏（OPD）训练 Qwen3.5-9B | lixiaochuan2020/agentic-context-management，MIT；`memtool` 模式不训练也能跑；BrowseComp-Plus 680/150 划分，GPT-5 评分 | [R] |
| Sculptor | 2508.04664，2025-08-06 | 切分片段、摘要/隐藏/恢复、搜索；针对旧信息干扰新信息（PI-LLM 上 RL 后 22.5% → 99.4%） | RL | — | [S] |
| Context-Folding | 2510.11967，2025-10-13 | 开子分支做子任务，回来时折叠成摘要 | RL | — | [S] |
| AgentFold | 2510.24699，2025-10-29 | 多粒度折叠 | RL | — | [S] |
| AgeMem | 2601.01885 | 用工具统一管理长短期记忆：存、取、改、摘要、丢弃；在 ALFWorld、SciWorld 上评测 | 三阶段 RL，按步 GRPO | — | [S] |
| ContextPilot（腾讯） | 2608.28476 | 更多"软删除"工具，被折叠的内容可以用关键词搜回来；在关键上下文决策点分支采样来分配功劳。和 CLM 方向相反：扩充工具集 | RL | Tencent/ContextPilot，Apache-2.0；`infer/` 有 InfBench、NovelQA、LongMemEval、BrowseComp+ 评测，`train/` 有 RL 代码；8B/14B/E4B 模型在 HF | [R] |
| AutoCompact | 2610.02163，2026-10-02 | 学习在编码 agent 中何时压缩 | 学习 | — | [T] |
| Practical Online KV Cache Compaction for LLM Agents | 2608.00902 | KV 缓存层面的在线压缩实证研究 | — | — | [T] |
| Beyond Token Savings | 2609.32961 | 压缩策略的系统研究，在 SWE-bench 和 Terminal-Bench 上约 3.5 万次运行 | — | — | [S] |
| Lost in Compaction | 2608.11242 | 压缩器只保留了 17% 的会话约束 | — | — | [S] |

### 2.4 推理过程内的分段与擦除

| 工作 | 编号 | 做法 | 标记 |
|---|---|---|---|
| PENCIL | 2503.14337 | 按规则擦掉已用完的思考 | [S] |
| InftyThink | 2503.06692 | 推理分段，段间只带摘要 | [S] |
| Delethink（Markovian Thinker） | 2510.06557 | 固定长度分段，携带文本状态 | [S] |

### 2.5 上下文作为环境变量

| 工作 | 编号 / 日期 | 做法 | 代码 | 标记 |
|---|---|---|---|---|
| RLM（Recursive Language Models，Alex Zhang、Omar Khattab） | 2512.24601，2025-12 | 输入作为只读变量放在 REPL 里，模型写代码去查，必要时递归调用自己；针对超长**输入** | alexzhang13/rlm，MIT，pip 可装；OpenAI/Anthropic/OpenRouter/vLLM；另有 rlm-minimal；Harbor 自带 `dspy_rlm` agent | [R] |

---

## 3. "状态派"

### 3.1 用状态取代历史

| 工作 | 编号 / 日期 | 状态的形态 | 历史怎么处理 | 关键数字 | 代码 | 标记 |
|---|---|---|---|---|---|---|
| **SKILL.state**（Badhe、Tiwari、Chung） | 2608.26263，2026-08-28 | 一份 JSON；schema 由技能作者按领域写一次；每步输出增量补丁（StatePatch） | 丢弃；每步只看规格、状态和最新观察；prompt 恒定约 1.7K–1.9K token，累计 token 从平方降到线性 | 仓库任务 200 步 0.94；100 步时累计 token 减少 16.2 倍；噪声下成功率仍 >0.95；InterCode CTF：SKILL.state 54.2，ReAct（记忆摘要）46.4，LangGraph Stateful 41.8。Table 1（Gemini-3-Flash）：ReAct 0.90 → 0.74（10 → 200 步），SKILL.state 1.00 → 0.94 | **无官方代码**，SkillExecBench 未公开。非官方：WUWeifeng710/skill-state-runtime（MIT，2026-09-27，标准库 Python，已接入 `third_party/`）、vitkuz573/skillstate（MIT，TS/npm，MCP server）、Nicolepcx/skill_state、vicnroll/open-skillstate、nachollorca/markov-agent、NosytLabs/skillstate-proxy（MIT）、JaviMaligno/delayed-relevance（无 license） | [S] 论文，[N] 笔记 |
| **PoS**（Beyond Memory: Explicit Belief States） | 2610.01415，2026-10-01 | 信念状态：实体、状态、关系，加上"还不知道的"和"还没做成的" | 不作为决策依据 | 见 §6.3 | luoyu100/PoS，MIT，2026-10-02；信念管理器、sentinel、陷阱检测、恢复、ReAct 基线和 2 个消融；ALFWorld、LOCA、RCA-100、ClinDiag 的 adapter；DashScope 的 OpenAI 兼容接口；README 说总 token 约为原始轨迹的 5 倍 | [R] |
| StateAct | 2410.02810，2024-10 | 每步写一串状态，并重复提醒目标（few-shot，固定字段） | 保留 | ALFWorld 上保留完整历史仍提升约 10% | — | [S] |
| Agent-BRACE | 2605.11436，2026-05 | agent 拆成信念模型 + 策略模型；信念是原子化自然语言命题，每条带概率；两者用 RL 联合训练 | 策略不看历史 | — | joykirat18/Agent-BRACE；LICENSE 文件是 Apache-2.0，README 徽章写 MIT；PPO/SFT 训练代码；vLLM 0.8.5；clone 2.7 GB | [R] |
| BeliefMem | 2605.05583，2026-05 | 每个观察保留多个候选结论及其概率，随新证据更新 | — | — | 未找到 | [S] |
| BB-WM（Belief-Based World Models） | 2609.00455，2026-08 | 可以用自然语言查询的信念世界模型：现在和将来哪些已知、哪些不确定 | — | — | skumar-ml/belief-world-models，**无 license**；ALFWorld/ScienceWorld/BabyAI；WALL-E 基线；`BACKEND=openrouter`；已接入 `third_party/`，只运行不复制 | [R] |
| Belief-State Engine | 2609.10036 | 部分可观测下的原则性规划 | — | — | — | [T] |
| From History to State | 2605.05413，2026-05 | 确定性追踪器生成状态块；每步只给状态和当前观察；把重复流程训练进权重（SFT + RL） | 丢弃 | — | 未找到 | [S] |
| Personalized State-Transition-Aware Memory for Clinical Agents | 2609.38490 | 临床 agent 的状态转移感知记忆 | — | — | — | [T] |
| CaveAgent | 2601.01569 | 把 LLM 变成有状态的运行时算子 | — | — | — | [T] |
| StateM | 2608.15089 | 通过 harness scaling 在 Terminal-Bench 2.1 上达到 95.3% | — | — | — | [T] |

### 3.2 记录和视图分离

| 工作 | 编号 / 日期 | 做法 | 关键数字 | 代码 | 标记 |
|---|---|---|---|---|---|
| **Scroll**（Context as an Environment；Lin Yin 等） | 2608.21690，2026-08-21 | 历史变成可执行的"会话环境"：只追加、每条有稳定地址和来源的事件日志 + 常驻沙箱 Python kernel；模型执行代码去查日志、调工具、算变量；只有 print 出来的才进下一轮上下文；超预算时只是不显示旧片段，原文可按地址找回；状态是模型定义的带类型变量 | Qwen3.8-Max：LongMemEval-S 94.8%；BEAM-10M 73.1%（比最好的已发表记忆系统高 5.1 分；另一份摘要写 91.2，矛盾）；LOCA-256K 86.7%（比最好的已发表长程 agent 高 37.4 分） | niceIrene/QwenPaw 的 `scroll-research` 分支，Apache-2.0，2026-08-26；方法约 9K 行，在 `src/qwenpaw/agents/context/scroll/`（事件日志、常驻 REPL、recall 工具）；**评测适配器在私有仓库 agentscope-ai/AgentZero**，README 说"很快开源"；其他分支 `scroll-paper-v2-dev`、`scroll-research-archive-20260825` | [R] 代码，[S] 数字 |
| InfiAgent | 2601.03204，2026-01 | 每个任务一个工作目录（计划、产物、工具输出、验证日志）作为权威记录；每步从目录快照 + 最近若干步重建上下文；消融显示持久状态"不能被长上下文压缩完全替代" | — | polyuiislab/infiAgent，**GPL-3.0**，产品化框架（pip、CLI、web UI），无论文评测脚本 | [R] |
| ESAA-Conversational（Elzo Brito） | 2606.23752，2026-06 | 事件溯源：把多个编码 agent 的对话收集成只追加日志，确定性地生成 state.md、decisions.md、tasks.json，供不同 agent 交接 | — | elzobrito/conversation-esaa，MIT，Node.js CLI + PowerShell hooks（Codex、Claude Code、Grok）；无评测 | [R] |
| NOOA（NVIDIA OO Agents） | 2607.20709，2026-07 | agent 就是 Python 对象：方法是动作，类型注解是契约，对象按引用传递、不塞进 prompt | 253 行，SWE-bench Verified 82.2%（GPT-5.5），高于 OpenCode 78.6%、Pi 78.2% | NVIDIA-NeMo/labs-OO-Agents，Apache-2.0；pip `nooa`；`nooa-bench` Harbor runner | [R] |
| State-Aware Runtime for Long-Horizon LLM Agents | ResearchGate | 概念框架和研究议程 | — | — | [T] |

### 3.3 让 prompt 的组装有依据

| 工作 | 编号 / 日期 | 做法 | 关键数字 | 代码 | 标记 |
|---|---|---|---|---|---|
| **VISTA**（LLM Agents Are Latent Context Managers） | 2606.30005，2026-06-29 | 不训练、只换接口：带类型、可寻址的工作记忆块；仪表盘显示每块的 token 数、新旧、访问次数；删掉的块完整归档、可找回。立场：管理能力潜藏在模型里，缺的是让它看见自己 | LOCA 上提升四个模型，Gemini-3-Flash 22.7% → 50.7%；38/75 vs ReAct 17/75 | binyxu/VISTA，**暂无 license**，2026-07-30；`context_workspace`；LOCA、BrowseComp-Plus、AMA、GAIA、SWE-Bench、LoCoBench 的 adapter；`prototype/` | [R] 代码，[S] 数字 |
| ContextRender | 2609.37743，2026-09-29 | 维护工具结果之间的依赖图，追踪后续步骤实际复用了哪些早期结果，据此决定放什么进 prompt | AppWorld 上 6K 历史预算接近或超过完整历史，成本降低 10%–32% | 未找到（搜索里有一个 Zenodo 记录 22904264，未核实） | [S] |
| ContextPipe | 2609.00749，2026-09 | 把组装上下文类比成数据库执行查询：分阶段规划和优化、缓存感知的优化器、可审计的执行追踪；自己承认缓存命中率下降 | SWE-bench Pro 子集上 token 降低 31% | 已集成进 matrixorigin/Astra（Rust 运行时 + TS SDK），Apache-2.0；作者 fork XuPeng-SH/Astra；无论文复现脚本 | [R] |

### 3.4 带不确定性、来源和准入控制的记忆

| 工作 | 编号 | 做法 | 标记 |
|---|---|---|---|
| When Does Belief-Based Agent Memory Help? 可靠性条件更新与来源上限的投毒防御 | 2606.22030 | 按来源限制可信度，防记忆投毒。**注意**：之前文档里把这个编号写成 "Nous"，是混淆。Nous 是另一篇（下一行） | [S] |
| Nous：Learning and Certifying Memory Decisions Before Source Calibration | 2610.00094 | 记忆决策的学习与认证 | [T] |
| Epistemic Admission in Shared Agent Memory | 2609.30813 | 共享记忆的认知准入基准与诊断 | [S] |
| When Stale Constraints Go Unchecked | 2608.25553 | 继承的 agent 记忆里，预算受限的校验失败 | [T] |
| Σ-Mem | 2607.27958 | 多 agent 系统的在线可靠性记忆 | [T] |
| ROAM | 2609.09778 | 用语义关系组织原子记忆 | [T] |
| Infini Memory | 2606.10677 | 可维护的主题文档作为长期记忆 | [T] |

---

## 4. 事务化、可验证的状态（已经很拥挤，我们不正面做）

| 工作 | 编号 | 做法 | 标记 |
|---|---|---|---|
| MemTX | 2607.23929 | 状态写入要经过"暂定 → 可行动"的生命周期，每条记录带证据和来源，先校验再提交 | [S] |
| MemTxn | 2607.27834 | 记忆更新的事务边界，支持有来源支撑的更新和完整状态恢复 | [T] |
| PatchOptic | 2607.05483 | 只提交经过验证的结构化修改 | [S] |
| PatchBoard | 2605.29313 | 按 schema 约束状态修改；也覆盖多 agent | [S] |
| State-Bound Evidence / Looping Is Not Reliability | 2607.24604 | 把证据和代码修复的状态绑定 | [S] |
| To Know is to Construct：Schema-Constrained Generation for Agent Memory | 2604.20117 | schema 约束的记忆生成 | [S] |

我们的自适应绑定里的部件 ④（写入校验）和这一组重叠，所以它只是方法的一部分，不作为卖点。

---

## 5. 让模型写程序、演化策略或 harness

### 5.1 模型自己写环境模型或记忆程序

| 工作 | 编号 / 日期 | 做法 | 代码 | 标记 |
|---|---|---|---|---|
| Schema（Agentic Program Induction） | 2609.39140，2026-09-30 | agent 把对环境的理解写成可执行程序，再用交互历史检验；ARC-AGI-3 从 58.7% 提到 99.2% | 项目页 schema-harness/schema-harness.github.io 没有代码，只链到 HF 数据集 `schema-harness/arc-agi-3-schema-traces`（未核实） | [S] |
| M* | 2604.11811，2026-04 | 为每类任务演化一个记忆程序（数据 schema、存储逻辑、工作流指令），离线演化，agent 固定 | 有 GitHub | [S] |
| ALMA | 2602.07755，2026-02-08 | 元 agent 搜索写成代码的记忆设计（跨会话） | — | [S] |
| MemEvolve | 2512.18746 | 同上 | — | [S] |
| Recuris | 2608.24876，2026-08 | 工作记忆跟踪进度；元 agent 跨任务修补记忆组件 | 有 GitHub | [S] |
| Experience Funnel | 2609.08919 | 状态—策略交替循环的自进化 agent | — | [T] |

### 5.2 外循环优化策略、playbook 或 harness

| 工作 | 编号 / 日期 | 优化对象 | 标记 |
|---|---|---|---|
| Dynamic Cheatsheet | 2504.07952 | 跨任务演化的策略笔记 | [R] GitHub |
| ACE | 2510.04618，2025-10-06 | 演化 playbook | [S] |
| ACON | 2510.00615，2025-10 | 用成功和失败轨迹的对比，以自然语言优化压缩准则 | [S] |
| Meta-Harness（Yoonho Lee、Chelsea Finn） | 2603.28052 | 让编码 agent 读原始日志、改写整个 harness 的代码 | [S] |
| **ContextEvo** | 2609.34649，2026-09-28 | 在 Pi harness 上演化输入组装、历史维护、上下文编排；声称打平或超过 Codex、OpenCode、OpenClaw。无代码 | [S] |
| GEPA、Darwin Gödel Machine、Hyperagents、AutoMem、Meta Context Engineering | — | CLM 论文相关工作里列出的元优化方法 | [N] |
| Memory Reward Inflation in Self-Improving LLM Agents | 2608.00017 | 自改进 agent 的记忆奖励膨胀 | [T] |
| CLM 的 ICL | — | 演化写进 prompt 的上下文管理技能说明 | [R] |

---

## 6. 关键实证证据（我们论证要用到的）

### 6.1 delayed-relevance 复现项目（JaviMaligno，2026-10-02）[R]

- **仓库**：github.com/JaviMaligno/delayed-relevance，**无 license**，只读参考，固定在 `53ceaf4`，已接入 `third_party/references/`。
- **博客**："When the Fact Stops Being True"（javieraguilar.ai）；论文草稿 *The Runtime Is Not Infrastructure: A Replication of SKILL.state*。
- **设置**：Claude Haiku 4.5、Sonnet 5；按 SKILL.state §4.1 重新实现仓库环境；T ∈ {10, 25, 50, 100, 200}；上下文密度是原论文的 1.2–1.4 倍。比较四种运行时：ReAct、Memory、Stateful、SKILL.state。
- **成本**（T=50，3 个种子，开缓存）：

| 运行时 | 得分 | 原始 token | 计费 token | 缓存节省 |
|---|---|---|---|---|
| ReAct | 1.00 | 82.6 万 | **15.2 万** | 82% |
| Stateful | 1.00 | 87.3 万 | 87.3 万 | 0% |
| Memory | 0.87 | 31.3 万 | 31.3 万 | 0% |
| SKILL.state | 1.00 | **10.9 万** | 10.9 万 | 0% |

  - SKILL.state 的优势：原始 token 7.54 倍，计费只剩 1.39 倍。
  - 按 token 排序：SKILL.state < Memory < ReAct < Stateful；按钱排序：SKILL.state < **ReAct** < Memory < Stateful。
  - 同样的内容，可变状态放在历史前面（Stateful），成本是放在后面（ReAct）的 5.7 倍。
  - 结论："压缩上下文和缓存上下文相互冲突"。
- **延迟相关**（隔离通知在 k 步后才用上，Haiku，k=40）：

| 条件 | 正确率 | 95% CI |
|---|---|---|
| ReAct，完整历史 | 2/12 = 17% | 5–45% |
| SKILL.state，没有对应字段 | 0/24 = 0% | 0–14% |
| SKILL.state + 原文照搬提醒（981 字符） | 16/24 = 67% | 47–82% |
| SKILL.state + 提炼后的提醒（67 字符，3 个字段） | 24/24 = 100% | 86–100% |
| notes 字段（重复测量后） | 约 21% | 9–40% |
| oracle schema（有对应字段） | 100% | — |

  - 结论："失败不在注意力或距离，而在表示方式"；"提醒要提炼字段，不能只重贴记录"。
- **事后更正**（第 t 步入库，第 t+10 步更正说没完成；按依赖步计数；种子 4、10、6 × 2 次重复）：

| 模型 | 运行时 | 应用了更正 | 其他错误 | 无动作 | 不合 schema 的补丁 |
|---|---|---|---|---|---|
| Haiku 4.5 | ReAct | 3/44 = 6.8% | 0 | 0 | 0 |
| Haiku 4.5 | SKILL.state | 44/44 = 100% | 0 | 0 | 0 |
| Sonnet 5 | ReAct | 15/38 = 39.5% | 5 | 1 | 0 |
| Sonnet 5 | SKILL.state | 49/49 = 100% | 21 | 10 | 66 |

  - 合计：显式状态 93/93，完整历史 18/82。
  - ReAct 的失败是"按场景全有或全无"：要么整段都对，要么整段都错。
  - Sonnet 上显式状态的主要失败是"符合 schema、但语义抄错的补丁"。
- **方法论教训**：先测噪声底（同种子 8 次重复）；至少四个结论在测噪声后被推翻；按"依赖步数"选种子，信号多 4 倍；它的第二个环境 Repo 被撤回，因为失效距离固定为 2。

### 6.2 Supersede [R]

LongMemEval 知识更新子集（78 题，oracle 版），完整上下文 vs 300 字符笔记：

| 模型 | 完整上下文 | 笔记 |
|---|---|---|
| gpt-4.1-mini | 82 | 63 |
| gpt-4.1 | 91 | 64 |
| gpt-5.4 | 92 | 77 |

给更多笔记空间也恢复不了差距。

### 6.3 PoS（仓库 `docs/results.md`）[R]

Qwen3.7-Plus：

| 基准 | Raw | 最强记忆基线 | PoS |
|---|---|---|---|
| ALFWorld | 62.69 | 72.39 | 88.81 |
| LOCA（7 个长度 × 75 例汇总） | 43.62 | — | 56.38 |

同一仓库里三个模型的范围：ALFWorld 上 Raw 62.7/81.3/93.3 → PoS 88.8/94.0/97.0；LOCA 上 Raw 43.6–69.7，PoS 56.4–74.3。RCA-100 上总 token 是原始轨迹的 5.06 倍。

论文摘要里另有 [S]：ALFWorld 相对提升 22.68%；RCA-100 联合准确率相对提升 37.89%；消融显示校验和陷阱恢复各自都有独立贡献。

### 6.4 自我条件化与重试

| 工作 | 编号 | 结论 | 标记 |
|---|---|---|---|
| The Illusion of Diminishing Returns | 2509.09677 | 上下文里自己的错误会推高后续出错率（self-conditioning）；thinking 模型较能抵抗。dict_sum 任务：多轮维护累加和；仓库无 license，数据在 HF | [S] |
| Why Retrying Fails | 2605.08563 | SWE-bench Verified 上拟合出污染比 ε1/ε0 = 7.1：失败尝试留在上下文里时，重试的单步错误率是干净上下文的 7.1 倍；证明"清空后重启"占优。和 Manus 的"把错误留着"冲突 | [S] |
| FSM 执行研究 | 2511.14777 | 早期的状态错误会一路传播 | [S] |
| Chroma Context Rot | 博客 | 性能在每个长度增量上都在下降，不只是快满时 | [S] |
| Your LLM Agents are Temporally Blind | 2510.23853 | 工具使用决策与人类时间感知不一致 | [T] |

---

## 7. 工业界与实践

| 来源 | 内容 | 和我们的关系 | 标记 |
|---|---|---|---|
| Manus 工程博客（2025-07-18） | 保持前缀稳定以吃 KV 缓存；todo.md 不断重写，把目标"复述到上下文末尾"；文件系统当上下文；工具只遮蔽不删除；"把错误留在上下文里"；每任务约 50 次工具调用 | 尾部状态的工程先例；但 todo 通过工具调用追加，旧副本会累积，没有消融 | [S]（页面被拦） |
| Anthropic context editing + memory tool（2025-09-29） | 自动清理旧工具结果（文档承认会使缓存前缀失效）；/memories 跨对话持久；API 指示"做任何事之前先看记忆目录" | 记忆在上下文之外 | [R] |
| Claude Code 文档 | 自动压缩；TodoWrite/Task 工具在较新模型上**默认关闭**，因为"Claude 不需要书面清单就能跟踪多步工作，而工具定义和提醒会占上下文" | 前沿模型上显式状态可能多余，和 H5 相关 | [R] |
| MemGPT / Letta | 2310.08560：模型自编辑的核心记忆块，渲染在 system prompt 里；放在前面，每次编辑打断缓存。Letta PR #199（2026-09-30 合并）绕过这个问题 | 状态块的最早先例 | [S] 论文，[R] PR |
| Letta Context-Bench | 上下文工程能力基准（2025-10） | 可能的评测 | [S] |
| planning-with-files（othmanadi） | hooks 每轮在尾部重新注入一个固定形状的计划块，"为了前缀缓存一致" | 尾部状态的实践版 | [R] |
| "Your Agent's Memory Is a Liability: Track State, Not History"（Towards AI） | 社区文章，主张跟踪状态而不是历史 | 趋势的社区表述 | [S] |
| TokenPilot / LightRSI（zjunlp） | 2606.17016：入口处稳定前缀；按批、按生命周期驱逐。代码在 zjunlp/LightRSI（MIT），TS 插件和代理，接 OpenClaw、Codex、Claude Code、OpenCode | 缓存感知 | [R] |
| CADOC / Self-GC / SmoothAgent | 2609.37012（2026-09-29）/ 2607.00692 / 2607.00151：缓存感知的批量替换、缓存感知的提交、前瞻 KV 预计算 | 系统侧 | [S] |
| 非前缀 KV 复用 | Prompt Cache、CacheBlend、EPIC、PIE、Memento（CLM 论文列出） | SCR 的相关工作 | [N] |

---

## 8. 基准全表

### 8.1 选用的

| 基准 | 测什么 | 规模 | 真值 | 已发表可对比的数字 | license / 数据 | 在我们这里的用途 |
|---|---|---|---|---|---|---|
| **仓库探针**（自建） | 依赖一条早先事实的决策 | 按需生成 | 构造时精确 | delayed-relevance | 本仓库 | H1–H5、I3、I4 |
| **ALFWorld** | 家务任务；需求可预测 | valid-unseen 134 局，每局 50 步上限 | TextWorld facts（已验证） | PoS | MIT；GitHub release 下载已验证 | H4、真实基准 |
| **LOCA-bench**（hkust-nlp，2602.07962） | 从模拟 Canvas、邮件、WooCommerce、BigQuery 等服务收集和组合事实；需求中等可预测 | 15 个环境 × 5 个种子 = 每档 75 例；8K–256K；≤100 次工具调用 | 生成的环境数据库 + 部分任务的 `groundtruth_workspace` | PoS、Scroll、VISTA；部分模型 8K 时 >70%，256K 时跌到 2.7% | MIT；数据本地生成；需要 Node.js、MCP | H4、和状态派比 |
| **LongMemEval-S + 知识更新子集** | 读完对话后才问问题；需求不可预测 | S 版 500 题，每题约 11.5 万 token；KU 78 题 | 答案 + `has_answer` 证据轮次 | Scroll（S 版 94.8）、Supersede（KU） | MIT；**数据在 HF / Google Drive，本环境下载不了** | H4、不可预测一侧 |
| **Supersede** | 同上，带完整上下文和有界笔记两种模式 | 78 题 | 同上 + 合成模式追踪新旧值 | 见 §6.2 | Apache-2.0；`rollout.py` 与框架无关 | H3 锚点 |

### 8.2 备选

| 基准 | 说明 | 标记 |
|---|---|---|
| BEAM（mohammadtavakoli78） | 10 种能力，含知识更新和矛盾；对话在 git 仓库里（约 4.2 GB）；每段对话 20 题；128K 档 20 段，10M 档 10 段（光摄入就要 1 亿 token 以上）；只有 Scroll 报过数。用 128K/1M 档 | [R] |
| STATE-Bench（微软，2026-05） | 450 个 τ 风格任务，3 个领域，检查最终数据库状态；GPT-5.4 用户模拟器和评审被协议锁定，官方每任务跑 5 次；`BaseAgent.generate_next_turn(conversation)` 可换上下文；没有可对比的已发表数字 | [R] |
| InterCode CTF | 100 题，Docker，只看 flag；只有 SKILL.state 报过（Gemini-3-Flash、Gemma-4-31B、Qwen-3-8B） | [R] |
| τ-bench | 50 airline + 115 retail，最终数据库状态 | [R] |
| AppWorld | 数据库状态评测；数据是 S3 包；只有 ContextRender 报过 | [R] |
| RCA-100 | 103 例诊断；答案在 PoS 仓库，观察来自 aliyuncs（本环境 403）；每回合约 35.6 万 token | [R] |
| Prosus Vending Bench（Vending-Bench 的开源复现） | 30 或 365 天的库存、订单、现金模拟；已是 Harbor 任务；Apache-2.0；每次 $0.08–13.55；参考脚本 30 天 506 次工具调用；基线是最近窗口 + 笔记工具；按最终余额打分 | [R] |
| TextQuests | 25 个 Infocom 文字游戏，默认 500 步，上下文 >10 万；已记录的失败是以为自己还拿着已丢掉的物品；跑一遍约 6.38 亿 token（5.83 亿命中缓存）；MIT | [R] |
| UltraHorizon | 发现隐藏规则；已记录 agent 锁死在早期假设上；无 license；LLM 评审（DeepSeek-R1）；作者建议每设置 32 次重复 | [R] |
| MemoryArena | 多会话 agent 任务，后面依赖前面；766 题；数据 CC-BY-4.0，代码无 license；记忆系统并不稳定胜过长上下文 | [R] |
| SlopCodeBench | 需求不断变化时扩展自己的代码而不引入回归；36 题 196 个检查点；到最后一个检查点严格通过率 0.5%；已是 Harbor 数据集 | [R] |
| dict_sum（Illusion of Diminishing Returns） | 多轮维护累加和，直接测自我条件化；无 license，可自己重写 | [R] |

### 8.3 不用的（及原因）

| 基准 | 原因 |
|---|---|
| StateMemBench、WorldLines、BeliefShift（2603.23848，2400 条轨迹） | 没找到代码或数据 |
| SkillExecBench | 未公开 |
| DreamBench-SWE（2608.20664，iroiro147/dreambench-swe） | README 说隐藏的评分 oracle 不公开，无法给自己的运行打分 |
| MemoryAgentBench（2507.05257） | 数据只在 HF；我们关心的方法都没在上面报过数。内容：4 种能力 AR、TTL、LRU、CR，只有 CR（FactConsolidation，事实被覆盖）是可变状态；2071 题，每段上下文 10.3 万–144 万 token；所有方法在多跳 CR 上 ≤6–7%。如果以后要补"流式更新"一侧，可以重新考虑 |
| CooperBench（652 个双 agent 协作任务） | harness 固定，和本课题无关 |
| ClinDiag | 环境提供者和评审固定为 Qwen3.7-Plus |
| LongMemEval-V2（2605.12493） | 干草堆 2500 万–1.15 亿 token，读取器固定 Qwen3.5-9B，含多模态；Codex 这类基于文件的 agent 已达 69.9%（RAG 42.8–58.6%，AgentRunbook-C 74.9%） |
| AMA-Bench | 对静态轨迹做问答，没有交互；真实子集 2496 题（约 208 条轨迹，平均约 5.7 万 token）；AMA-Agent 57.22% |
| TBLite | CLM 已经领先 Summary（73.7 对 67.0），当"不退步"对照意义不大 |
| Vending-Bench 1/2（Andon Labs 原版） | 不公开 |
| TALES / Jericho、BALROG | 失败主要来自技能而不是状态；NetHack 很贵 |
| TextArena、τ3-bench、TheAgentCompany、OdysseyBench、LoCoBench-Agent、LongCLI-Bench、LHTB、EdgeBench | 太短、太重、太贵，或没有功能性评分 |

### 8.4 CLM 论文用的基准

ContextBench（自建，未发布：Needle Retention、Sudoku Sketchpad、KV Store、Log Triage）、TerminalBench 2.1（89 题）、TBLite、BrowseComp-Plus（830 题）、四个 AlphaEvolve/OpenEvolve 数学优化问题（circle packing、Heilbronn、min-max/min-distance、Erdős minimum overlap）、EdgeBench-10（48 个公开任务中的 10 个）、Software World（自建：requests、urllib3 和 4 个下游包）。

**结构性观察**：现在有两个主战场。LOCA-bench 是状态派的（PoS、Scroll、VISTA）；BrowseComp-Plus 是上下文管理派的（CLM、VISTA、ACM、SelfCompact、ContextPilot）。VISTA 横跨两边。

---

## 9. Harbor 数据集注册表（CLM 能直接跑的）

CLM 跑在 Harbor（`harbor==0.16.1`）上。官方注册表有 80 个数据集，下载于 2026-10-02 [R]。下面是和本研究相关的部分，括号里是任务数：

| 类别 | 数据集 |
|---|---|
| 智能体 | terminal-bench@2.0（89）、terminal-bench-pro（200）、swebench-verified、swebenchpro（731）、swe-lancer-diamond（463）、gaia（165）、featurebench（200）、crustbench（100，整仓 C 转 Rust） |
| 长程优化与科研 | algotune（154）、gso（102）、mlgym-bench（12）、ml-dev-bench、replicationbench（90）、researchcodebench（212） |
| 推理与代码 | aime、gpqa-diamond、livecodebench@6.0、usaco、arc_agi_2（167）、reasoning-gym-hard、ineqmath、kumo |
| 数据与领域 | dabstep（450）、bixbench（205）、spreadsheetbench-verified（400）、financeagent（50）、ade-bench（48，dbt/SQL）、scale-ai/swe-atlas-qna（124） |
| 安全 | binary-audit（46，在编译后的二进制里找后门） |
| 多 agent | cooperbench（652） |
| 不退化检查 | gpqa-diamond、aime、simpleqa |
| 训练环境集 | termigen-environments（3566 个终端环境）、seta-env（1376 个） |

Harbor 自带的 agent：claude_code、codex、openhands、terminus_2、mini_swe_agent、pi、dspy_rlm。有 `harbor leaderboard submit` 命令，也有 adapter 模板和向导。

GitHub 上的 Harbor adapter 列表里没有 MemoryAgentBench、LongMemEval-V2、AMA-Bench、TextQuests、BALROG、UltraHorizon、LOCA-bench。本地注册表里只有 cooperbench；tau3-bench（375 题）的 adapter 只在 `harbor-framework/harbor/adapters/` 里。

---

## 10. 撞车监控

**每次动笔前一周重新检索**，关键词：

- state vs history、explicit state ablation、tombstone、supersession、superseded facts、stale memory
- delayed relevance、retroactive invalidation、schema coverage
- act-time retrieval、bind at act、recall before action
- write error vs read error、encoding vs decoding error、reconstruction error agent
- event count vs context length、state tracking scaling
- tenure crossover、memory architecture ranking
- （遗忘方向）agent unlearning、execution-state unlearning、deletion is not forgetting、behavioral residual、provenance purge、selective replay、record omission linear attention、forgetting benchmark

**重点盯的作者和项目**：CLM 作者（`research_ledger`、`ctx_archive` 这些被删掉的开关）、delayed-relevance（可能很快挂 arXiv）、Scroll 的评测仓库 AgentZero（开源后可直接比）、PoS、Supersede、Ground Truth First。

---

## 11. 记忆遗忘与删除（2026-10-06 检索）

起因：用户提出"能不能用 CLM 做 memory unlearning"。检索结果是，这个方向在 2026 年 8–9 月已经非常拥挤，"现在的记忆系统只管记、不管删"这个前提已经不成立。讨论和结论见 [idea_bank.md](idea_bank.md) F1。以下全部只看过搜索摘要 [S]。

| 工作 | 编号 / 日期 | 做了什么 | 和 CLM 遗忘的重叠 |
|---|---|---|---|
| **Forgetting Without Restarting：Execution-State Unlearning for Stateful LLM Agents**（Chao Yao 等） | 2609.04875，2026-09 | 形式化"执行状态遗忘"：收到遗忘请求后，agent 要表现得像从没见过目标。长程 agent 会积累压缩摘要、明文记忆、待执行的工具计划和 KV 缓存。用诱导式、随机采样、不依赖字符串的行为测试，覆盖 3 个 agent 套件、9 个基线、3 个模型家族。结论：删记忆条目对泄漏毫无改变；靠指令让模型忘记，在诱导下会失效；只删来源，80% 的回合里仍按被撤回的偏好行事。方法是溯源引导的选择性重放：用溯源图定位注入点，把 KV 缓存裁到检查点，再干净地重放反事实后缀。与完全重置无法区分，重算 token 最多少 9 倍 | **最接近**，基本覆盖"删除 + 按反事实重放"这条路 |
| **Deletion Is Not Forgetting：Behavioral Residuals of Deleted Memories in Stateful LLM Agents**（Jiabei Liu、Mingyang Li） | SSRN 7428966，2026-09-07 | 提出"行为反事实遗忘"。原生删除让来源标记在所有情况下都找不回来，但 agent 的选择仍向被删的偏好偏移 +59 到 +100 个百分点。用 GPT-4o-mini 做合并模型时，100% 把来源偏好编码进派生物，同时 100% 洗掉审计标记，行为残留 +96 个百分点。按依赖关系清除后降到 0% | "派生产物把信息洗走"已被证实 |
| **Exact Record Omission in Delta Attention：A Transport Criterion, Its Cost, and a Replay Certificate**（Vishwajith Ramesh） | 2609.06872，2026-09 | 线性注意力（delta rule）的记忆能否回到"从没存过这条记录"的状态。记下记录进入时的差量、跟着后续更新传递、删除时减掉：仍然不行，4096 个 token 之后还留着约 4.5% 状态范数的印记。只有检查点重放能精确删除，代价和重放的后缀长度成正比 | 覆盖了"线性注意力状态里有残留"的原理；SCR 的 fork 模式是它的一个具体实例 |
| Agentic Unlearning：When LLM Agent Meets Machine Unlearning | 2602.17692，2026-02 | 信息回流：只从参数里删掉，检索机制仍能读到残留，又被写回记忆；参数里的残留也能再生成被删内容。必须同时处理参数和记忆 | 概念框架 |
| MemLeak：Diagnosing Information Leaks in Multimodal Agent Memory | 2606.29788，2026-06 | 删掉文本条目后，事实仍能从保留的用户图片里恢复；提出按"可删除性"给记忆表示分类的信息溯源图（IPG） | 溯源图 |
| Control-Plane Placement Shapes Forgetting（Dongxu Yang） | 2606.15903，2026-06 | 13 种系统配置；ForgetEval，1385 个用例，遗忘分 5 族（取代、衰减、失忆、清除、漂移），10 类攻击 | 遗忘基准已有 |
| Can an AI Assistant Really Forget? Auditable Deletion from Addressable Memory | 2607.27539，2026-07 | 在冻结的 Gemma 3 里装一个支持向量门；删除后，逐条记录的攻击仍能区分"删过"和"从没存过" | 可审计删除 |
| What a Deletion Certificate Covers, and Where It Expires | 2607.12204，2026-07 | 支持向量记忆的可审计删除 | 同上 |
| Towards Reversible Forgetting | 2608.18177，2026-08-18 | 记忆分为 active、dormant、retired 三态，可以重新激活；区分暂时抑制和永久删除 | 和"作废 vs 遗忘"的区分相关 |
| TEPA：Revoking Stale Memories for Conflict-Robust Language Agents | 2608.07429 | 撤销过时记忆 | [T] |
| What Should an Agent Forget? Separating What Is Stored from What Is Used | 2609.10263 | 区分"存了什么"和"用了什么" | [T] |
| GateMem | 2606.18829 | 多主体共享记忆的治理基准：效用、访问控制、主动遗忘 | 基准 |
| Forgetful but Faithful | 2512.12856 | 隐私感知的认知记忆架构和基准 | 基准 |
| FSFM | 2604.20300 | 受生物启发的选择性遗忘 | — |
| Always-On Agents 综述 | 2606.30306 | 一条遗忘请求必须传到每一层派生数据，否则只是改了检索；还要防派生数据和删除模式本身造成的推断泄漏 | 综述 |
| Leak@k；Do LLMs Really Forget? | 2511.04934；2609.36612 | 权重遗忘在随机采样下会重新泄漏；隐藏状态里仍有编码 | 权重侧 |

产品侧：Mem0 的更新阶段有 ADD/UPDATE/DELETE/NOOP，遗忘时按语义检索后按 ID 删除；Letta 遗忘时由 LLM 从编号列表里挑出要删的条目。两者都只删文本条目，不处理派生产物。

**CLM 代码里和遗忘直接相关的事实 [R]**（`suffix_cache_reuse/README.md`、`overlay.py:1178` 附近）：
- SCR 默认 `KVREUSE_SSM_MODE=fork`：编辑后，线性注意力层（Qwen3.6-27B 64 层中的 48 层）从"上一轮 prompt 末尾保存的 recurrent state"继续算。这个状态是在包含被删内容的上下文上算出来的；编辑只在 16 个全注意力层里生效。
- 被删段之后保留下来的 token，KV 也是在旧前缀下算的。README 原话是这种过时状态"有时反而有益，保留了更多过去的信息"。
- 另外两种模式：`strict`（只在保存的状态恰好覆盖被复用的那一段时才搬）、`none`（不恢复状态）。
- 论文验证的是准确率，没有验证遗忘。
