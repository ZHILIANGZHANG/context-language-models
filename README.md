# Self-Paging Language Models

**让模型自己管理 KV 缓存：像 CLM 把上下文管理交给模型那样，把 KV 的调度管理直接交给模型。**

**2026-11-16 实际运行归档：** [Qwen3-8B SFT 训练、200 次评测、代码与证据](experiments/self_paging_20261116/README.md)。包含 431 步训练、9 次验证和可复核的逐题结果。名义 `full` 的 91% 结果仍执行了模型换页，**不是严格完整历史 Full 基线**；原始权重基线和 RL 尚未结算，详见归档限制。

**原始权重基线中间快照：** [Recent / QuestPage 已完成 224/300 次评测](experiments/self_paging_baselines_partial_20261116/README.md)，含 37 个共同世界的同题 SFT 对照和环境调用错误统计；尚非最终结果。

本仓库最初 fork 自 [facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models)（CLM，arXiv 2609.37725），现在是一个研究工作区：

- **CLM 原始实现**：保留在 [`clm/`](clm/) 和 [`suffix_cache_reuse/`](suffix_cache_reuse/)，是最主要的对照组（模型决定、作用在文本上）；论文原文在 [`clm.pdf`](clm.pdf)；
- **新主线代码**：[`self_paging/`](self_paging/README.md)：已有页表、`PagedKV` 参考引擎、单次前向训练掩码和等价性测试；协议、环境、基线、训练还在写；
- **旧主线代码**：[`state_study/`](state_study/) 的仓库探针，复用为新主线的测试床 T1 和残留实验 E2；
- **研究文档**：在 [`docs/`](docs/)。新会话从 [`CLAUDE.md`](CLAUDE.md) 开始读。

核心形式化：KV 是一份只追加的日志，每个 token 的 KV 只在写入时算一次；模型能改的只有"驻留集"（哪些页在显存里，受预算 B 约束），动作是换出、换回，以及对照条件里的删除。由此得到零重算、精确可逆、单次前向训练三个工程性质，以及"删除不等于遗忘"这个科学问题。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/research_line.md`](docs/research_line.md) | **当前主线**：形式化、P1–P4、空位表、K1–K5、测试床、基线、代价口径、分阶段计划、风险 |
| [`docs/baseline_protocol.md`](docs/baseline_protocol.md) | **公平对比协议**：和 H2O、SnapKV、StreamingLLM、Quest 等 KV 剪枝 / 换页方法怎么比；训练对齐的对照、预算统一计量、信息与调参对齐、基准选择、开发任务 D10–D13 |
| [`docs/runbook.md`](docs/runbook.md) | **运行手册**（给执行实验的 agent）：硬件、软件、模型、数据和基准的准备；阶段 0–9 的实验顺序、命令和验收标准；汇报格式；剩下的开发任务 D1–D9；默认超参；排错 |
| [`self_paging/README.md`](self_paging/README.md) | **代码文档**：核心概念、每个文件做什么、主要接口、episode 记录格式、测试状态、已知限制 |
| [`docs/experiment_plan.md`](docs/experiment_plan.md) | **实验方案 v2（Qwen3-8B）**：模型的影响（全局位置和 YaRN）、tool call 接口、基准和测评协议（含 LongMemEval-S）、SFT、RL、SGLang overlay 与 vLLM、代码进度、实验矩阵、算力和时间线 |
| [`docs/data_and_training.md`](docs/data_and_training.md) | **SFT 与 RL 的数据和训练设计**："需要"的定义、Belady 计划、特权提示生成、轨迹重标注与反事实验证、拒绝采样、RL 的 episode、奖励、代价、题库过滤、阶段、变体 |
| [`docs/algorithm_design.md`](docs/algorithm_design.md) | **算法与实验设计**：协议、推理和训练实现、代价折算、基线实现、等价性测试、E0–E2 协议、SFT/RL、待实现清单 |
| [`docs/related_work.md`](docs/related_work.md) | **相关工作全集**：§0 是新主线的撞车表；§1–§9 是旧主线阶段的调研；§10 撞车监控 |
| [`docs/research_log.md`](docs/research_log.md) | **研究日志**：决策时间线、纠正过的错误、已验证事实、环境限制与搭建命令、待核实清单、下一步 |
| [`docs/clm_notes.md`](docs/clm_notes.md) | **CLM 笔记**：harness 机制、配置、ICL、RL、SCR、成本口径、我们发现的漏洞 |
| [`docs/idea_bank.md`](docs/idea_bank.md) | **idea 库**：所有考虑过的方向及状态；F1 是用户原始视频 KV 换页 proposal 的评估 |
| [`docs/archive/`](docs/archive/) | 旧主线"状态 vs 历史"的全部文档（主线、算法设计、提案、文献地图），以及视频 proposal 原件 |
| [`third_party/README.md`](third_party/README.md) | **第三方清单**：每个 submodule 的上游地址、固定的 commit、license 和用途 |
| [`CLAUDE.md`](CLAUDE.md) | 给 Claude Code 新会话的项目记忆：先读什么、现状、约定 |
| [`clm/README.md`](clm/README.md) | CLM 上游的原始 README |

## 目录结构

```
.
├── CLAUDE.md                新会话的项目记忆
├── clm.pdf                  CLM 论文原文
├── docs/                    主线、算法设计、相关工作、研究日志、CLM 笔记、idea 库；archive/ 是归档
├── clm/                     CLM 原始实现（harness、ICL、RL patch）—— 主要对照组
├── suffix_cache_reuse/      CLM 的 SGLang 推理服务优化（SCR）
├── third_party/             旧主线选的方法和基准
│   ├── methods/             pos、skill-state-runtime、scroll、vista、rlm、selfcompact、acm、belief-world-models
│   ├── benchmarks/          loca-bench、alfworld、longmemeval、supersede、state-bench（可选）、beam（可选，约 4 GB）
│   └── references/          delayed-relevance（只读的设计参考，没有 license）
├── scripts/
│   └── setup_third_party.sh 按分组初始化 submodule
├── self_paging/             新主线代码：页表、PagedKV 参考引擎、掩码、等价性测试
└── state_study/             旧主线代码：probes/（仓库探针，复用为 T1）、pilots/（冒烟测试记录）、tests/
```

## 快速开始

```bash
git clone https://github.com/ZHILIANGZHANG/context-language-models.git
cd context-language-models

# 1.（可选）拉取旧主线的第三方依赖
scripts/setup_third_party.sh core
#    其他分组：methods | benchmarks | references | optional | all

# 2. CLM（对照组）的安装方式与上游一致
pip install -e .

# 3. 仓库探针的测试（只需标准库 + pytest）
python -m pytest state_study/tests
```

旧主线各基准的环境依赖见 [`third_party/README.md`](third_party/README.md) 和[归档的提案](docs/archive/state_vs_history/proposal.md) §10。

## 当前进度

新主线（2026-11-15 起）：

- [x] 评估用户的视频 KV 换页 proposal（[`docs/idea_bank.md`](docs/idea_bank.md) F1）
- [x] 确定主线"让模型自己管理 KV 缓存"，完成设计文档和撞车检索
- [x] 具体实验方案（[`docs/experiment_plan.md`](docs/experiment_plan.md)）
- [x] 改用 Qwen3-8B；实验方案 v2（实现、训练、数据、代码、测评）
- [x] 阶段 0 的核心：`self_paging/` 的页表、`PagedKV` 参考引擎、掩码，13 个等价性测试在 Qwen3 结构上通过
- [ ] 阶段 0 的其余部分：协议、T1 环境、基线和 Belady、HF 循环、E0–E2 脚本
- [ ] 阶段 1：E0 提升空间、E1 零样本协议、E2 残留（用户 GPU，Qwen3-8B），做继续/止损判断
- [ ] 阶段 2–3：SFT、RL，在 T1 和 BrowseComp-Plus 上和 CLM、SideQuest 式、Quest 式等基线比较
- [ ] 阶段 4：视频扩展或接入服务框架，写作

旧主线（已归档）留下的：仓库探针和全部条件（20 个测试通过）、ALFWorld 逐步真值、两轮冒烟测试、PoS 环境搭建，详见 [`docs/research_log.md`](docs/research_log.md)。

## License

- `clm/` 和 `suffix_cache_reuse/` 沿用上游的 [CC BY-NC 4.0](LICENSE)，另见 [NOTICE](NOTICE)。
- `third_party/` 下的每个 submodule 遵循各自上游的 license，见 [`third_party/README.md`](third_party/README.md)。**VISTA、belief-world-models、delayed-relevance 没有 license**，只作为基线运行或设计参考，不要把它们的代码复制进本仓库。
