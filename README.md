# Where Agents Keep the Truth

**长程 Agent 上下文表示研究工作区：状态 vs 历史，结构来源、错误归因与绑定时机。**

本仓库最初 fork 自 [facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models)（CLM，arXiv 2609.37725），现已改造成一个完整的研究工作区。它由三部分组成：

- **CLM 原始实现**：保留在 [`clm/`](clm/) 和 [`suffix_cache_reuse/`](suffix_cache_reuse/)，在研究中作为"自由形式上下文编辑"这一对照组；
- **第三方代码与基准**：2026 年"状态派"工作（PoS、SKILL.state、Scroll、VISTA……）的代码和要用的基准，以固定版本的 git submodule 形式接入 [`third_party/`](third_party/)；
- **研究文档**：主线、算法设计、相关工作、研究日志、提案和文献地图，在 [`docs/`](docs/)。新会话从 [`CLAUDE.md`](CLAUDE.md) 开始读。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/research_line.md`](docs/research_line.md) | **当前主线**："写入时付代价，还是读取时付代价？"：写入误差与读取误差的分解、五个假设 H1–H5、论文贡献与章节、自适应绑定方法、风险与分阶段计划 |
| [`docs/algorithm_design.md`](docs/algorithm_design.md) | **算法与实验设计**：探针环境规格、11 种条件、错误账本、真值抽取、自适应绑定伪代码、各假设的实验方案、统计与成本协议、冒烟测试协议、待实现清单 |
| [`docs/related_work.md`](docs/related_work.md) | **相关工作全集**：按类别列出全部调研过的论文、代码和 license、关键数字、和主线的区别、基准全表、Harbor 注册表、撞车监控 |
| [`docs/research_log.md`](docs/research_log.md) | **研究日志**：决策时间线、纠正过的错误、已验证事实、代码与基准可用性、冒烟测试结果、云环境限制与搭建命令、待核实清单 |
| [`docs/clm_notes.md`](docs/clm_notes.md) | **CLM 笔记**：harness 机制、配置、ICL、RL、SCR、成本口径、论文结果与公平性问题、我们发现的漏洞 |
| [`docs/idea_bank.md`](docs/idea_bank.md) | **idea 库**：所有考虑过的方向，以及各自的状态（主线 / 并入 / 暂缓 / 放弃）和原因 |
| [`docs/proposal.md`](docs/proposal.md) | 研究提案 v0.1：四个 idea（I1 结构来源与规模、I2 状态错误账本、I3 新旧矛盾 vs 唯一真相来源、I4 动作时刻召回）的详细实验设计。已被主线吸收 |
| [`docs/literature_map.md`](docs/literature_map.md) | 文献地图（简明版）：CLM 和状态派各工作在做什么、区别、证据和趋势 |
| [`third_party/README.md`](third_party/README.md) | **第三方清单**：每个 submodule 的上游地址、固定的 commit、license，以及在本研究中的用途 |
| [`CLAUDE.md`](CLAUDE.md) | 给 Claude Code 新会话的项目记忆：先读什么、现状、约定 |
| [`clm/README.md`](clm/README.md) | CLM 上游的原始 README |

## 目录结构

```
.
├── CLAUDE.md                新会话的项目记忆
├── docs/                    主线、算法设计、相关工作、研究日志、CLM 笔记、idea 库、提案、文献地图
├── clm/                     CLM 原始实现（harness、ICL、RL patch）—— 对照组 A2
├── suffix_cache_reuse/      CLM 的 SGLang 推理服务优化（SCR）
├── third_party/
│   ├── methods/             pos、skill-state-runtime、scroll、vista、rlm、selfcompact、acm、belief-world-models
│   ├── benchmarks/          loca-bench、alfworld、longmemeval、supersede、state-bench（可选）、beam（可选，约 4 GB）
│   └── references/          delayed-relevance（只读的设计参考，没有 license）
├── scripts/
│   └── setup_third_party.sh 按分组初始化 submodule
└── state_study/             我们自己的代码：probes/（I3 仓库探针）、pilots/（冒烟测试记录）、tests/；后续加入各对照组的 ContextProvider、真值抽取、错误账本
```

## 快速开始

```bash
git clone https://github.com/ZHILIANGZHANG/context-language-models.git
cd context-language-models

# 1. 拉取第一批实验需要的依赖：PoS 循环、LOCA、ALFWorld、SKILL.state 运行时、Supersede、LongMemEval
scripts/setup_third_party.sh core
#    其他分组：methods | benchmarks | references | optional | all

# 2. CLM（对照组 A2）的安装方式与上游一致
pip install -e .

# 3. I3 探针的测试（只需标准库 + pytest）
python -m pytest state_study/tests
```

各基准的环境依赖（ALFWorld 的数据、LOCA 的 Node.js 和 MCP、LongMemEval 的数据下载）见 [`third_party/README.md`](third_party/README.md) 和提案 §10。

## 当前进度

- [x] 文献与代码调研，第三方仓库以固定版本接入
- [x] 研究提案 v0.1
- [x] 仓库探针与全部条件（`state_study/probes`，20 个测试通过），覆盖 I1 / I3 / I4
- [x] I3 冒烟测试：Haiku，36 题（[`pilots/2026-10-03_haiku_smoke`](state_study/pilots/2026-10-03_haiku_smoke/README.md)）
- [x] I1 + I4 冒烟测试：Haiku / Sonnet / Opus，36 题（[`pilots/2026-10-03_i1_i4_scale`](state_study/pilots/2026-10-03_i1_i4_scale/README.md)）
- [x] I2：ALFWorld 逐步真值抽取（`state_study/groundtruth/alfworld_facts.py`），6 种任务类型、0 违规；探针上的错误账本
- [x] PoS 循环在云端装好，自带离线检查通过
- [x] 确定主线"写入时付代价，还是读取时付代价？"，四个 idea 全部并入（[`docs/research_line.md`](docs/research_line.md)）
- [x] 全部调研、算法设计、相关工作、决策过程写进文档（`docs/` 和 `CLAUDE.md`）
- [ ] 阶段 1（不需要 API 的部分）：H1 旋钮（事件数 vs token 数）、`control` 场景、新条件、API 调用器、噪声底脚本
- [ ] 阶段 1（需要 API）：噪声底；在 3 个以上规模上做 H1 和 H5；止损判断
- [ ] 阶段 2–3：第 2、3 个探针领域；多轮闭环（H2）；交叉相图（H3）；真实基准预测（H4）；自适应绑定方法和成本

## License

- `clm/` 和 `suffix_cache_reuse/` 沿用上游的 [CC BY-NC 4.0](LICENSE)，另见 [NOTICE](NOTICE)。
- `third_party/` 下的每个 submodule 遵循各自上游的 license，见 [`third_party/README.md`](third_party/README.md)。**VISTA、belief-world-models、delayed-relevance 没有 license**，只作为基线运行或设计参考，不要把它们的代码复制进本仓库。
