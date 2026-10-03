# Where Agents Keep the Truth

**长程 Agent 上下文表示研究工作区：状态 vs 历史，结构来源、错误归因与绑定时机。**

本仓库最初 fork 自 [facebookresearch/context-language-models](https://github.com/facebookresearch/context-language-models)（CLM，arXiv 2609.37725），现已改造成一个完整的研究工作区。它由三部分组成：

- **CLM 原始实现**：保留在 [`clm/`](clm/) 和 [`suffix_cache_reuse/`](suffix_cache_reuse/)，在研究中作为"自由形式上下文编辑"这一对照组；
- **第三方代码与基准**：2026 年"状态派"工作（PoS、SKILL.state、Scroll、VISTA……）的代码和要用的基准，以固定版本的 git submodule 形式接入 [`third_party/`](third_party/)；
- **研究文档**：提案和文献地图，在 [`docs/`](docs/)。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/proposal.md`](docs/proposal.md) | **研究提案**：四个 idea（I1 结构来源与规模、I2 状态错误账本、I3 新旧矛盾 vs 唯一真相来源、I4 动作时刻召回）的出发点、假设、实验设计、指标、时间线、预算和风险 |
| [`docs/literature_map.md`](docs/literature_map.md) | **文献地图**：CLM 和状态派各工作在做什么、彼此有什么区别、代码和基准是否可用、有哪些证据、趋势是什么 |
| [`third_party/README.md`](third_party/README.md) | **第三方清单**：每个 submodule 的上游地址、固定的 commit、license，以及在本研究中的用途 |
| [`clm/README.md`](clm/README.md) | CLM 上游的原始 README |

## 目录结构

```
.
├── docs/                    提案与文献地图
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
- [x] I3 探针环境与渲染条件（`state_study/probes`，19 个测试通过）
- [x] I3 冒烟测试：Haiku 子 agent，36 题，管线端到端跑通（[`state_study/pilots/2026-10-03_haiku_smoke`](state_study/pilots/2026-10-03_haiku_smoke/README.md)）
- [ ] I3 正式实验：需要模型 API 密钥（见提案 §6.7）
- [ ] W1：在选定模型上复现 PoS 在 ALFWorld 和 LOCA 8K 上的结果（检查点 G1）
- [ ] W2：实现各对照组和真值抽取器

## License

- `clm/` 和 `suffix_cache_reuse/` 沿用上游的 [CC BY-NC 4.0](LICENSE)，另见 [NOTICE](NOTICE)。
- `third_party/` 下的每个 submodule 遵循各自上游的 license，见 [`third_party/README.md`](third_party/README.md)。**VISTA、belief-world-models、delayed-relevance 没有 license**，只作为基线运行或设计参考，不要把它们的代码复制进本仓库。
