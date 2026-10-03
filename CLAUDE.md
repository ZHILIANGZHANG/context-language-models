# CLAUDE.md

这是一个研究工作区，课题是长程 agent 的上下文表示：显式状态 vs 历史。仓库 fork 自 facebookresearch/context-language-models（CLM），现在 CLM 只是其中一个对照组。用户用中文交流，文档也用中文写。

## 先读什么

新会话开始时按这个顺序读，就能接上全部进度：

1. `docs/research_line.md`：**当前主线**。"写入时付代价，还是读取时付代价？"：误差分解、H1–H5、自适应绑定方法、分阶段计划
2. `docs/research_log.md`：时间线、纠正过的错误、已验证事实、冒烟测试结果、环境限制和搭建命令、待核实清单、下一步
3. `docs/algorithm_design.md`：探针、11 种条件、错误账本、真值、自适应绑定伪代码、各假设的实验方案、统计和成本协议、待实现清单
4. 需要时再查：
   - `docs/related_work.md`：相关工作全集，带核实标记；
   - `docs/clm_notes.md`：CLM 代码和论文细节；
   - `docs/idea_bank.md`：所有考虑过的 idea 和放弃原因；
   - `docs/proposal.md`：I1–I4 的原始提案，已被主线吸收；
   - `docs/literature_map.md`：简明文献地图。

## 现状（2026-10-03）

- **主线已定**。用户确认的四个 idea（I1 结构来源 × 规模、I2 错误账本、I3 新旧矛盾、I4 动作时刻召回）全部并入主线。沿用 I1–I4 这套编号，不要改名。
- **已实现**：
  - `state_study/probes/`：仓库探针、11 种条件、出题和打分、错误账本；
  - `state_study/groundtruth/alfworld_facts.py`：ALFWorld 逐步真值；
  - `state_study/tests/`：20 个测试；
  - 两轮冒烟测试，在 `state_study/pilots/`。
- **卡点**：没有模型 API 密钥；多数模型服务域名和 arxiv、HF 被网络策略拦截。所以正式实验都没跑。
- **下一步**：阶段 1。先做不需要 API 的部分：H1 旋钮、`control` 场景、新条件、API 调用器、噪声底脚本。见 research_log §8。

## 仓库地图

```
clm/, suffix_cache_reuse/   CLM 上游代码（CC BY-NC 4.0），对照组 A2；不要改动
third_party/                15 个固定版本的 submodule（方法、基准、参考），见 third_party/README.md
scripts/setup_third_party.sh  按分组初始化 submodule：core | methods | benchmarks | references | optional | all
state_study/                我们自己的代码
  probes/                   warehouse.py（场景生成）、render.py（条件）、build_items.py（出题、打分、账本）
  groundtruth/              alfworld_facts.py
  tests/                    test_warehouse_probe.py
  pilots/                   冒烟测试记录（README、key、responses、blind_map、scores）
docs/                       全部研究文档
```

## 常用命令

```bash
python3 -m pytest -q state_study/tests                         # 探针测试，只需标准库 + pytest
scripts/setup_third_party.sh core                              # 拉第一批依赖
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 --plan explicit:raw,state
python -m state_study.probes.build_items score --out OUT       # 读 OUT/responses.jsonl
```

PoS 和 ALFWorld 的虚拟环境搭建、离线检查、真值重放命令见 `docs/research_log.md` §6.3。虚拟环境放在 scratchpad 里，不要放进仓库。

## 约定

- **分支**：只在 `claude/tender-cerf-q8x3of` 上开发和推送（`git push -u origin claude/tender-cerf-q8x3of`）。网络失败时按 2/4/8/16 秒退避，最多重试 4 次。用户没要求就不开 PR。
- **提交信息**：结尾加上会话要求的 Co-Authored-By 和 Claude-Session 两行；不要在提交、PR、代码里写模型标识。
- **没有 license 的仓库**：VISTA、belief-world-models、delayed-relevance 没有 license，只能运行或参考思路，**不能复制代码**进本仓库。仓库探针是按 SKILL.state 论文 §4.1 的描述重新实现的。
- **凭证**：不要把本会话自己的登录凭证拿去调模型 API；不要让用户把密钥贴到聊天里。密钥应该由用户在云环境设置里配置。
- **冒烟测试**：没有 API 时，可以用 Claude 子 agent 中转做管线冒烟测试。规则是题目文件盲化、答案单独存放、每题一个子 agent。结果只说明方向，**不能当实验数据**。
- **核实标记**：论文数字要标来源：[R] 读过仓库、[N] 全文读书笔记、[S] 只看过搜索摘要。[S] 的数字引用前必须核对原文。
- **提交前**：清理 `__pycache__`、`.pytest_cache`；不要提交 submodule 里下载的数据，例如 `third_party/methods/pos/benchmarks/ALFWorld/`。
- **新领域的探针**：必须配"规则读者"测试，证明只看文本就能答对，否则模型答错不能归因于模型。
- **撞车检查**：这个方向每周都有新论文。动笔前一周重新检索，关键词见 `docs/related_work.md` §10。
