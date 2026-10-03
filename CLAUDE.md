# CLAUDE.md

这是一个研究工作区，课题是长程 agent 的上下文表示：显式状态 vs 历史。仓库 fork 自 facebookresearch/context-language-models（CLM），现在 CLM 只是其中一个对照组。用户用中文交流，文档也用中文写。

## 先读什么

新会话开始时按这个顺序读，就能接上全部进度：

1. `docs/validation_2026-10-03.md`：**最新**。第三轮验证：上一轮的"干扰换持久"大半未通过，delayed-relevance 的事后更正探针自相矛盾，修正论点为"决定成败的是到达时的写入，不是表示方式"。主线可能要据此改写，先读这份
2. `docs/research_line.md`：**当前主线**。"写入时付代价，还是读取时付代价？"：误差分解、H1–H5、自适应绑定方法、分阶段计划
3. `docs/research_log.md`：时间线、纠正过的错误、已验证事实、冒烟测试结果、环境限制和搭建命令、待核实清单、下一步
4. `docs/algorithm_design.md`：探针、11 种条件、错误账本、真值、自适应绑定伪代码、各假设的实验方案、统计和成本协议、待实现清单
5. 需要时再查：
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
  - `state_study/validation/`：第三轮验证的自写探针，以及 delayed-relevance 轨迹的重放和分析（运行时导入，不复制）；
  - `state_study/validation/wh_env.py`：闭环仓库环境，子 agent 自己逐步玩完一个回合；
  - `state_study/tests/`：99 个测试；
  - 三轮冒烟测试，在 `state_study/pilots/`。
- **卡点**：测试不再卡在 API 密钥上（默认用子 agent，见"约定"）；仍然被拦的是 arxiv、HF 等网络，读不了论文原文。
- **第三轮验证（2026-10-03）**：结论见 `docs/validation_2026-10-03.md`。主线是否改写成"到达时绑定"，待用户决定。
- **下一步**：见 `docs/validation_2026-10-03.md` §5：读最接近的五篇原文 → 在更大的模型上复核 → 闭环实验 → 第二个领域。重放题在 sonnet / opus 子 agent 上被安全分类器拦截，所以复核改用闭环环境 `wh_env.py`（见 `docs/research_log.md` §5.5）。闭环试验的规模和模型**等用户确认**后再跑。

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
  validation/               第三轮验证：自写探针，以及 delayed-relevance 轨迹的重放和分析（只运行、不复制）；闭环环境 wh_env.py、wh_pilot.py
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
- **云端测试默认用子 agent**（用户 2026-10-03 指示）：不等 API 密钥，直接用 Agent 工具调 Claude 子 agent 测，可选 haiku / sonnet / opus / fable 四档。
  - **先说明再跑**：每次启动子 agent 之前，先告诉用户跑什么、跑几个、预计用量（每个子 agent 都要重读 Claude Code 的系统提示，一个 50 步的闭环回合约 280 万 token 缓存读取），等用户确认再跑（用户 2026-10-03 要求）。
  - **题目类测试**：题目文件盲化、答案单独存放、每题一个子 agent，子 agent 只许 Read 题目和 Write 一次答案。
  - **闭环测试**（`state_study/validation/wh_env.py`）：子 agent 只许跑 start / act 两条命令；事后用 `wh_pilot.py audit` 读子 agent 记录，核对它没跑别的命令。
  - **安全分类器**：题目里带着别的模型写的推理文字（例如 delayed-relevance 轨迹里的推理），sonnet / opus 子 agent 会被拦截（`reasoning_extraction`）。不要改写题目去绕开它，改用闭环。
  - **限制**：子 agent 带着 Claude Code 的系统提示，温度不可控。子 agent 记录（`~/.claude/projects/<项目>/<会话>/subagents/agent-*.jsonl`）里能读到输入和缓存 token 数，输出 token 数不可靠，thinking 内容是空的。结果要标成"子 agent 测量"，并写明这些限制。投稿前是否再用 API 复核，由用户决定。
- **核实标记**：论文数字要标来源：[R] 读过仓库、[N] 全文读书笔记、[S] 只看过搜索摘要。[S] 的数字引用前必须核对原文。
- **提交前**：清理 `__pycache__`、`.pytest_cache`；不要提交 submodule 里下载的数据，例如 `third_party/methods/pos/benchmarks/ALFWorld/`。
- **新领域的探针**：必须配"规则读者"测试，证明只看文本就能答对，否则模型答错不能归因于模型。
- **撞车检查**：这个方向每周都有新论文。动笔前一周重新检索，关键词见 `docs/related_work.md` §10。
