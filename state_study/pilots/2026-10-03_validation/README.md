# Pilot 2026-10-03（第三轮）：验证"状态是用干扰换持久"

和前两轮一样，子 agent 部分是**方向性的冒烟测试，不是实验结果**：每题一个 Haiku 子 agent，带着 Claude Code 自己的系统提示，温度不可控；题目盲化，答案在打分时重新生成。标 [R] 的部分是对 delayed-relevance 真实 API 轨迹的重新统计，不依赖子 agent。

结论和解读见 [docs/validation_2026-10-03.md](../../../docs/validation_2026-10-03.md)；代码和复现命令见 [state_study/validation](../../validation/)。

## 目录

| 目录 | 内容 | 打分 |
|---|---|---|
| `trace_analysis/` | [R] `dr_tables.txt`（L1、L2 原轨迹的表格和自由字段分解）、`dr_persistence.txt`（信念偏差持续时间、错误成串）、`dr_adoption.txt`（被拒后的采纳率）、`dr_coherence.txt`（L2 通知与历史是否一致） | — |
| `dr_l2/` | L2 反事实重放：`blind_A.json`、`outA/`（第 10 步作答）、`blind_B.json`、`outB/`（第 11 步作答） | `scores.txt` |
| `dr_l1/` | L1 重放：`blind.json` + `outA/`（BASE/CHECK/JIT），`blind_D.json` + `outW/` + `outD/`（PIN/FACT 两次调用） | `scores.txt` |
| `ig/` | 整合门控探针：`blind_map.json` + `outA/`（第一步），`blind_map_B.json` + `outB/`（第二步） | `scores.txt` |
| `vdepth/`、`retro/`、`echo/` | 版本深度、事后更正、更正后自我复述：`blind_map.json` + `responses/haiku.jsonl` | `scores.txt` |

重放用的 prompt 没有存：它们含有 delayed-relevance 的轨迹内容，而该项目没有 license。用 `state_study/validation` 的脚本可以按原样重新生成（已验证与当时逐字一致）。自写探针的题目也可以从种子重新生成。

## 结果摘要

| 实验 | 条件 | 结果 |
|---|---|---|
| L2 原轨迹 [R] | 第一个依赖于更正的决策，ReAct vs SKILL.state | Haiku 1/24 vs 24/24；Sonnet 9/24 vs 22/24；Gemini 0/24 vs 24/24 |
| L2 一致性 [R] | 通知是否与历史一致 | 用过的 6 个种子全部矛盾；种子 0–39 中 36 个矛盾 |
| L2 重放 | ORIG / COH（只改 `corrects_step`）/ INT | 4/12 / **10/12** / 8/12 |
| L1 原轨迹 [R] | 历史 / 历史 + 提醒 / 状态无字段 / 状态 + 专设字段 / 状态 + 自由字段（Haiku） | 0/24 / 23/24 / 0/24 / 24/24 / 10/24 |
| L1 自由字段分解 [R] | 到达时写下 vs 没写下 → 决策正确 | Haiku 10/11 vs 0/13；Sonnet 8/9 vs 2/15 |
| L1 重放 | BASE / CHECK / FACT / JIT / PIN | 0/12 / 0/12 / 0/12 / 11/12 / **12/12**（PIN 写入率 12/12） |
| 整合门控 | 历史三种说明书 / 状态 | 24/24 / 7/8 |
| 版本深度 | 0–12 次翻转 | 18/20，看不出随深度变化 |
| 事后更正、自我复述 | 各格 1–2 题 | 全对，样本太小 |
| 状态代价 [R] | Haiku T=200 SKILL.state | 信念偏差中位持续 24 步，46 个里 15 个没修好；被拒后修好 8/78 |
| 闭环沙盒（`closed_loop/`） | Haiku，BASE，种子 0，自写环境 `wh_env.py`，只验证流程 | 50 步对 49 步，唯一的错是第 48 步存到隔离货架；n=1，不是结果 |

## 已知的局限

- 未完成：整合门控探针中性说明书的 40 题只跑了第一步的 10 题（`ig/outA` 里有这 10 题的回答，没有第二步）。
- L2 重放的代理校准不好：原 API 轨迹 ORIG 是 1/24，子 agent 是 4/12。
- 每个种子的 4 个回合共享同一段环境事件，样本不完全独立；Fisher 检验的 p 值偏乐观。
- 自由字段的"写下"是启发式正则判断（`dr_tables.py`），需要人工复核。
- 闭环环境的正式试验还没跑：重放题在 sonnet / opus 子 agent 上被安全分类器拦截（`reasoning_extraction`），闭环的规模和模型等用户确认（见 `docs/research_log.md` §5.5）。
