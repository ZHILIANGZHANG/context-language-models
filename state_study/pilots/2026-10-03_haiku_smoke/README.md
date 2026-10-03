# Pilot 2026-10-03：仓库探针冒烟测试（Haiku 子 agent）

**目的**：在云端没有模型 API 的情况下，验证 I3 的实验管线能否端到端跑通，并看一眼方向性信号。
**这不是实验结果**：每格只有 n=3，模型调用也不干净（原因见下文），不能当作证据引用。

## 设置

- **环境**：`state_study/probes`，仓库探针，按 SKILL.state 论文 §4.1 的描述重新实现。参数：warm-up 200 个事件，gap 60，种子 0/1/2。每个场景在第 270–290 步结束于一个决策点。
- **设计**：先由 oracle 生成前缀（之前的每一步动作都正确），再只问决策点这一步，形式是单题问答。三种场景：
  - `explicit`：审计更正明确说"某次入库没完成，货架是空的"，作废之前那条入库记录；
  - `implicit`：库存扫描显示货架为 EMPTY，但不提之前那条记录；
  - `delayed`：60 个事件之前发的隔离通知，到决策点才派上用场。
- **对照条件**：定义见 `state_study/probes/render.py`。
  - `raw`：原始历史日志
  - `tomb_tail`：原始日志 + 末尾追加一份作废清单（只追加，不破坏缓存）
  - `tomb_inline`：在被作废的旧条目原地标注 VOID
  - `delete`：删掉被作废的旧条目
  - `state`：只给当前状态的 JSON，不给历史
  - `state_noq`：同 `state`，但状态里没有"隔离"这个字段
  - `state_log`：原始日志 + 末尾附上状态 JSON
- **提示长度**：带历史的条件约 1.1 万 token，只给状态的约 900 token。按字符数 ÷ 4 估算，因为 tiktoken 的词表下载被网络策略拦截。
- **模型**：每题启动一个 Claude Haiku 子 agent，用 Read 工具读入盲化后的提示文件（文件名不暴露条件），只返回 JSON。答案单独存放，子 agent 看不到。

## 结果（`scores.txt`）

| 场景 / 条件 | 正确 | 说明 |
|---|---|---|
| delayed / raw | 2/3 | 唯一的错误是选了一个已被占用的货架，不是忽略了隔离 |
| delayed / state | 3/3 | |
| delayed / state_noq | **0/3** | 3 次全部选了被隔离的货架：schema 里没有这个字段，就一定会错 |
| explicit / raw | 2/3 | 错误是选了 S1，状态重建出错 |
| explicit / tomb_inline | 2/3 | 错误同样是 S1，状态重建出错 |
| explicit / tomb_tail、delete、state、state_log | 各 3/3 | |
| implicit / raw | 2/3 | 1 次选了作废前的旧答案 |
| implicit / tomb_tail、state | 各 3/3 | |

## 能得出的结论

1. **管线跑通了**：从场景生成、条件渲染、盲化、调用模型、解析回答到打分，36/36 全部可解析。
2. **"状态里没有对应字段"会稳定导致失败**：`state_noq` 是 0/3，而且每次都选了"忽略关键事件时会选的答案"。这和 delayed-relevance 复现项目的结论方向一致。
3. **原始历史出错不只是因为"用了旧值"**：`raw` 一共错了 3 次，只有 1 次是选了作废前的旧答案，另外 2 次是从 270 多步日志里重建状态时出错（以为 S1 是空的）。这提示历史失败至少有两种机制：
   - **干扰**：新旧信息同时存在，模型用了旧的；
   - **重建负担**：模型要从很长的日志里自己推出当前状态，推错了。
   
   这是 I3 正式实验要拆开的两个变量。
4. **还不能说明的**：n=3 统计力为零；单题加 oracle 前缀比完整多轮回合容易，存在天花板（`raw` 仍有 67% 正确）；子 agent 带着 Claude Code 自己的系统提示，不是干净的 API 调用，温度不可控，也拿不到 token 用量。

## 复现

```bash
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 \
  --plan explicit:raw,tomb_tail,tomb_inline,delete,state,state_log implicit:raw,tomb_tail,state delayed:raw,state,state_noq
# 盲化 + 收集模型回答到 OUT/responses.jsonl 后：
python -m state_study.probes.build_items score --out OUT
```

`blind_map.json` 记录盲化后的文件名和实际条件的对应关系。生成是确定性的，同样的参数会生成同样的提示文件。
