# Pilot 2026-10-03（第二轮）：I4 动作时刻召回 + I1 结构来源 × 模型规模

和第一轮一样，这是**方向性的冒烟测试，不是实验结果**：每格 n=3；调用的是 Claude 子 agent，带着 Claude Code 自己的系统提示，温度不可控。

## 设置

- **场景**：与第一轮相同，种子 0/1/2，gap 60，warm-up 200。生成是确定性的，所以 Haiku 在 `explicit/raw` 和 `explicit/state` 上的答案直接沿用第一轮的结果，`responses.jsonl` 里标注了 `from_pilot`。
- **I4 的条件**（定义见 `state_study/probes/render.py`）：
  - `state_stale`：模型手里拿着一份错过了关键事件的状态；
  - `recall_cand`：在 `state_stale` 的基础上，从日志里召回"打算放的那个货架"的全部事件，提炼成一行状态；
  - `recall_alt`：在 `recall_cand` 的基础上，再召回所有编号更低、但状态里标为已占用的货架。
  - 打算放的货架用"陈旧状态下会选的答案"代替。正式方法里，这一步由模型自己先提出候选动作。
- **I1 的条件**：
  - `raw`：没有结构，只给原始历史；
  - `state`：人规定好结构的状态；
  - `self_schema`：模型先自己设计并填写一份状态，再做决定。
  - 用 Haiku、Sonnet、Opus 三档模型构成规模梯度。
- **I2**：打分脚本里加入了错误账本，每个错误归入以下一类（`build_items.classify`）：
  - `stale`：选了"忽略关键事件时会选的答案"；
  - `occupied`：选了一个实际已被占用的货架，属于状态重建错误；
  - `missed_lower`：跳过了编号更低的空货架。

## 结果（`scores.txt`）

**I4：动作时刻召回（Haiku）**

| 场景 | 陈旧状态 | 只召回目标货架 | 召回目标 + 更低编号货架 |
|---|---|---|---|
| delayed：漏了一条隔离通知 | 0/3（第一轮 `state_noq`，全是 stale） | **3/3** | 2/3（1 个 missed_lower） |
| explicit：漏了一条"货架其实是空的"更正 | 0/3（全是 stale） | **0/3（全是 stale）** | **3/3** |

**I1：结构来源 × 模型规模（explicit 场景）**

| 模型 | 原始历史 | 规定结构的状态 | 模型自建状态 |
|---|---|---|---|
| Haiku | 2/3（1 个 occupied） | 3/3 | 3/3 |
| Sonnet | 3/3 | 3/3 | 3/3 |
| Opus | 3/3 | 3/3 | 3/3 |

## 能得出的结论

1. **I4 的核心设计点得到了初步验证，而且暴露出一个不对称性。**
   - 漏掉的信息如果是一条**约束**（比如隔离），只召回打算放的那个货架就能修好：0/3 → 3/3。
   - 漏掉的信息如果是一个**机会**（比如一个更低编号的货架其实空了），只召回目标货架完全没用（0/3），必须同时召回候选之外的选项才能修好（3/3）。
   - 所以动作时刻召回不能只检查"我打算做的动作是否合法"，还要检查"有没有更好的选项"。这一点写进了提案 §7。
   - 召回内容变多后，delayed 场景出现了 1 个新错误。召回的范围和召回带来的噪声之间需要权衡。
2. **I1 在这个探针上测不出规模效应。**
   - Sonnet 和 Opus 在三种条件下都是满分，只有 Haiku 在原始历史上错了 1 次。
   - 这说明约 1.1 万 token 的单题探针对强模型太简单。
   - I1 必须换到多步回合（PoS 循环 + ALFWorld/LOCA，需要 API），或者把探针加长、加难之后才有意义。**目前不能对 I1 下任何结论。**
3. **I2 的错误账本可以在探针上自动归类**，例如区分 `stale` 和 `occupied`。ALFWorld 上的逐步真值抽取另外验证过，见 `state_study/groundtruth/alfworld_facts.py`。

## 复现

```bash
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 \
  --plan delayed:recall_cand,recall_alt explicit:state_stale,recall_cand,recall_alt,raw,state,self_schema
python -m state_study.probes.build_items score --out OUT   # responses.jsonl 每行可带 "model" 字段
```
