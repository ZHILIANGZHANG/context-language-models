# 算法与实验设计

本文档是主线 [research_line.md](research_line.md) 的技术底稿：探针环境怎么构造、每个实验条件是什么、错误账本怎么归类、真值怎么取、方法（自适应绑定）怎么实现、每个假设怎么测、成本和统计怎么算。

- **已实现**的部分写到代码级别，并标出文件路径；
- **计划中**的部分写到可以直接动手实现的程度，并标为【待实现】。

最后更新：2026-10-03。

---

## 0. 总览：主线、模块与代码的对应关系

| 主线里的概念 | 实现它的模块 | 代码位置 | 状态 |
|---|---|---|---|
| 可控探针（领域 1：仓库） | 场景生成器 | `state_study/probes/warehouse.py` | 已实现 |
| 实验条件（干预） | 渲染器，11 种条件 | `state_study/probes/render.py` | 已实现 |
| 出题、盲化、打分 | 出题与打分 | `state_study/probes/build_items.py` | 已实现 |
| 探针上的错误账本 | `classify()` | `state_study/probes/build_items.py` | 已实现 |
| 探针的正确性保证 | 20 个测试，含"规则读者" | `state_study/tests/test_warehouse_probe.py` | 已实现 |
| ALFWorld 逐步真值 | facts 抽取与专家重放 | `state_study/groundtruth/alfworld_facts.py` | 已实现 |
| H1 旋钮：事件数 vs token 数 | 生成器参数 | `warehouse.py` | 【待实现】§1.5 |
| 对照场景 `control`（只测重建） | 生成器新 kind | `warehouse.py` | 【待实现】§1.6 |
| 领域 2：代码仓库状态；领域 3：个人助理偏好 | 新生成器 | `state_study/probes/` | 【待实现】§1.7 |
| 多轮闭环模式（H2） | 闭环驱动器 | `state_study/closed_loop/` | 【待实现】§7 |
| 对照组 A0–A6 | PoS 的 `ContextProvider` 子类 | `state_study/providers/` | 【待实现】§6 |
| 自适应绑定 | 一个 `ContextProvider` + 召回器 | `state_study/providers/adaptive_binding.py` | 【待实现】§5 |
| 带缓存的成本统计 | 成本模块 | `state_study/cost/` | 【待实现】§10 |
| LOCA 逐步真值、LongMemEval-KU 适配、ALFWorld 扰动 | 真值与适配器 | `state_study/groundtruth/`、`state_study/adapters/` | 【待实现】§4、§12 |

---

## 1. 探针环境（领域 1：仓库）

### 1.1 规则与"决策步"

agent 只遵守一条规则（`warehouse.RULE`）：

> 物品到货时，放到**当前为空、且没有被隔离**的、编号最小的货架上。隔离通知在收到解除通知前一直有效。审计更正和库存扫描覆盖之前的记录。

这样，任何一步的正确动作都是真实世界状态的一个纯函数，答案可以精确算出。

每个场景（`Scenario`）是一段由 **oracle agent**（之前的每一步都做对）生成的事件前缀，以一个**决策步**结束。决策步是一次到货，它满足：

- 正确答案 `answer` ≠ "忽略了某条关键事件时会选的答案" `deaf_answer`。

也就是说，每道题都**必然依赖那条关键事件**（生成器里有 `assert answer != deaf`）。

`Scenario` 的字段：`kind, seed, gap, n_shelves, events, decision_item, answer, deaf_answer, key_step, shelves`（决策时刻的真实货架内容）、`quarantined`。

### 1.2 事件类型

| 类型 | 文本样式 | 对世界状态的影响 | 备注 |
|---|---|---|---|
| `arrival` | `Inbound: item I-xxx arrived at dock …（supplier, carrier, weight, pallets, po, temp_class）` + `ACTION: STORE I-xxx -> Sk` + `RESULT: OK` | 占用最低空闲货架 | oracle 处理，带动作和结果 |
| `dispatch` | `Outbound: order SO-… picked item I-xxx from Sk …; Sk is now EMPTY.` | 清空货架 | 只作用于编号高于 `dispatch_above` 的货架 |
| `telemetry` | 温度、湿度、叉车电量、队列、传送带速度等 | 无 | 纯噪声，用来加长上下文 |
| `quarantine` | `Facilities notice FN-…: shelf Sk QUARANTINED (…do_not_store=true…)` | 加入隔离集合 | delayed 场景的关键事件 |
| `correction` | `Audit correction AC-…: the storage recorded at step t (I-xxx -> Sk) never completed … Sk is EMPTY.` | 清空货架；把原入库事件标为 `superseded_by` | explicit 场景的关键事件，明确引用旧条目 |
| `scan` | `Inventory scan (zone sweep …): S3: OCCUPIED; S5: EMPTY; …` | 清空目标货架；原入库事件标为被取代 | implicit 场景的关键事件，**不引用**旧条目，另外随机列出 4 个其他货架 |
| `decision` | 最后一次到货，没有动作 | — | 要回答的问题 |

随机性全部来自 `random.Random(f"{kind}-{seed}")`，所以同样的 `(kind, seed, gap, warmup)` 一定生成同样的场景。

### 1.3 三种场景的生成步骤（`generate()`）

公共的预热阶段：

1. `filler(warmup, arrivals=True, dispatch_above=0, p_arrival=0.35, p_dispatch=0.12)`：到货、出货、遥测混合；
2. 持续补货，直到 S1–S8 全部被占用（期间以 0.5 的概率插入遥测）；
3. 随机选目标货架 `target ∈ [3, 6]`。

然后按场景分别构造：

**explicit（明确更正）**
1. `dispatch(target)`：目标货架被清空，成为编号最小的空货架；
2. `arrival()`：下一件货放上目标货架。这条入库记录之后会被作废；
3. `filler(10, arrivals=True, dispatch_above=target)`；
4. `correction(stored)`：审计更正，说这次入库没完成，目标货架是空的；
5. `filler(gap, arrivals=False, dispatch_above=target)`：在 gap 期间不到货，保证关键事件之后的第一次到货就是决策步；
6. `answer = lowest_free()`；`deaf = lowest_free(treat_occupied=(target,))`。

**implicit（隐式扫描）**：与 explicit 相同，只是第 4 步换成 `scan(stored)`。扫描结果只说"Sk: EMPTY"，不提那条旧记录。

**delayed（延迟相关）**
1. `quarantine(target)`：目标货架此时仍被占用，所以这条通知暂时无关；
2. `filler(gap, arrivals=True, dispatch_above=target, keep=(target,))`：期间有到货，但不动目标货架；
3. `dispatch(target)`：目标货架被清空，成为编号最小的空货架，但它被隔离了；
4. `answer = lowest_free()`（跳过隔离）；`deaf = lowest_free(ignore_quarantine=True)`。

货架数默认 `n_shelves = 20 + (warmup + gap) // 2`。原因是预热期间到货快于出货，货架太少会触发 `RuntimeError("warehouse full")`。

### 1.4 已有的旋钮

| 参数 | 含义 | 冒烟测试用的值 |
|---|---|---|
| `kind` | explicit / implicit / delayed | 全部 |
| `seed` | 种子 | 0, 1, 2 |
| `gap` | 关键事件与决策步之间的事件数 | 60 |
| `warmup` | 预热事件数，决定日志总长 | 200 |
| `n_shelves` | 货架数 | 默认公式 |
| `filler` 的 `p_arrival`、`p_dispatch` | 背景事件里状态变化事件的比例；其余都是遥测 | 预热 0.35 / 0.12；之后 0.3 / 0.15 |

warm-up 200、gap 60 时，决策步大约在第 270–290 步；带历史的提示约 1.1 万 token（按字符数 ÷ 4 估算），只给状态的约 900 token。

### 1.5 H1 旋钮：事件数 vs token 数【待实现】

**目标**：把"上下文有多长"和"状态变了多少次"这两个因素分开调。

**设计**：在 `generate()` 里加两个独立参数，替换目前只靠 `warmup` 一起控制的做法：

- `n_state_events`：背景里状态变化事件（到货 + 出货）的个数；
- `n_telemetry`：背景里遥测事件的个数。

用以下两组扫描检验 H1：

| 扫描 | 固定什么 | 变什么 | H1 的预测 |
|---|---|---|---|
| A：token 扫描 | `n_state_events` | `n_telemetry` ∈ {0, 100, 300, 1000} | `raw` 准确率基本不变或只轻微下降 |
| B：事件扫描 | 总 token 数（每加一个状态事件就减掉等长的遥测） | `n_state_events` ∈ {20, 60, 150, 400} | `raw` 准确率单调下降 |

**实现要点**：

1. 预热改成先用 `n_state_events` 生成状态事件序列，再按 `n_telemetry` 把遥测随机插进去。插入位置由独立的随机数流决定，保证两组扫描在同一种子下，状态事件序列完全相同。
2. 扫描 B 要"等 token"：到货、出货、遥测三类文本的平均长度先测好，用遥测条数补齐 token 差额，误差控制在 ±3% 以内。每道题的 `est_tokens` 已经写进 `key.jsonl`，可以直接检查。
3. 状态事件必须真的"相关"：只统计影响决策时最低空货架判断的那些事件，也就是编号不高于 `answer` 的货架上的到货和出货。生成时把这个数记进 `key.jsonl`，作为 `n_relevant_events`。
4. 加一个"多次作废"参数 `n_voided ∈ {1, 3, 10}`：同一场景里多次"入库 → 被更正"，用来测干扰随冲突副本数的变化。

### 1.6 对照场景 `control`：只测重建负担【待实现】

**目的**：把重建误差（M2）从干扰（M1）里单独分离出来。

**构造**：没有任何更正、扫描或隔离。答案完全取决于大量出入库的累积结果。具体做法：

- 在决策步之前安排一批出货，让最低空货架落在一个"最近一次被改动是很久以前"的货架上；
- `deaf_answer` 定义为"把最后 m 条状态事件之前的状态当成当前状态"时会选的货架（m 取 20），用来识别"只看了最近一段"这种错误。

**可应用的条件**：`raw`、`state`、`state_log`、`self_schema`、`recall_*`。`tomb_*`、`delete` 不适用，因为没有被作废的条目。

### 1.7 领域 2 和领域 3【待实现】

目的：回应"只是仓库任务"的质疑。每个领域都要满足四条：

1. 有一条确定性规则，正确动作是真实状态的纯函数；
2. 有 explicit、implicit、delayed、control 四类场景；
3. 支持 §1.5 的两个旋钮；
4. 有自己的"规则读者"测试（§1.8）。

**领域 2：代码仓库状态**
- 状态：函数签名、配置项、文件存在性、CI 状态。
- 事件：提交（改名、改签名）、回滚、配置覆盖、CI 结果、无关的日志输出（相当于遥测）。
- 决策：调用某个函数时该用哪个名字和参数；应该读哪个配置值。
- explicit：一条 "revert commit abc" 明确撤销之前的改名；implicit：一条 `ls` 或 `grep` 输出里没有旧文件名；delayed：一条"冻结某目录、不许改"的通知，很久以后才用上。
- 注意：delayed-relevance 复现里的 Repo 环境被它自己撤回了，原因是"合并 PR 之后正好两步就失效"，k 固定为 2，测不出延迟相关。我们的设计必须让 gap 可调。

**领域 3：个人助理偏好**
- 状态：用户偏好（饮食、座位、时间）、约定（会议、预订）、联系人信息。
- 事件：用户说的话、邮件、日历通知、无关闲聊。
- 决策：替用户订餐、订座、排会议。
- explicit："我不再吃素了"；implicit：用户点了一份牛排；delayed："下周三以后别给我排早上 9 点前的会"。
- 这个领域正好对上 LongMemEval 的知识更新类问题，方便在 H4 里做从探针到真实基准的预测。

### 1.8 探针的正确性保证（20 个测试）

`state_study/tests/test_warehouse_probe.py` 覆盖以下几类：

| 测试 | 保证什么 |
|---|---|
| 确定性 | 同样参数生成逐字节相同的场景 |
| 依赖关键事件 | 每个场景 `answer != deaf_answer` |
| 独立重新模拟（`_replay`） | 只用事件文本重新模拟一遍，算出的货架状态与 `sc.shelves` 一致 |
| 规则读者（`_solve_from_text`） | 一个按规则逐行解析提示文本的程序，在每种条件下都能答对，处理陈旧状态和召回行。**按设计答不对的只有**：`state_noq`；`state_stale`；explicit/implicit 下的 `recall_cand` |
| 条件形状 | 每种条件的提示包含且只包含应有的部分，例如 `state_noq` 里没有 `"quarantined"` 这个 JSON 键 |
| `test_recall_contents` | 召回行的内容和真值一致 |

**规则读者是关键的有效性论证**：它证明模型答错是模型自己的问题，而不是题目信息不足。新领域必须配同样的测试。

运行：`python -m pytest state_study/tests`（只需要标准库和 pytest）。

---

## 2. 实验条件（干预）

### 2.1 已实现的 11 种条件（`render.py`）

所有条件渲染**同一个场景前缀**，只改变表示方式。提示的统一格式：

```
<RULE>

You are the warehouse agent. Your working context follows.

=== CONTEXT ===
<body：随条件变化>
=== END CONTEXT ===

[step N] <决策步到货文本>
Which shelf do you store item I-xxx on?
Reply with JSON only, exactly of the form {"shelf": "S<number>"}.
```

| 条件 | body 的内容 | 前缀缓存安全？ | 适用场景 | 对应的误差项（见 research_line §3） |
|---|---|---|---|---|
| `raw` | 完整时间顺序日志 `HISTORY LOG:` | 是（只追加） | 全部 | 基线：同时含重建和干扰 |
| `tomb_tail` | 日志 + 末尾追加 `SUPERSEDED ENTRIES (do not rely on them):` 作废清单 | 是 | explicit、implicit | 去掉干扰（只追加的方式） |
| `tomb_inline` | 日志里被作废的条目原地加 `[VOID: superseded by step t]` | 否（改了过去） | explicit、implicit | 去掉干扰（原地标注） |
| `delete` | 删掉被作废的条目 | 否 | explicit、implicit | 去掉干扰（删除） |
| `state` | 当前货架状态 JSON（含 `quarantined` 列表），不给历史 | 不适用（整段重写） | 全部 | 去掉重建（以及干扰） |
| `state_noq` | 同 `state`，但没有 `quarantined` 字段 | 不适用 | delayed | 制造覆盖缺失 |
| `state_log` | 日志 + 末尾附当前状态 JSON | 否（尾部重写） | 全部 | 状态与历史并存 |
| `state_stale` | 一个漏掉了关键事件的 agent 手里的状态 | 不适用 | 全部 | I4 的起点：带写入误差的状态 |
| `recall_cand` | `state_stale` + 从日志召回"打算放的货架"的全部事件，提炼成一行 | 不适用 | 全部 | 动作时刻召回，只召回候选 |
| `recall_alt` | `recall_cand` + 召回所有编号更低、但陈旧状态标为已占用的货架 | 不适用 | 全部 | 动作时刻召回，候选 + 被挤掉的选项 |
| `self_schema` | 原始日志；要求模型先在 `<state>…</state>` 里自己设计并填写状态，再决定，最后一行输出 JSON | 是 | 全部 | I1 的代理：结构由模型自己定 |

几个实现细节：

- **陈旧状态**（`_stale_shelves`）：explicit/implicit 下，把被作废的那条入库当成仍然有效；delayed 下，去掉隔离列表。
- **候选**：在探针里，候选直接取 `deaf_answer`，也就是陈旧状态下会选的货架。正式方法里由模型自己先提出（§5）。
- **召回提炼**（`recall()`）：按时间扫描该货架的所有事件，输出一行 `- Sk: status=EMPTY|OCCUPIED (原因); quarantined=YES (…)|NO`。被作废的入库写成 "recorded"，有效的写成 "stored"。
- **提示长度**：带历史的约 1.1 万 token，只给状态的约 900 token（warm-up 200、gap 60）。

### 2.2 计划新增的条件【待实现】

| 条件 | 内容 | 用途 |
|---|---|---|
| `state_head` | 状态 JSON 放在日志**前面** | 和 `state_log` 对比，量化状态位置的影响；成本上对应 delayed-relevance 测出的"同样内容顺序不同，成本差 5.7 倍" |
| `recall_raw` | 召回但不提炼，把相关事件原文贴进来 | 复现"原文照搬提醒 67%、提炼后 100%"的对比 |
| `recall_all` | 召回所有货架 | 召回范围的上限，测噪声代价 |
| `ask_then_act` | 先问"Sk 现在是什么状态"，再在同一上下文里让模型决定 | 测"用上失败"：两次回答不一致的比例 |
| `self_tomb` | 提示模型在发现更正时，自己在日志末尾追加一条作废记录 | 闭环模式下"模型自写作废标记"的方法版本 |
| `verified_state` | 状态的每个字段都附上它依据的事件编号 | 测带证据的写入能否降低写错（多轮模式下才有意义） |
| `position_*` | 把旧值和更正分别放在上下文开头、中间、结尾 | 位置效应分析 |

---

## 3. 错误账本

### 3.1 探针上的自动归类（已实现，`build_items.classify`）

每个回答先用正则 `"shelf"\s*:\s*"S(\d+)"` 取最后一个匹配，再按顺序归入第一个成立的类别：

| 标签 | 条件 | 含义 |
|---|---|---|
| `correct` | 等于 `answer` | 正确 |
| `unparsed` | 解析不出货架号 | 格式错误 |
| `stale` | 等于 `deaf_answer` | 用了"忽略关键事件"时的答案：干扰，或覆盖缺失 |
| `occupied` | 选的货架在真实状态里是满的 | 状态重建错误 |
| `missed_lower` | 选的货架编号大于 `answer` | 跳过了更低的空货架 |
| `other` | 以上都不是 | 其他 |

`stale` 在不同条件下的含义不同：
- 在有历史的条件（`raw`、`tomb_*`）里，主要是**干扰**；
- 在缺字段或陈旧状态的条件（`state_noq`、`state_stale`）里，是**覆盖缺失**。

### 3.2 通用错误分类（用于真实基准）

先定义**依赖决策点**：某一步 d，它的正确动作取决于一条事实 f，而 f 在 d 时刻的真值是 v。每个出错的依赖决策点归入第一个成立的类别：

| 类别 | 定义 | 属于写入还是读取误差 | 典型场景 |
|---|---|---|---|
| E1 没覆盖 | 从 f 第一次被观察到直到 d，agent 的表示里从未出现过 f | 写入：覆盖缺失 | schema 里没有对应字段；被摘要压缩掉了 |
| E2 写错 | f 被写入了，但写入时的值和观察到的不一致 | 写入：写错 | 补丁格式合法，但语义抄错 |
| E3 过时 | f 之前写对了，后来真值变了，到 d 时还没更新 | 写入：漏更新 | 漏掉了一次更正 |
| E4 没用上 | d 时刻的表示里有正确的 v，但动作和 v 不一致 | 读取：干扰或用上失败 | 历史里新旧信息都在，模型用了旧的 |
| E5 没取回 | v 在日志里，但到 d 时没有被取进上下文 | 读取：检索 | Scroll、自适应绑定这类"从日志里取"的方法 |
| R 重建 | 对历史类表示：v 可以从日志推出，但模型推错了 | 读取：重建 | 从长日志里推当前状态推错 |
| 其他 | 无法归到某条具体事实的推理或规划错误 | — | — |

对历史类表示（A0–A2），E1–E3 通常不适用，因为没有单独的"写入"。它们的错误落在 R、E4。区分 R 和 E4 的办法：
- 看错误答案是否等于"旧值会推出的答案"（是 → E4 干扰）；
- 或是否对应一个与历史不一致的状态（是 → R 重建）。

### 3.3 识别逻辑：用条件差值做归因

每种干预只去掉一项误差，所以准确率的差值可以归因到具体子项：

| 差值 | 归因到 |
|---|---|
| `tomb_tail` − `raw` | 干扰（只追加的方式能修复的部分） |
| `delete` − `tomb_tail` | 删除相比标注的额外收益（"看见旧值"本身的干扰） |
| `state` − `tomb_tail` | 重建 |
| `state_log` − `state` | 状态旁边加上历史，是帮忙还是添乱 |
| `recall_cand` − `state_stale` | 覆盖缺失里"约束型"的部分 |
| `recall_alt` − `recall_cand` | 覆盖缺失里"机会型"的部分 |
| `control` 场景下 `state` − `raw` | 纯重建负担（没有任何干扰源） |

前提是同一场景、同一种子的配对比较（§9）。

### 3.4 从表示里提取断言（真实基准）

- **结构化对照组**（A3–A6）：直接读状态 JSON。
- **文本对照组**（A0–A2）：用 LLM 提取"agent 当前认为 f 的值是什么"，在人工标注的 100 个子集上报告提取器准确率。
- **"知道 vs 用上"**：在依赖决策点 d，另开一次调用问模型"f 现在是什么"（不影响主轨迹）。知道但没用上，就记为 E4 的"用上失败"子类。

---

## 4. 真值

### 4.1 仓库探针

构造时就精确知道：`sc.shelves`、`sc.quarantined`、`sc.answer`，每条事件都带 `superseded_by`。

### 4.2 ALFWorld（已实现，`state_study/groundtruth/alfworld_facts.py`）

**问题**：ALFWorld 在 `alfred_tw_env.py` 里构造 `EnvInfos` 时，只有手写专家模式（`AlfredExpertType.HANDCODED`）才请求 `facts`（第 96 行）。PoS 的 adapter 只返回观察、可选动作和是否成功。

**做法**：

```python
infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True,
                           extras=["gamefile", "expert_plan"])
wrappers = [AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)]
```

- `make_env(game_file, max_steps=80)`：打开带 facts 的环境；
- `state_from_facts(facts)`：只保留任务相关的谓词 `TRACKED = {atlocation, holds, inreceptacle, opened, isclean, ishot, iscool, issliced, istoggled}`，输出 `agent_at`、`holding`、`in`（物体 → 容器）、`opened`、`props`；
- `replay_expert(game_file)`：用专家计划逐步执行，每步记录状态；
- `check_transitions(records)`：检查每个动作带来的状态变化是否符合预期，例如拿起后在手上、去某处后位置改变、清洗后带 clean 属性。

**命令**：`python -m state_study.groundtruth.alfworld_facts --data DIR --games 12 --stride 11`（`--stride` 每隔 k 局取一局，用来覆盖不同任务类型）。

**已验证**（2026-10-03）：
- valid-unseen 134 局；重放 12 局，覆盖 6 种任务类型（照明查看、拿取放置、清洗、冷却、加热、拿两件）；
- 每局 13–58 个物体；
- 转移检查 0 违规；clean、hot、cool 属性都在对应动作后正确出现。

**接入 PoS 时**：在 PoS 的 ALFWorld adapter 里打开 `facts=True`，每步把 `state_from_facts` 的结果写进事件日志，账本据此判定依赖决策点。规则是：动作的前置条件涉及的对象位置和属性，就是这一步依赖的事实 f。

### 4.3 LOCA-bench【待实现】

- 每次工具调用后，给模拟服务（Canvas、邮件、WooCommerce、BigQuery 等）的数据库和工作区拍快照；
- 先做带 `groundtruth_workspace` 的 3 到 5 个任务家族；
- 依赖决策点：从真值工作区反推出需要哪些事实。先用 LLM 标注，再人工抽查 100 个，报告标注准确率。

### 4.4 LongMemEval 知识更新子集【待实现】

- 78 题，oracle 版和 S 版都跑（Supersede 用的是 oracle 版）；
- 数据里 `has_answer` 标注了证据轮次，`answer_session_ids` 标注了证据会话；
- **还缺**"哪一轮是被取代的旧值"：用 LLM 标注加人工抽查；
- f 就是被更新的属性，d 就是回答问题的那一刻；
- 数据在 Hugging Face 或 Google Drive，这个云环境下载不了，需要用户自己下载。

---

## 5. 方法：自适应绑定（Adaptive Binding）【待实现】

### 5.1 设计原则

由误差分解直接推出，四个部件各自针对一项误差：

| 部件 | 降低哪项误差 | 关键设计 |
|---|---|---|
| ① 尾部状态块 | 重建 | 只存可预测的字段（目标、规则涉及的关键实体）；放在 prompt 末尾，历史部分的缓存不受影响 |
| ② 只追加日志 + 作废标记 | 干扰；同时保证不丢信息 | 发现更正时，在末尾追加一条作废记录，不改动前面的内容 |
| ③ 动作时刻召回 | 覆盖缺失 | 模型先提出候选动作，再从日志召回两类事实：候选本身，以及按规则排在候选前面、被它挤掉的那些选项。提炼成字段后，让模型确认或修改 |
| ④ 写入校验 | 写错 | 状态更新必须引用日志里的事件编号；先执行动作、看到结果，再提交状态 |

### 5.2 prompt 布局（对缓存友好）

```
[固定前缀]   system + 任务规格 + 规则                     ← 永远缓存
[只追加日志] e1, e2, ..., e_t（每条带稳定编号）            ← 只在末尾增长，缓存命中
[作废记录]   夹在日志里按时间追加："e_17 VOID (superseded by e_42)"
---------------------------------------------------------------- 以下每步重写，只占几百 token
[尾部状态]   可预测字段的当前值，每个字段附依据的事件编号
[召回块]     仅在召回轮出现：候选 + 被挤掉选项的提炼行
[当前观察]   最新一条观察
```

只有虚线以下的部分每步变化。delayed-relevance 实测，同样的内容把可变状态放在历史前面，成本是放在后面的 5.7 倍；只追加的历史能省下 82% 的输入成本。

### 5.3 每一步的流程（伪代码）

```python
def step(obs):
    log.append(obs, eid=next_id())                       # ② 只追加
    for c in detect_supersession(obs, log):              # ② 识别更正（规则或模型）
        log.append(f"VOID e{c.old} (superseded by e{c.new})")

    ctx = render(prefix, log, tail_state, obs)           # §5.2 布局
    cand = llm.propose_action(ctx)                       # ③ 第一遍：候选动作

    if should_recall(cand, tail_state, log):             # ③ 召回触发
        entities = extract_entities(cand)                #   候选涉及的实体（货架、文件、联系人）
        displaced = llm.list_preferred_alternatives(ctx, cand)   # 规则下会优先、但被判为不可行的选项
        facts = index.lookup(entities + displaced)       #   按实体建的结构化索引，不用向量检索
        recall_block = distill(facts)                    #   每个实体一行：当前状态 + 依据事件编号
        action = llm.confirm_or_revise(ctx + recall_block, cand)
    else:
        action = cand

    result = env.execute(action)                         # 先执行
    log.append(result, eid=next_id())
    patch = llm.propose_state_patch(ctx, action, result) # ④ 再提交
    if verify(patch, log):                               # ④ 每个字段必须引用存在且支持该值的事件编号
        tail_state.apply(patch)
    else:
        tail_state.mark_unverified(patch.fields)         # 不确定的字段打标记，下次涉及时强制召回
    return action
```

### 5.4 召回范围：为什么必须覆盖"被挤掉的选项"

来自 I4 冒烟测试（Haiku，每格 n=3）：

| 漏掉的信息 | 陈旧状态 | 只召回候选 | 召回候选 + 更低编号货架 |
|---|---|---|---|
| 约束：隔离通知（delayed） | 0/3 | **3/3** | 2/3（1 个 `missed_lower`） |
| 机会：更低货架其实空了（explicit） | 0/3 | **0/3** | **3/3** |

结论：

- 只检查"我打算做的动作是否合法"，只能修复**违反约束**这类错误；
- 修复**错过更好选项**这类错误，必须召回规则下排在候选前面的那些选项；
- 召回内容变多会引入噪声（delayed 场景多了 1 个错误），所以召回范围是消融维度之一。

**通用任务里怎么确定"被挤掉的选项"**：
- 有明确排序规则的任务（仓库：编号更低的货架），由规则直接给出；
- 没有显式规则时，让模型列出"如果可行就会优先选择"的 k 个替代动作（`list_preferred_alternatives`），再召回这些动作涉及的实体。k 取 3–5，作为超参数做消融。

### 5.5 召回触发

| 触发策略 | 说明 | 成本 |
|---|---|---|
| 每步都召回 | 上限 | 每步多 2 次调用 |
| 实体久未出现 | 候选涉及的实体，最近一次出现在 ≥ τ 个事件之前 | 省钱版本 |
| 状态字段未验证 | ④ 标为 unverified 的字段被涉及时 | 和写入校验联动 |
| 模型自决 | 给模型一个 `recall(entity)` 工具 | 测模型的元认知 |

### 5.6 写入校验（④）

- **补丁格式**：`{"field": ..., "value": ..., "evidence": ["e42", "e57"]}`；
- **校验规则**：引用的事件必须存在；对可规则化的字段（例如货架状态），由一个确定性的检查器从被引用的事件推出值并比较；不可规则化的字段，只检查证据存在且相关（让模型自检，或抽样人工检查）；
- **时序**：先执行、看到结果，再提交。这一点修正了读书笔记指出的 SKILL.state 问题：它先提交状态更新再执行动作，一次失败的工具调用会让状态显示"已成功"。

### 5.7 消融

1. 逐个去掉 ①–④；
2. 召回范围：只召回候选 / 候选 + 被挤掉的选项 / 全部；
3. 召回内容：提炼成一行 vs 原文贴入；
4. 召回触发：每步 / 实体久未出现 / 字段未验证 / 模型自决；
5. 作废标记：由 harness 用规则识别 vs 由模型识别。

### 5.8 和相近方法的区别

| 方法 | 它怎么做 | 本方法的区别 |
|---|---|---|
| Scroll | 模型在常驻 kernel 里写代码查询日志，查询时机不固定 | 召回在**动作发生时**触发，并且召回范围由"候选 + 被挤掉的选项"决定 |
| ContextRender | 按工具结果之间的依赖关系决定保留什么 | 我们按"候选动作涉及的实体"召回，不需要依赖图 |
| PoS | 显式信念状态 + 一致性校验 + 卡住检测 | PoS 不保留历史作为依据；我们保留只追加日志，覆盖缺失可以靠召回补 |
| SKILL.state | 人写 schema，历史丢弃，先提交后执行 | 历史保留；先执行后提交；字段带证据 |
| MemTX、PatchOptic 等 | 事务化、带证据的写入 | 只对应我们的部件 ④，我们的重点在 ③ 和误差分解 |

**新意**：在动作发生时取回；召回范围覆盖候选和被挤掉的选项；对缓存友好；每个部件都能用错误账本验证它修复了哪一项误差。

### 5.9 成本

每步额外的调用：提出候选（1 次）、确认或修改（召回轮 1 次）、状态补丁（1 次）。全部计入成本并单独报告（§10）。省钱版本只在实体久未出现或字段未验证时召回。

---

## 6. 对照组（真实基准上的实验臂）

### 6.1 底座：PoS 的 agent 循环

- 代码：`third_party/methods/pos`（MIT，固定在 `d6acb43`）；
- 上下文策略通过 `ContextProvider` 接口注入（`contexts/base.py`），只有 4 个方法：`reset`、`update`、`get_context`、`get_result`。原始轨迹的实现 `contexts/raw.py` 只有 45 行；
- 已接好 ALFWorld、LOCA-Bench、RCA-100、ClinDiag，自带 `raw.yaml` 和 `pos.yaml`；
- 云端已装好，自带的离线检查通过（"Passed: Raw, full PoS, both ablations, and diagnostic PoS"），28 个单元测试通过；
- **不改上游代码**：需要改的地方用子类或 patch 文件实现。

### 6.2 对照组定义

| 编号 | 名称 | 策略模型每步看到什么 | 谁来写状态 | schema 从哪来 | 保留历史吗 | 缓存布局 |
|---|---|---|---|---|---|---|
| A0 | Raw | 任务 + 完整的动作和观察历史 | — | — | 全部 | 只追加，对缓存最友好 |
| A1 | Summary | 任务 + 摘要 + 最近 k 步 | LLM 按阈值写摘要 | — | 被摘要替换 | 每次摘要都打断缓存 |
| A2 | CLM | 任务 + 可编辑的上下文文本 | 模型用 Python 改写 | 模型自己长出来 | 由模型决定 | 编辑点之后全部打断 |
| A3 | Schema-State | 任务 + JSON 状态 + 最新观察 | 模型输出补丁 | 人按基准写一次 | 丢弃 | 每步变，但 prompt 很短 |
| A4 | Self-Schema | 同 A3 | 模型输出补丁 | 模型在任务开始时设计，中途可扩展 | 丢弃 | 同 A3 |
| A5 | PoS | 任务 + 信念状态 + 当前缺口 | 信念管理器 + 校验器 | 框架规定 | 不作为依据 | 每步变 |
| A6 | State+Log | 只追加日志 + 末尾状态块 | 模型输出补丁 | 人写（或 A4 的方式） | 全部 | 历史可缓存，只有尾部在变 |
| AB | 自适应绑定 | 见 §5 | 模型 + 校验器 | 人写的最小 schema + 召回 | 全部 | 同 A6 |
| 参考 | VISTA、Scroll | 各自的官方实现，只在 LOCA 上跑 | — | — | — | — |

实现细节：

- **A2（CLM）**：ALFWorld 和 LOCA 的循环里没有 bash 沙箱，所以提供一个 `edit_context(python_code)` 工具。渲染和解析直接复用 `clm/clm_harness/context_utils/context_string.py` 的 `render_editable` 和 `parse_back`。注意 CLM 的两个已知问题（角色折叠和头伪造，见 [clm_notes.md](clm_notes.md) §7），实验里要么修掉，要么在所有臂上保持一致。
- **A3**：参考 `third_party/methods/skill-state-runtime`（MIT）。先执行后提交，修正上游问题。
- **A6**：状态块放在末尾。

### 6.3 公平性约定

- 所有组统一：步数上限、工具集合、观察截断规则、JSON 重试次数、温度；
- **任何组都不享有额外待遇**。CLM 论文里只有 CLM 有三项：预算提醒、编辑轮不计步数、超预算回滚。要么所有组都有，要么都没有；
- 额外的 LLM 调用（PoS 的校验器、自适应绑定的召回）计入成本，并单独报告。

### 6.4 附录：各基准的状态 schema 草案

见 [proposal.md](proposal.md) 附录 A（ALFWorld、仓库探针、LOCA 通用部分）。

---

## 7. 多轮闭环模式（H2）【待实现】

### 7.1 为什么需要

单题模式下，前缀由 oracle 保证正确，所以测不出"错误会不会传染到后面的步骤"。H2（写入误差会粘住，读取误差是一次性的）只能在闭环里测。

### 7.2 设计

- **驱动器**：环境逐步给出事件；每次到货都由模型决定货架；环境按模型的**实际**动作更新世界状态。如果模型放错了，世界状态就照着错的放法继续走，但正确性仍按规则对照真实状态判定；
- **表示**：每种条件在闭环里各自维护。`raw` 追加模型自己的动作和结果；`state` 由模型输出补丁；`state_log`、自适应绑定同理；
- **注入写入错误**（因果检验）：在第 t 步把状态里的某个字段改错（例如把空货架标为占用），观察之后多少步还会错。对照：在 `raw` 条件下往历史里插入一条等价的错误记录；
- **每回合长度**：100–300 次决策，按 §1.5 的旋钮控制事件密度。

### 7.3 指标

- `P(err_{t+1} | err_t)` 与 `P(err_{t+1} | ok_t)` 之比；
- 连续出错长度的分布（run length）；
- 注入错误之后的恢复时间：从注入到第一次答对之间的步数；
- 依赖步准确率（只统计依赖关键事件的步）。

### 7.4 预测

- 状态类条件：注入错误之后长时间持续出错，run length 呈重尾；
- 历史类条件：错误更分散，下一步重新读时有一定概率恢复；
- 自适应绑定：④ 的写入校验应把状态类的持续错误压下去。

---

## 8. 各假设的实验方案

| 假设 | 因素与水平 | 条件 | 模型 | 每格样本 | 主图 |
|---|---|---|---|---|---|
| **H1** 事件数而非 token 数 | 扫描 A：`n_telemetry` ∈ {0, 100, 300, 1000}；扫描 B：`n_state_events` ∈ {20, 60, 150, 400}（等 token） | `raw`、`tomb_tail`、`state`、`state_log` | 3 个以上规模 | 20 个以上种子，配对 | 两条曲线：准确率 vs token（平），准确率 vs 事件数（降） |
| **H2** 写入误差会粘住 | 闭环；注入错误 vs 不注入 | `raw`、`state`、`state_log`、AB | 2 个规模 | 每格 20 回合 | run length 分布；条件出错概率 |
| **H3** 交叉点 | 回合长度 × 需求可预测性（`state_noq` 式的缺字段概率 p_miss ∈ {0, 0.1, 0.3, 0.5}） | `raw`、`state`、AB | 3 个规模 | 20 | 相图：每格标出哪种表示赢 |
| **H4** 预测真实基准 | 探针上拟合的参数 + 基准的统计量 | A0、A3、A6、AB | 2–3 个 | 按基准 | 预测胜负 vs 实际胜负 |
| **H5** 规模不对称 | 规模 × 误差子项 | 全部探针条件 | 3 个以上规模、2 个以上家族 | 20 | 每个子项的误差率随规模变化的曲线 |

### 8.1 H4 的预测模型（初版）

把每种表示方式的错误率写成两项之和：

- 历史类：`err_hist ≈ r(n_events) + i(n_conflicts)`
  - `r`：重建误差，随相关状态变化次数增长；
  - `i`：干扰误差，随同一事实的冲突副本数增长。
- 状态类：`err_state ≈ p_miss + w`
  - `p_miss`：任务需要的事实不在 schema 里的比例（需求可预测性的反面）；
  - `w`：每次写入出错的概率，乘以它影响到的依赖步数。

步骤：
1. 在探针上拟合 `r`、`i`、`w`（每个模型分别拟合）；
2. 在真实基准上测任务统计量：`n_events`（每个依赖决策点之前相关状态变化的次数）、`n_conflicts`、`p_miss`（用人写 schema 去覆盖依赖事实，看漏掉多少）；
3. 预测哪种表示赢、赢多少，再和实测比较。

**预注册**：在跑真实基准之前，把拟合出的参数和预测写进 `state_study/preregistration/`，带时间戳提交。

### 8.2 可以预先写好的预测

- 固定 token 数、增加状态变化次数时，`raw` 的准确率单调下降；
- 固定状态变化次数、增加遥测 token 时，`raw` 的准确率基本不变或只轻微下降；
- `tomb_tail` 的提升随 `n_voided` 增大而增大；
- `state` 在 `control` 场景下相对 `raw` 的提升，随 `n_state_events` 增大而增大；
- 模型越强，`occupied`（重建）类错误下降越快；`state_noq` 的 0% 不随规模变化。

---

## 9. 统计协议

delayed-relevance 复现项目的教训：它文档里至少有四个结论在测量噪声底之后被推翻（例如 notes 字段的 60% 实为 21%，ReAct 的 0% 实为 17%）。所以：

1. **先测噪声底**：每组大实验之前，选 2 个条件 × 10 个用例，用同一种子重复 8 次。
2. **配对设计**：同一场景、同一种子，比较不同条件。
3. **区间**：成功率报告 Wilson 95% 置信区间；配对比较用 McNemar 检验；多组比较做 Holm 校正。
4. **按依赖步计数**：单位不是回合，而是"每个正确动作取决于那条事实的步骤"。
5. **按测得的依赖步数选种子**：先不调 API，算出每个种子有多少个依赖步，选依赖步多的种子。delayed-relevance 里，换种子后同样成本多出 4 倍信号。
6. **自适应加密**：先用少量种子扫出每个模型"开始出错"的区间，再只在那附近加密采样。
7. **事先写好**：样本量、种子选择规则、排除规则（例如截断的回答）。
8. **温度**：固定为 0；API 不支持的模型，用重复次数估计方差。

---

## 10. 成本核算

### 10.1 三个口径同时报告

1. **厂商实际计费**：区分命中缓存和没命中的 token。Anthropic 的缓存读取约 0.1 倍价，写入约 1.25 倍价，以官方价格表为准；
2. **prefix-reuse FLOPs**（开源模型）：复用 `clm/clm_harness/flops_metrics/kv_cache_flops.py`。它按 vLLM 默认的 KV 块大小（`DEFAULT_BLOCK_SIZE = 16`）计算前缀复用，把编辑导致的重新 prefill 算进去。`flops_model_key` 只支持 `9b` 和 `27b`；API 模型用 `cost_metric=usd`；
3. **原始 token 数**：用于和已发表论文对齐。

**排序以前两个口径为准。**

### 10.2 每种表示的缓存行为

| 表示 | 每步变化的部分 | 缓存命中率预期 |
|---|---|---|
| A0 Raw | 只在末尾追加 | 约 80%（delayed-relevance 实测 82%） |
| A1 Summary | 每次摘要后整段重写 | 摘要之间命中，摘要时全失效 |
| A2 CLM | 编辑点之后全部重算 | 取决于编辑位置和频率 |
| A3/A4 | 整段每步重写，但很短 | 0%，但总量小 |
| A5 PoS | 每步变 | 低 |
| A6、AB | 只有尾部状态和召回块 | 接近 A0 |

delayed-relevance 的数据（T=50，3 个种子）：ReAct 原始 82.6 万 token，计费 15.2 万（省 82%）；Stateful 87.3 万，计费 87.3 万（省 0%）；SKILL.state 10.9 万，计费 10.9 万。**token 优势 7.54 倍，计费优势只剩 1.39 倍**。

### 10.3 记录要求

每一步记录：发给模型的完整 prompt（用来回放缓存命中情况）、状态快照、动作和观察、token 用量（含命中缓存的部分）。在 PoS 已有的 `events.jsonl` 上加字段；条件允许时与 CLM 的 ATIF-CTX 轨迹格式兼容。

---

## 11. 冒烟测试协议（没有 API 时的子 agent 中转）

**用途只限于验证管线**，不能当实验数据。

### 11.1 流程

1. `build_items build` 生成题目和答案：题目在 `items/<id>.txt`，答案单独在 `key.jsonl`；
2. **盲化**：文件名改成 `sha1(salt + id)` 的前 10 位，对应关系存在 `blind_map.json`，子 agent 看不到条件名；
3. 每题启动一个 Claude 子 agent，用 Read 工具读入盲化后的提示文件，只返回 JSON；
4. 回答写入 `responses.jsonl`，每行 `{"id", "model", "response"}`；
5. `build_items score` 打分，输出每个 `(model, kind, condition)` 的正确数和账本分布。

### 11.2 局限（每次报告都要写明）

- 子 agent 带着 Claude Code 自己的系统提示和工具，不是干净的 API 调用；
- 温度、采样不可控，拿不到 token 用量和缓存命中；
- 每格 n=3，没有统计力；
- 模型版本不受控制，凑不出同一家族的规模梯度。

### 11.3 复现命令

```bash
# 第一轮（I3）
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 \
  --plan explicit:raw,tomb_tail,tomb_inline,delete,state,state_log implicit:raw,tomb_tail,state delayed:raw,state,state_noq
# 第二轮（I1 + I4）
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 \
  --plan delayed:recall_cand,recall_alt explicit:state_stale,recall_cand,recall_alt,raw,state,self_schema
python -m state_study.probes.build_items score --out OUT
```

---

## 12. 待实现清单（按主线的阶段排列）

| 阶段 | 任务 | 位置 | 依赖 |
|---|---|---|---|
| 1 | H1 旋钮：`n_state_events`、`n_telemetry`、`n_voided`，记录 `n_relevant_events` | `warehouse.py` | 无 |
| 1 | `control` 场景 | `warehouse.py`、`render.py`、测试 | 无 |
| 1 | `state_head`、`recall_raw`、`recall_all`、`ask_then_act` 条件 | `render.py`、测试 | 无 |
| 1 | API 调用器：温度 0、记录 token 和缓存命中、断点续跑 | `state_study/runner/` | API 密钥 |
| 1 | 噪声底测量脚本 | `state_study/runner/` | API |
| 2 | 领域 2（代码仓库）、领域 3（个人助理）生成器，各配规则读者测试 | `state_study/probes/` | 无 |
| 2 | 闭环驱动器 + 错误注入 | `state_study/closed_loop/` | 无（跑需要 API） |
| 3 | A0–A6、AB 的 `ContextProvider` | `state_study/providers/` | PoS |
| 3 | PoS ALFWorld adapter 打开 facts，接入账本 | patch 文件 | 无 |
| 3 | ALFWorld 扰动：中途移动一个物体，同时发一条通知（explicit），或不发通知（implicit，只能靠观察发现） | `state_study/adapters/alfworld_perturb.py` | 无 |
| 3 | LongMemEval-KU 适配：raw、作废标注、删除、笔记、AB 五个条件 | `state_study/adapters/longmemeval_ku.py` | 数据需用户下载 |
| 3 | LOCA 逐步快照（3–5 个家族） | `state_study/groundtruth/loca_snapshot.py` | LOCA 环境（Node.js、MCP） |
| 3 | 成本模块 | `state_study/cost/` | 无 |
| 3 | 预注册：H4 的参数和预测 | `state_study/preregistration/` | 阶段 1、2 的结果 |
