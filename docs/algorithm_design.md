# 算法与实验设计：Self-Paging LM

主线的"为什么"和"做什么"见 [research_line.md](research_line.md)；本文档写"怎么做"，包括形式化、协议、推理和训练的实现、代价口径、基线实现、E0–E2 的实验协议、SFT 和 RL 的细节，以及待实现清单。

旧主线的算法设计归档在 [`archive/state_vs_history/algorithm_design.md`](archive/state_vs_history/algorithm_design.md)。

---

## 1. 形式化

| 符号 | 含义 |
|---|---|
| L | KV 日志。按生成或输入的顺序只追加；第 j 个 token 的全局位置就是 j，**永不改变** |
| pg(j) | token j 所属的页 |
| 页 | 系统切分的单位：每个工具观察一页、每个 assistant 回合一页、每段 `<note>` 一页。每页由**页头**（第一行，系统渲染，常驻）和**正文**两部分组成 |
| R_t | 处理 token t 时的驻留集（页的集合）。token t 只能注意 R_t 里的页，以及它自己所在页里排在它前面的 token |
| B | 驻留预算（token 数），要求 Σ_{p∈R_t} \|p\| ≤ B，页头和钉住的页都计入 |
| H | 内存层：被换出、可以换回的页 |
| 钉住 | system prompt、任务描述、全部页头、当前正在写的页，永远驻留 |

**写入时可见性**：token j 的 KV 由它写入那一刻的 R_j 决定，之后再也不重算。这条规则同时带来 P1–P4（[research_line.md](research_line.md) §2）。

**和 CLM 的区别**：CLM 改写的是 L 本身（文本被替换，之后的 token 重新 prefill，KV 在新历史下重算）；本文 L 不变，只改 R。

## 2. 协议

### 2.1 渲染

```
[[PAGE 12 tool=search q="Helix Pharma recall 2021"]]      ← 页头，常驻
<正文：工具输出全文>
```

- assistant 回合的页头：`[[PAGE 13 role=assistant]]`；
- `<note>…</note>` 的内容单独成页，页头为 `[[PAGE 14 note]]`。这样模型可以换出自己冗长的推理，只留下笔记；
- 页头格式只能由系统生成。**工具输出里出现的 `[[PAGE` 一律转义**，吸取 CLM 角色头可以被伪造的教训（[clm_notes.md](clm_notes.md) §7.2）。

### 2.2 动作标签

| 标签 | 语义 | 合法条件 |
|---|---|---|
| `<out p="12,15-17"/>` | 把页移到 H | 页存在、驻留、没有钉住 |
| `<in p="12"/>` | 从 H 换回 | 页在 H 里，换回后不超过 B |
| `<drop p="12"/>` | 删除，不可逆 | 只在"没有内存层"的对照条件里开放 |

- 标签闭合（`/>`）的那一刻生效。标签自己的 token 属于当前页；
- 非法动作不生效，系统在下一轮附一行说明（和 CLM 的 edit gate 一样，[clm_notes.md](clm_notes.md) §2.1）；
- 一个标签里可以写多个页，减少控制 token。

### 2.3 仪表盘

每轮在工具输出之后附一行，作为系统页；旧的仪表盘页由系统自动换出，不计入模型的动作：

```
[KV] resident 21.3K/32K (67%) | pages on GPU: 1-3,9,12-14 | in host: 4-8,10-11
```

- 用量到 50%、75%、90% 时各加一句提醒；
- **超预算**：新观察到达后如果超过 B，系统按 LRU 换出最久没被注意的非钉住页，并记一次"强制换出"。它计入代价，也在仪表盘上说明。

### 2.4 零样本 prompt 要点

- 解释页、页头、三种标签和仪表盘；
- 告诉模型：换出不丢信息，随时能换回，但换回之前看不到正文；
- 告诉模型：写笔记会占预算，但笔记可以替代整页正文；
- 给 2 个简短示例，一个换出后又换回，一个先写笔记再换出。

## 3. 推理实现（HF transformers 原型）

### 3.1 数据结构

```
PageStore
  log_len: int                         # 下一个 token 的全局位置
  pages: dict[pid, Page]
Page
  pid, kind (header|body|note|system), pinned: bool
  span: (start, end)                   # 全局位置区间
  where: gpu | host | dropped
  kv: list[(K, V)] per layer           # 形状 [kv_heads, n, head_dim]，已经加过 RoPE
  last_attended_step: int              # 给 LRU 用
```

### 3.2 前向

1. 把驻留页的 KV 按全局位置排序，逐层拼成一个连续的 `DynamicCache`，长度记为 L_res；
2. 新 token（工具观察的 prefill，或一个解码 token）调用 `model(input_ids, past_key_values=cache, position_ids=全局位置, cache_position=arange(L_res, L_res+n))`；
   - `position_ids` 只决定 RoPE 用的位置号。全局位置有空洞没关系；
   - `cache_position` 只用来建因果掩码和往 cache 里写入；
3. 把 cache 尾部新增的 KV 切出来，写回 PageStore 里对应的页。

**为什么精确**：Qwen3 写进 cache 的 key 已经旋转过（HF 的实现里 RoPE 在 `cache.update` 之前）。所以 query 和 key 的点积只取决于两者的全局位置差，和 key 在 cache 里排第几无关；softmax 对 key 的顺序也不敏感。这样拼出来的结果，等于"整条序列一次前向、加上驻留集掩码"的结果。§7 的等价性测试负责验证这一点。

### 3.3 优化

- 只有 R 改变时才重新拼 cache，平时照常追加；
- 换出用锁页内存加 `non_blocking` 拷贝；
- 后续接入服务框架：vLLM 的 block table 本身就是页表，换出对应 swap out。Leyline、vToken 说明这条路走得通。

### 3.4 模型限制

- 只用纯 dense、全注意力的模型：Qwen3-4B / 8B，不开滑动窗口；
- 混合线性注意力模型（Qwen3.5/3.6 的 GatedDeltaNet 层）的循环状态没法换页，不在主实验里。

## 4. 训练实现（单次前向）

### 4.1 掩码

把一条轨迹按动作生效的时刻切成若干**段**；每个 token i 有段号 s(i)，同一段里驻留集不变。Res[s][p] 表示段 s 时页 p 是否驻留。

```
mask(i, j) = (j ≤ i) and ( pg(j) == pg(i)  or  Res[s(i)][pg(j)] )
```

- 位置号 = 全局位置，和推理完全一致；
- 用 FlexAttention 的 `create_block_mask` 实现。页和 128 token 的块不对齐也没关系，部分块会正常处理；FlexAttention 支持 Ampere 卡；
- 测试时用稠密 4D 掩码走 eager 或 SDPA 路径，和推理路径逐位比较（§7）。

### 4.2 损失

- 只在 assistant token 上算损失；
- 角色掩码 m[t]：控制标签和 `<note>` 里的 token 为 1，其他为 0，给 RL 的效率项用；
- LoRA 起步：rank 16、alpha 32，挂在注意力和 MLP 上。

## 5. 代价口径

记每个解码步、每个 prefill token 的成本：

```
C = Σ_{解码步 t} (κ + |R_t|)  +  Σ_{prefill token} (κ + |R|)
```

- |R_t| 是该步读的 KV token 数，按每 token 的 KV 字节数折算；
- κ 是"每生成或输入一个 token 的权重读取"，**折算成 KV token 数**，它取决于服务时的 batch 大小。按 Qwen3-8B 估算：bf16 权重约 16 GB，KV 约 144 KB/token，所以：
  - batch = 1 时 κ ≈ 11 万 token：权重占大头，**多生成控制 token 很贵**；
  - batch = 32 时 κ ≈ 3.4K token：上下文一长，KV 读取占大头。
- 主文报告 batch = 1 和 batch = 32 两种折算；另外单独报告峰值驻留、C_pcie（换入换出字节数）、强制换出次数；
- CLM 一侧的重新 prefill 用 `clm/clm_harness/flops_metrics/kv_cache_flops.py` 的 prefix-reuse 口径算，再按同样的方式折算；
- 上面的字节数是按 Qwen3-8B 公开配置的估算（36 层、8 个 KV 头、head_dim 128），**需要对照实际 config 核对**（[research_log.md](research_log.md) §7.1）。

## 6. 基线的实现

所有"系统决定"的基线都在同一个 PageStore 上实现，只是由系统代替模型改 R，以保证公平：

| 基线 | 规则 |
|---|---|
| Full | 不设预算 |
| Recent | 钉住的页 + 最近的页，直到填满 B（StreamingLLM 的页级版本） |
| AttnEvict | 新观察到达时，按"最近一个窗口的 query 对各页注意力的累计值"淘汰最低的页，不可逆（SnapKV / H2O 的页级版本） |
| QuestPage | 每页存一个代表 key（各层各头的均值）；每个回合开始时用当前 query 选 top 页，可逆（Quest / InfLLM 的页级版本） |
| Belady | 知道未来每页什么时候要用，换出"下次使用最远"的页，并在使用前换回。这是固定容量下缺页最少的最优策略，作为上界和 SFT 的 oracle |
| CLM | 原版 harness 的文本编辑，同一个模型；另跑一组加 SCR 的 |
| MemGPT 式 | 文本存档和召回工具，召回时重新 prefill |

## 7. 等价性测试（阶段 0，云端 CPU 即可）

用随机初始化的小 Qwen3 配置（例如 2 层、hidden 64），不需要下载权重：

1. **拼 cache 推理 = 掩码单次前向**：随机生成一条带若干次换出、换回的轨迹，比较两条路径每个位置的 logits，误差 < 1e-5（fp32）；
2. **换回逐位相同**：换出再换回后，KV 张量和换出前完全相等；
3. **位置空洞**：驻留集不连续时，第 1 条仍然成立；
4. **协议解析**：标签解析、非法动作、钉住页、页头转义的单元测试；
5. **Belady 最优性**：在小的访问序列上暴力枚举，验证 Belady 的缺页数最少；
6. **规则读者**：在 T1 上，规则读者只看 Belady 驻留集里的文本（加上笔记）就能 100% 答对。这样模型答错就不能归因于 oracle 驻留集信息不够（项目约定，见 CLAUDE.md）。

## 8. 测试床 T1：多轮仓库环境

基于 `state_study/probes/warehouse.py`：

- 环境每轮推送一批事件（到货、出库、隔离、更正、盘点、遥测），作为一页工具观察；
- 遇到"决策"事件时，模型必须回答货架号。规则是选编号最小的、空的、没被隔离的货架；
- 旋钮：
  - 每批事件数、遥测比例（只加 token）；
  - 状态变化次数；
  - 关键事件类型：`explicit` 更正、`implicit` 盘点、`delayed` 隔离；
  - 决策点之间的距离（控制"以后要用什么"的可预测性，给 K4 用）；
  - 预算 B。
- 真值：每一步的世界状态，以及每页里的"活事实"（还没被取代的入库记录、还在生效的隔离），由生成器直接给出；
- **为什么适合测相似度检索的盲区**：`delayed` 隔离通知和"把新到的货放哪"这个问题在字面上不相似，但按规则它决定答案。QuestPage 式的相似度选择应该在这里失效，这正是"让模型决定"可能有空间的地方。

## 9. 继续/止损实验协议

都只需要推理，用 Qwen3-8B（E1 加 4B）；同一批场景在各条件间配对；二值结果用 McNemar 检验，区间用 bootstrap。

### E0 提升空间

- **问题**：同一预算下，系统换页离 oracle 有多远？
- **设置**：T1 的三种关键事件 × 预算 {25%, 50%, 100%} × 策略 {Full, Recent, AttnEvict, QuestPage, Belady}，每格 100 个场景；
- **指标**：决策准确率；错误按 `stale` / `occupied` / 其他分类，沿用 `build_items.classify`；
- **判断**：在 `delayed` 上，Belady − QuestPage < 3 个点，**而且**在 T2 抽样的 100 题上也是这样，就判定准确率空间不足，主线转向成本、受控对比和残留。

### E1 零样本协议

- **设置**：T1 多轮，预算 50%；Qwen3-4B 和 8B；每个模型 100 条轨迹；
- **指标**：标签语法合法率、非法动作率、强制换出次数、换出和换回次数、准确率、代价 C；
- **判断**：8B 合法率 < 80% 时，SFT 是必需的，阶段 2 要加大数据量。

### E2 残留

- **问题**：页换出之后，它驻留期间写下的 KV 还带着它的信息，这会不会让模型答出旧值？
- **设置**：T1 的 `explicit` 场景。页 a 有旧的入库记录（货架 k 有货），页 b 是更正（"从未完成"，k 其实是空的），之后是决策点，正确答案是 k。四个条件：

| 条件 | 页 a | 页 a 之后的 KV 怎么算的 |
|---|---|---|
| FULL | 驻留 | 带着 a |
| OUT-LATE | 决策前才换出 | 全部带着 a |
| OUT-EARLY | 收到 b 后立刻换出 | (a, b) 之间的页带着 a；b 之后的页没有 |
| REWRITE | 删掉文本，后面全部重新 prefill（CLM 的做法） | 都不带 a |

- **识别**：OUT-LATE、OUT-EARLY、REWRITE 三者**可见文本完全相同**，差别只在 KV 是在什么历史下算出来的。所以：
  - OUT-LATE − REWRITE = 全部残留；
  - OUT-EARLY − REWRITE = 只由 (a, b) 之间的页带来的残留；
- **剂量**：a 和决策点之间的 token 数取 {1K, 4K, 16K}，看旧值答案率是否随之上升；
- **指标**：旧值答案率（`stale`）、准确率；
- **结论**：残留可以忽略，就说明 KV 操作可以放心替代文本删除；残留显著，就是 K3 的主结果，并且引出"写一条更正笔记"或 KVEraser 式修正的对比。

## 10. SFT

### 10.1 T1：程序 oracle 轨迹

三种 oracle，产生三份数据，用来考察 K4：

| oracle | 行为 |
|---|---|
| Belady-paging | 只用 `<out>` 和 `<in>`，按 Belady 换页 |
| note-then-out | 每页到达时把活事实写成一条模板笔记，然后换出正文 |
| mixed | 有预算压力时，可预测要用的写笔记，其余换页 |

- 推理文字用模板生成，例如"页 7 的隔离通知在 k 号货架释放前一直有效，先写笔记"。它只教格式，真正的决策边界留给 RL 去校准；
- 每份约 3000 条轨迹，按场景种子和测试集严格分开。

### 10.2 T2：拒绝采样

- 零样本策略每题采 8 条轨迹，只保留成功、没超预算、协议合法的；
- 每题最多保留 2 条，避免简单题占多数。

## 11. RL

- **算法**：GRPO，每题 G = 8；预算 B 每条轨迹随机取 {25%, 50%, 75%}，让策略见过不同的压力；
- **奖励**：任务成功 r ∈ {0, 1}；
- **advantage**（沿用 CLM 的双通道，[clm_notes.md](clm_notes.md) §5）：

```
A[i, t] = r̂_i + w_eff · a_eff_i · m_i[t]
a_eff_i = clip((C̄ − C_i) / C̄, −1, 1)     # 只在组内成功轨迹之间比；成功少于 2 条的组为 0
```

  - C 按 §5 计算，κ 取 batch = 32 的值；
  - w_eff = 0.25；m 是控制标签和笔记 token 的掩码；
  - 强制换出另外每次计入一个固定成本；
- **rollout**：HF 循环加 PageStore；同一题的 G 条轨迹共享 prompt 的 KV（复制 cache）；
- **训练**：§4 的单次前向掩码，整条轨迹一条序列，不需要切段。

## 12. 待实现清单

新包 `self_paging/`，放在仓库根目录：

| 文件 | 内容 | 阶段 |
|---|---|---|
| `pages.py` | Page、PageStore、驻留集的不变量检查 | 0 |
| `protocol.py` | 渲染（页头、转义、仪表盘）、标签解析、合法性检查、超预算处理 | 0 |
| `runner_hf.py` | HF 推理循环：prefill 观察、解码、执行动作、重新拼 cache | 0 |
| `masks.py` | 从动作日志生成页号、段号和 Res 矩阵，再生成稠密 4D 掩码和 FlexAttention BlockMask | 0 |
| `costs.py` | C、峰值驻留、C_pcie，两种 κ 的折算 | 0 |
| `policies.py` | Full、Recent、AttnEvict、QuestPage、Belady | 0–1 |
| `envs/t1_warehouse.py` | 多轮仓库环境、真值、活事实标注、规则读者 | 0–1 |
| `experiments/e0.py`、`e1.py`、`e2.py` | 三个实验的驱动和打分 | 1 |
| `train/sft.py`、`train/grpo.py` | SFT 和 RL | 2–3 |
| `envs/t2_bcp.py` | BrowseComp-Plus 接入，本地检索和本地评判 | 2 |
| `tests/` | §7 的六组测试 | 0 |

依赖：torch（CPU 版即可跑测试；FlexAttention 需要 2.5 以上）、transformers（需要支持 Qwen3 的版本）。云端的虚拟环境放在 scratchpad 里，版本号写进 `self_paging/README.md`。
