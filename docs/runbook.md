# 运行手册：准备什么、按什么顺序跑、每一步怎么验收

本手册写给在用户服务器上执行开发和训练的 agent。读完它和 [`self_paging/README.md`](../self_paging/README.md)（代码文档），就可以开始干活。方法背景见 [research_line.md](research_line.md)；具体设计见 [experiment_plan.md](experiment_plan.md)（v2，Qwen3-8B）和 [data_and_training.md](data_and_training.md)。

---

## 0. 目标和规矩

**研究目标**：让模型自己管理 KV 缓存。模型通过 `kv_out` / `kv_in` / `note` 三个工具决定哪些页留在显存里，并用 SFT + RL 训练它把这件事做好；再和系统换页（Recent、Belady）以及文本编辑（CLM、ACM）比较准确率和代价。

**必须遵守**：
1. **token 进、token 出**：训练数据和 rollout 记录里的 `tokens` 就是喂给模型的那串 id，任何环节都不能把文本重新分词再训练；
2. 改了 `hf/logstore.py`、`hf/masks.py`、`train/batch.py`、`agent/episode.py` 中的任何一个，都要重跑 `self_paging/tests`。其中 `test_rollout_logprobs_match_training_pass` 必须通过，否则 RL 的重要性比率是错的；
3. 主实验只用 dense 的 Qwen3-8B。不要换成 Qwen3.5/3.6 这类混合线性注意力模型（理由见 experiment_plan §1.3）；
4. 日志超过 32K 的基准，对**所有方法**统一开 YaRN（`--yarn 4`）；T1 不开；
5. 评测集不能进入任何训练数据：T1 的测试种子从 100000 起，训练种子用 0–99999；BCP 的评测 150 题不能进训练；
6. 墙钟延迟只在接入 SGLang、真正换页之后才能报告。HF 原型只报告解析代价（`cost`、`peak_resident`）；
7. 不要改 `clm/`、`suffix_cache_reuse/`（上游代码）；没有 license 的仓库（VISTA、belief-world-models、delayed-relevance）不能复制代码；ACM（MIT）可以复用，注明出处。

---

## 1. 准备清单

### 1.1 硬件

| 用途 | 最低 | 建议 |
|---|---|---|
| 测试、造 SFT 数据（oracle） | CPU 即可 | — |
| 评测、拒绝采样、GRPO rollout（8B bf16 推理） | 1 张 ≥ 40 GB 的卡 | 每张卡独立跑一个进程，按种子区间并行 |
| SFT 全参数 | 8 × 80 GB（FSDP2） | 卡少就用 LoRA（`--lora-r 64`），单卡 80 GB 可以跑 |
| GRPO | 每卡一份完整副本 + LoRA | 8 卡 |

### 1.2 软件

```bash
uv venv -p 3.11 .venv && . .venv/bin/activate
uv pip install torch            # CUDA 版，torch ≥ 2.5（FlexAttention）；云端测试用的是 2.15
uv pip install "transformers==5.23.0" peft numpy pytest
```

- transformers 必须是 5.x：代码用到 `DynamicCache.layers`、`set_attn_implementation`，前向不传 `cache_position`；
- BCP 那条线另外需要 `pyserini`、JDK 21（设置 `JAVA_HOME`）、`openai`（评判客户端），以及 SGLang 或 vLLM（用来起评判服务）。

### 1.3 模型

| 模型 | 用途 | 准备 |
|---|---|---|
| **Qwen3-8B** | 策略模型 | `hf download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B` |
| Qwen3.5-9B（已有） | BCP 和 LongMemEval 的本地评判；特权提示老师的备选 | 用 SGLang 或 vLLM 起 OpenAI 兼容服务 |

### 1.4 数据和基准

| 基准 | 什么时候需要 | 准备 |
|---|---|---|
| **T1 仓库**（合成） | 现在 | **不需要准备**：由 `envs/t1_warehouse.py` 按种子生成，依赖仓库里的 `state_study/probes/warehouse.py` |
| BrowseComp-Plus | 阶段 7 | ① 划分文件：`git submodule update --init third_party/methods/acm`，用其中的 `data/bcp_train_680.json`、`data/bcp_eval_150.json`；② 语料、BM25 索引、检索代码：按 ACM README 执行 `git clone https://huggingface.co/datasets/hltcoe/BrowseComp-Plus data/BrowseComp-Plus`，索引在 `indexes/bm25`；③ evidence / gold 文档标注（重标注要用）在 HF `Tevatron/browsecomp-plus`。查询可能是加密发布的，解密方法见 texttron/BrowseComp-Plus 仓库，需要核实 |
| LongMemEval-S | 阶段 8 | 数据按 `third_party/benchmarks/longmemeval` 的 README 下载（`git submodule update --init third_party/benchmarks/longmemeval`）；需要 `answer_session_ids` 字段 |

---

## 2. 实验顺序

下面统一设 `M=/models/Qwen3-8B`。产物放在：
- `runs/<日期>/data/`：数据；
- `runs/<日期>/ckpt/`：checkpoint；
- `runs/<日期>/logs/`：训练日志；
- `runs/<日期>/eval/`：评测结果。

每个阶段完成后，按 §3 的格式汇报。

### 阶段 0：环境检查（约 1 小时）

| 步骤 | 命令 | 通过标准 |
|---|---|---|
| 0.1 单元测试 | `python -m pytest -q self_paging/tests` | 35 个通过，1 个跳过 |
| 0.2 **模板比对** | `QWEN3_TOKENIZER=$M python -m pytest -q self_paging/tests/test_template_server.py` | 通过。不通过就把 diff 发回来，按 diff 改 `core/protocol.py` 的渲染函数，**不能跳过** |
| 0.3 oracle 健全性（真实分词器） | `python -m self_paging.eval.t1_eval --model $M --methods oracle --ratios 0.6 0.45 --seeds 100000 100020 --out eval/oracle.jsonl` | oracle 的 reward 全是 1.0 |
| 0.4 真实模型能跑通 | `python -m self_paging.eval.t1_eval --model $M --methods model --ratios 0.6 --seeds 100000 100002 --out eval/smoke.jsonl` | 能跑完、不报错；`invalid` 不全是满值 |
| 0.5 核对 config | 读 `$M/config.json` | 36 层、8 个 KV 头、head_dim 128。如果不同，改 `core/costs.py` 里的常数 |

### 阶段 1：T1 零样本和系统基线（E0、E1 的 T1 部分）

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python -m self_paging.eval.t1_eval --model $M \
    --methods full recent belady model --ratios 0.25 0.4 0.6 \
    --seeds $((100000 + i*25)) $((100025 + i*25)) --out eval/t1_zeroshot_part$i.jsonl &
done; wait
```

- 规模：200 个种子 × (full 1 档 + 3 种方法 × 3 档) = 2000 个 episode。HF 解码很慢，每个 episode 可能要几分钟，所以先用 50 个种子看趋势；
- 汇总：

```bash
python -c "import json,glob;from self_paging.eval.t1_eval import aggregate;rows=[json.loads(l) for f in glob.glob('eval/t1_zeroshot_part*.jsonl') for l in open(f)];print(json.dumps(aggregate(rows),indent=1))"
```

**要看的数字和判断**：

| 问题 | 看什么 | 判断 |
|---|---|---|
| E0：系统换页有没有提升空间 | 每个预算下 belady − recent | 25% 和 40% 预算下差 ≥ 5 个点，说明"决定留什么"有空间；差很小，就向用户报告，研究重心转向代价 |
| E1：零样本会不会用协议 | model 的 `invalid`、`kv_out`、`kv_in`、`notes`、`truncated_tokens`、reward | 非法调用率 ≤ 20%，并且有成功的换页 → 可以考虑跳过 SFT 直接 RL；否则做 SFT |
| 不设预算的上限 | full 的 reward | 如果 full 本身就很低，说明任务对 8B 太难，调低 `--n-decisions` 或者增加 gap（`T1Config`） |

### 阶段 2：造 T1 的 SFT 数据

```bash
# 模板理由（快，只用 CPU）
python -m self_paging.data.build_t1_sft --tokenizer $M --out data/t1_sft_template.jsonl \
  --seeds 0 1000 --ratios 1.0 0.6 0.45 0.35 --modes paging notes eager
# 真实模型写理由（推荐作为主数据，需要 GPU）
python -m self_paging.data.build_t1_sft --tokenizer $M --teacher $M --out data/t1_sft_teacher.jsonl \
  --seeds 0 1000 --ratios 1.0 0.6 0.45 0.35 --modes paging notes eager
```

- 脚本会打印每个"预算 / 模式"的保留数；
- 再造一份 held-out，用来算 SFT 的 eval loss：

```bash
python -m self_paging.data.build_t1_sft --tokenizer $M --out data/t1_sft_heldout.jsonl \
  --seeds 90000 90100 --ratios 1.0 0.6 0.45 --modes paging notes
```

**验收**：
- 保留的 episode ≥ 2000 条；
- 不换页（ratio 1.0）的样本占 15–30%；
- 统计 `meta.future_informed_evictions` 的占比。占比很高时，加 `--max-future-share 0.5` 再造一份做对照；
- 用特权提示老师时，`teacher_fallbacks` 占回合数的比例应该 < 30%；
- 按 `docs/data_and_training.md` §2.8 的表格重新统计一遍（用的是 Qwen3 分词器），重点看：各格的保留率、含 `kv_in` 的 episode 比例（eager 模式应该 > 50%）、依赖未来信息的换出占比。

### 阶段 3：SFT

```bash
torchrun --nproc_per_node 8 -m self_paging.train.sft --model $M --tokenizer $M \
  --data data/t1_sft_teacher.jsonl --eval-data data/t1_sft_heldout.jsonl \
  --out ckpt/sft --log logs/sft.jsonl --attn flex_attention --grad-ckpt --lr 5e-6 --epochs 2 --max-tokens 32768
# 单卡：去掉 torchrun，加 --lora-r 64 --lr 1e-4
```

多卡路径没在 GPU 上测过，第一次先用 2 卡、`--epochs 1`、少量数据试跑。

**验收**：
- `logs/sft.jsonl` 里训练 loss 下降，eval loss 不上升；
- 用 held-out 种子评测（阶段 1 的命令，把 `--model` 换成 `ckpt/sft`，`--methods model`）：
  - 非法调用率 ≤ 5%；
  - 各预算下的 reward ≥ 零样本的 model；
  - 40% 和 60% 预算下接近 belady。

### 阶段 4（可选）：拒绝采样，再做一轮 SFT

```bash
python -m self_paging.data.reject_sample --model ckpt/sft --tokenizer $M --out data/t1_reject.jsonl \
  --seeds 1000 1500 --ratios 0.25 0.4
```

把 `data/t1_reject.jsonl` 和阶段 2 的数据合在一起，再做 1 个 epoch 的 SFT。

### 阶段 5：GRPO（RL-1，只用 T1）

```bash
torchrun --nproc_per_node 8 -m self_paging.train.grpo --model ckpt/sft --tokenizer $M \
  --out ckpt/rl --log logs/rl.jsonl --save-every 20 \
  --steps 200 --prompts 16 --group 8 --lora-r 64 --lr 1e-5 --attn flex_attention --max-turns 30
```

- **先估算时间**：每步要跑 16 × 8 = 128 个 episode，HF rollout 每个可能要几分钟，8 卡并行时一步可能要几十分钟。先用 `--prompts 8 --steps 20` 试跑，测出每步耗时再定规模。太慢就先做开发任务 D8（批量 rollout）或 D6（SGLang）；
- **每 20 步评测一次**：对 `ckpt/rl/step_K` 跑阶段 1 的命令（`--methods model`，100 个 held-out 种子）。

**日志里要看的**（`logs/rl.jsonl`）：

| 字段 | 期望 | 异常时怎么办 |
|---|---|---|
| `reward`、`success` | 上升 | — |
| `groups_kept / groups` | 大于 0.3 | 太低说明题太难或太容易：调 `--n-decisions`，或者改 `GRPOConfig.ratios` |
| `truncated`、`forced_evictions` | 下降 | 上升说明模型学会了放任截断，见 data_and_training §3.8 |
| `kv_out`、`kv_in`、`notes` | 有变化，不归零 | 归零可能是效率项太强，试 `--w-eff 0.1` |
| `logprob_mismatch` | < 0.05 | 偏大说明 rollout 和训练两条路径数值不一致，先查 bf16 和注意力实现 |
| `invalid` | 很低 | — |

### 阶段 6：消融（各对应一个假设）

| 变体 | 命令差异 | 对应 |
|---|---|---|
| 不可逆（只能删） | `--irreversible` | K2：可逆性会不会改变策略 |
| 没有笔记工具 | `--no-notes` | K4：写笔记 vs 换页 |
| 没有效率项 | `--w-eff 0` | 效率通道的作用 |
| 跳过 SFT | `--model $M` | data_and_training §2.7 |
| LRU 兜底 | 需要开发任务 D7 | 模型 + 系统混合管理 |

每个变体用同样的步数和种子，评测和主设置一样。

### 阶段 7：BrowseComp-Plus（要先完成 D2，有条件时做 D3、D6）

1. 起评判服务，例如用 SGLang 起 Qwen3.5-9B，开 OpenAI 兼容接口；
2. **D2 写好后**，跑 E1 的 BCP 部分：Qwen3-8B 在训练 680 题上的 pass@8，开 YaRN，不设预算。如果 pass@8 > 0 的题少于 15%，按 experiment_plan §3.3 的备选方案处理，并向用户报告；
3. 评判校验：在 200 个答案上，把评判结果和 exact match 对比，不一致的抽 50 个人工看，报告一致率；
4. 评测 150 题：比较零样本、SFT、RL 和各基线。CLM 和 ACM 的零样本版本在 Qwen3-8B 上重跑；
5. 在 BCP 上做 RL 前，必须先完成 D6（SGLang），否则太慢。

### 阶段 8：LongMemEval-S（要先完成 D4）

用于零样本迁移评测，问题在最后给出，预算 8K / 16K。重点看可逆和不可逆的差别（K2），以及笔记和换回的使用（K4）。

### 阶段 9：E2 残留实验（要先完成 D1）

只需要推理。在 T1 的 `explicit` 场景上比较 FULL、OUT-LATE、OUT-EARLY、REWRITE 四个条件，剂量取 1K、4K、16K（algorithm_design §9）。

---

## 3. 每个阶段汇报什么

1. 跑了什么命令，用的哪个 commit；
2. 汇总表：每个方法 × 预算的 reward、cost、peak_resident、invalid、truncated；
3. 和验收标准对照：通过还是不通过；
4. 异常情况，以及原始日志的路径。

同一份结果追加到 `docs/research_log.md` 的时间线里。数字一律标明是哪个 checkpoint、多少个种子。

---

## 4. 剩下的开发任务（按优先级）

每个任务都要写测试，并保证 `self_paging/tests` 全部通过。

| 编号 | 任务 | 规格在哪 | 要点和验收 |
|---|---|---|---|
| **D1** | E2 残留实验脚本 `experiments/e2_residue.py` | algorithm_design §9 E2；data_and_training §1.1 | T1 的 `explicit` episode：页 a 是被更正的入库记录，页 b 是更正。FULL 什么都不换出；OUT-LATE 在决策前才换出 a；OUT-EARLY 在 b 到达后立刻换出 a；REWRITE 把 a 的 token 从日志里删掉、其余重新 prefill（新建一个 `PagedKV`，喂删掉后的 token 序列）。在决策点比较正确货架和旧货架编号的 log-prob，并统计贪心答案。剂量用 gap 控制。三个 OUT 和 REWRITE 条件的可见文本必须完全相同，要写测试检查 |
| **D2** | BCP 评测脚本 `eval/bcp_eval.py` | experiment_plan §3；`envs/t2_bcp.py` | 读划分文件；`PyseriniSearcher` + `OpenAIJudge`；`run_episode` 开思考模式（`EpisodeConfig(thinking=True)`）、YaRN 4、日志上限 120K（到上限就提示作答，需要在 episode 里加一个日志长度检查）；输出 JSONL，汇总 pass@k；再写一个评判校验脚本 |
| D3 | BCP 轨迹重标注（SFT 数据来源 B） | data_and_training §1.2、§2.3 | 原料是 Qwen3-8B 在完整上下文下答对的轨迹；需要信号用 evidence/gold 文档、显式引用和注意力；按 Belady 插入 `kv_out` / `kv_in`；反事实 ΔNLL 验证复用 `train/batch.py` 的 `token_logprobs`（同一条轨迹算两次，一次全驻留、一次按计划） |
| D4 | LongMemEval-S 环境 `envs/longmemeval.py` | experiment_plan §3.2；data_and_training §2.6（读会话时不能用证据信息） | 每个会话作为一页，每页之后模型有一个短回合，最后给出问题；用官方评判提示；再做一个"问题先给出"的变体 |
| D5 | QuestPage、AttnEvict 基线 `agent/pagers.py` | algorithm_design §6 | QuestPage：每页取各层各头 key 的均值作代表，用当前回合最后几个 token 的 query（或 key 作近似）打分，选最高的页直到填满预算，可逆；AttnEvict：按最近窗口对各页的累计注意力淘汰，不可逆。两者都从 `PagedKV.kv` 里读 |
| D6 | SGLang overlay（rollout 加速） | experiment_plan §7.2；参考本仓库 `suffix_cache_reuse/overlay.py` 的 monkeypatch 写法 | 会话私有的 KV 槽；`req_to_token` 只写驻留 token；位置偏移 `log_len − L_res`；新增 `/kv_set_resident` 接口。**一致性测试**：同一条轨迹，SGLang 返回的 logprob 和 `PagedKV` 的一致。最后作为 `rollout_fn` 接进 `train/grpo.py` |
| D7 | LRU 兜底变体 | data_and_training §3.7 | 给 `EpisodeConfig` 加 `overflow="truncate" \| "lru"`：超预算时由系统按 LRU 换出，而不是截断 |
| D8 | 批量 HF rollout（D6 之前的权宜之计） | — | 多条序列同时解码，每条有自己的驻留集；可以用左填充加注意力掩码。必须保持 `test_rollout_logprobs_match_training_pass` 这类一致性 |
| D9 | T1b 查找环境 `envs/t1b_lookup.py` | experiment_plan §3.1 | 实体记录逐页到达，按实体名提问；规则读者 + 最小充分集，和 T1 的接口一样；它是对相似度检索友好的对照 |

---

## 5. 默认超参

| 项 | 默认 | 在哪改 |
|---|---|---|
| T1 难度 | 4 个决策点，warmup 20，gap 5–25，每页 4–10 个事件 | `T1Config`；命令行 `--n-decisions` |
| 预算 | `t1_budget`：固定开销 + 比例 × 全部页文本长度 | `data/build_t1_sft.py` |
| SFT | 全参数 lr 5e-6，2 个 epoch，32K token 一个 micro-batch，flex_attention | `train/sft.py` 命令行 |
| GRPO | 每步 16 题 × 8，lr 1e-6（全参数）/ 1e-5（LoRA），clip 0.2 / 0.28，w_eff 0.25，TIS 上限 2，κ 取 batch 32，预算比例 {0.25, 0.4, 0.6}，温度 1.0 | `GRPOConfig` |
| 评测采样 | 温度 0.6（Qwen3 推荐 top_p 0.95、top_k 20） | `eval/t1_eval.py` |

---

## 6. 排错

| 现象 | 可能原因 |
|---|---|
| `AttributeError: 'DynamicCache' object has no attribute 'layers'` | transformers 版本太旧，装 5.23 |
| 模板比对失败 | 按 diff 改 `core/protocol.py`。常见差异：工具 JSON 的分隔符、system 结尾的换行、`<think>` 块的位置 |
| SFT 或 GRPO 显存不够 | 减小 `--max-tokens`；加 `--grad-ckpt`；改用 LoRA |
| GRPO 的 `groups_kept` 一直是 0 | 组内奖励全相同：题太难（全 0）或太容易（全 1）。调 `--n-decisions` 或预算比例 |
| `logprob_mismatch` 很大 | rollout（sdpa）和训练（flex）在 bf16 下数值有差：先用 `--attn sdpa` 训练对比，确认是不是注意力实现造成的 |
| 位置超过 40960 | 日志太长。T1 不该出现，调小 `T1Config`；BCP 要开 `--yarn 4` |
| episode 里大量 `truncated_tokens` | 预算比例太低，或者模型不会换出页。先看零样本的协议遵守情况 |
