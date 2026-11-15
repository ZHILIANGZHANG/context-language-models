# self_paging

新主线"让模型自己管理 KV 缓存"的代码。设计见 `docs/research_line.md`、`docs/experiment_plan.md`（v2，Qwen3-8B）、`docs/data_and_training.md`、`docs/algorithm_design.md`。

## 目录

| 路径 | 内容 | 测试状态 |
|---|---|---|
| `core/pages.py` | 页表：页、驻留集版本、`page_out` / `page_in` / `drop`、钉住页、预算 | CPU 已测 |
| `core/log.py` | `TokenLog`：不带模型的记账（造 SFT 数据用），导出轨迹记录 | CPU 已测 |
| `core/protocol.py` | Qwen3 格式的渲染（system+工具、工具结果块、页头、仪表盘、转义）、tool call 解析、分词器适配 | CPU 已测；**和官方模板的逐 token 比对要在服务器上跑** |
| `core/costs.py` | 每个 token 的可见 key 数、代价 C（两种 κ 折算） | CPU 已测 |
| `hf/logstore.py` | `PagedKV`：参考推理引擎，逻辑 / 物理换页，驻留集不变时复用 cache | CPU 已测 |
| `hf/masks.py` | 可见性矩阵、HF 加性掩码、FlexAttention `mask_mod` | CPU 已测 |
| `envs/t1_warehouse.py` | T1 多轮仓库环境：事件流、分页、决策、规则读者、最小充分集 | CPU 已测 |
| `envs/t2_bcp.py` | BrowseComp-Plus 环境：search / get_document / answer；pyserini BM25；OpenAI 兼容接口的本地评判（模板逐字取自 ACM，MIT） | 逻辑用假检索器测过；**真实语料、索引、评判要在服务器上跑** |
| `oracle/t1_oracle.py` | T1 oracle：按大小加权的在线 Belady，换页模式 / 笔记模式 / 混合；每次换出标注"可观察"还是"依赖未来信息" | CPU 已测 |
| `oracle/hinted_teacher.py` | 特权提示老师：动作来自 oracle，理由由真实模型写；泄漏提示就退回模板 | CPU 已测（随机小模型） |
| `agent/episode.py` | episode 循环（数据、RL、评测共用）；`ModelPolicy`（采样，记录 log-prob）、`ScriptedPolicy`（强制文本） | CPU 已测 |
| `agent/pagers.py` | 系统换页基线：Recent、Belady（T1 上界） | CPU 已测 |
| `data/build_t1_sft.py` | T1 SFT 数据：同题多预算 × 换页 / 笔记模式，质量门槛和可实现性过滤 | CPU 已测 |
| `data/reject_sample.py` | 预算压力下的拒绝采样 | 命令行跑通（随机小模型） |
| `train/batch.py` | 打包多条轨迹、全局页号和版本号、稠密掩码或 BlockMask、分块 checkpoint 的 log-prob | CPU 已测 |
| `train/sft.py` | SFT：单卡；torchrun 下全参数用 FSDP2，LoRA 用梯度 all-reduce | 单进程 CPU 已测；**多卡路径没测过** |
| `train/advantage.py` | GRPO 组内优势 + CLM 双通道效率项 + 动态采样 | CPU 已测 |
| `train/grpo.py` | GRPO：用 `PagedKV` rollout、token 级裁剪损失、TIS；torchrun 下每卡一份副本（8B 用 LoRA），梯度 all-reduce | 单进程 CPU 已测；**多卡路径没测过** |
| `eval/t1_eval.py` | T1 评测：model / full / recent / belady / oracle | 命令行跑通 |

还没写的：LongMemEval-S 环境；T2 的轨迹重标注（来源 B，需要真实轨迹）；QuestPage / AttnEvict 基线（需要模型内部状态）；SGLang overlay（rollout 加速，需要 GPU 开发）。

## 测试

```bash
uv venv -p 3.11 <scratchpad>/venv-sp && . <scratchpad>/venv-sp/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch     # 服务器上装 CUDA 版
uv pip install numpy pytest transformers
python -m pytest -q -p no:cacheprovider self_paging/tests
QWEN3_TOKENIZER=/path/to/Qwen3-8B python -m pytest -q self_paging/tests/test_template_server.py   # 服务器上
```

2026-11-15，torch 2.15.0+cpu、transformers 5.23.0：34 个测试通过，1 个跳过（模板比对，需要真实分词器）。关键的几条：

- **推理和训练一致**：逐步换页推理 = 整条日志加掩码的一次前向（sdpa、float64 下误差 < 1e-9）；
- **RL 一致**：通过 `PagedKV` rollout 记录的 log-prob，等于训练端打包后一次前向算出的 log-prob。脚本策略（有换页）和采样策略都测过；
- 物理换页和逻辑换页逐位相同；HF 的 `flex_attention` 直接接收我们的 BlockMask；
- T1：规则读者只看文本就能答对；最小充分集既充分又极小；oracle 在 60% 预算下也能全对；Belady 换页不比 Recent 差；
- SFT 能在 oracle 数据上过拟合；GRPO 一步之后，正优势轨迹的 log-prob 上升、负优势的下降。

## 在服务器上怎么跑（Qwen3-8B）

```bash
M=/path/to/Qwen3-8B
```

1. 检查模板（必须先过）：

```bash
QWEN3_TOKENIZER=$M python -m pytest -q self_paging/tests/test_template_server.py
```

2. 造 T1 的 SFT 数据（oracle + 模板理由；不需要 GPU）：

```bash
python -m self_paging.data.build_t1_sft --tokenizer $M --out data/t1_sft.jsonl --seeds 0 1000 --ratios 1.0 0.6 0.45 --modes paging notes
```

3. 零样本评测 + 系统基线（E0、E1 的 T1 部分）：

```bash
python -m self_paging.eval.t1_eval --model $M --methods full recent belady model --ratios 0.25 0.4 0.6 --seeds 100000 100200 --out eval/t1_zeroshot.jsonl
```

4. SFT（8 卡全参数；单卡就加 `--lora-r 64`）：

```bash
torchrun --nproc_per_node 8 -m self_paging.train.sft --model $M --tokenizer $M --data data/t1_sft.jsonl --out ckpt/sft --log logs/sft.jsonl --attn flex_attention --grad-ckpt --lr 5e-6 --epochs 2
```

5. 拒绝采样（可选，第二轮 SFT 数据）：

```bash
python -m self_paging.data.reject_sample --model ckpt/sft --tokenizer $M --out data/t1_reject.jsonl --seeds 1000 1500
```

6. GRPO（每卡一份副本，LoRA）：

```bash
torchrun --nproc_per_node 8 -m self_paging.train.grpo --model ckpt/sft --tokenizer $M --out ckpt/rl --log logs/rl.jsonl --steps 200 --prompts 16 --group 8 --lora-r 64 --lr 1e-5 --attn flex_attention
```

7. 评测 RL 后的模型：

```bash
python -m self_paging.eval.t1_eval --model ckpt/rl --tokenizer $M --methods model --ratios 0.25 0.4 0.6 --seeds 100000 100200 --out eval/t1_rl.jsonl
```

注意：
- `--model` 可以直接传 LoRA 目录：`load_model` 会读 `adapter_config.json`，加载底座再挂上 adapter（需要装 peft）；
- HF 的 `PagedKV` rollout 是逐 token 解码，一次只跑一条序列，**慢**。T1 规模可以接受，BCP 规模要等 SGLang overlay；
- 多卡路径（FSDP2、梯度 all-reduce）是按标准写法写的，但没在 GPU 上跑过。第一次请先用 2 卡、少量步数试。
