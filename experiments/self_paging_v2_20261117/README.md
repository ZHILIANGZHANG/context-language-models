# SFT v2 与已完成测评结果归档（2026-11-17）

本次发布包含实际运行代码、已完成训练、已写出测评结果及失败/停止记录。主快照截至北京时间 **2026-11-17 19:05:37**。服务器 LongBench 测评仍在继续，本次仅归档其 **92/180** 条完成结果，不能视为全套最终结果。文件逐个读取，并非所有文件在同一时刻原子捕获。

## 训练与验收

- 起点为原始 Qwen3-8B；SFT v2 使用 2,400 条训练轨迹、240 条独立验证轨迹，数据切分、真实换入/换出覆盖、拒收条件和 SHA 见 `training/data/`。
- LoRA rank 64、alpha 128、dropout 0，AdamW，学习率 1e-4，1 epoch，3 卡；实际完成 **516 个优化步**，每 50 步验证并保存。完整配置见 `training/training-plan.json`。
- 最终 240 条验证 loss **0.0454599692**；500 步为 **0.0454410422**。这是监督 token loss，不是任务准确率。完整曲线见 `tables/sft_validation.csv` 和 `training/sft.jsonl`。
- 训练退出码 0，完成记录及 checkpoint 元数据已归档。权重文件仍在服务器，未提交大型 safetensors 或训练语料全文；数据来源和散列已保存。
- [SwanLab 训练记录](https://swanlab.cn/@zhangcheng525/context-language-models/runs/52iv3vyl)，保留云端回读证据；访问受项目权限控制。未发布鉴权凭据。

## LongBench：同一批 17 题的结果

36 题选样清单固定，每个子集 6 题。当前五种方法均完成的共同样本是 **17 题**，所有方法使用同一份最终 SFT adapter。官方整篇提示词和分页工具协议不同，单列展示。

| 配置 | 共同 17 题逐题平均得分（0–100） |
|---|---:|
| 官方提示词，整篇输入，无 KV 限制 | 56.82 |
| 分页读取，无 KV 限制 | 43.89 |
| SFT 自主换页，峰值预算 4096 KV 槽 | 23.06 |
| StreamingLLM 移植版，峰值预算 4096 KV 槽 | 9.73 |
| H2O 移植版，峰值预算 4096 KV 槽 | 23.14 |

得分使用各任务官方指标，包括 F1、ROUGE、分类/检索准确率、代码相似度；汇总为逐题均值，不应称为统一准确率或 LongBench 全量榜单成绩。各子集题数尚不相等。逐题、分任务和共同样本 ID 见 `tables/`。先前未加计时的 80 条结果另存 `raw/longbench-v1/`，不与这次重跑拼接；最终题标记修复前的 1 条旧结果保存在 `raw/longbench-v1-before-final-marker/`。

### 自主换页轨迹核查

19 道已完成的 SFT 自主换页题：主动换出尝试 **30 次**，成功 **18 次**，失败 **12 次**；成功换出 **43 页**，其中模型输出 30 页、文档工具返回 10 页、笔记 3 页。成功换入 **0 次**。另有系统兜底强制换出 **121 次**，输入截断 **18,315 tokens**。

因此尚不能据此宣称在真实长文本上学会可靠自主换页。失败调用与成功换页必须分开，系统兜底不计为模型学会的操作。完整逐题统计见 `tables/sft_kv_out_audit.csv`。

### 计时与基线实现的限制

StreamingLLM/H2O 按作者保留规则移植到 Qwen3 QK-normalized GQA；不是未经修改的作者原始程序或优化后的速度实现。StreamingLLM 保留 attention sink+recent 并执行位置重排；H2O 累积注意力分数并保留 heavy+recent。GQA 组内平均属于明确的移植选择。

为限制包括新 token 在内的峰值槽数为 B，当前实现只保留 B−1 个槽。缓存满后，新文档按单 token 整模型前向推进，这是当前长文档预填充慢的主要实现因素。H2O 使用 SDPA 产生输出，并额外计算注意力概率用于评分。不能用这些耗时宣称优于官方优化实现几倍。SFT 还存在输入截断，与基线的实际处理量不同。

每题记录端到端、文档读入、生成、换页与缓存重建时间；模型加载、外部打分、序列化不计入推理时间。缓存重建与部分阶段重叠，不能直接将所有阶段求和。GPU 共用监控记录保存在 `control/longbench-v1/latency/`，发生竞争的行不用于独占速度比较。具体计时定义见归档的 LongBench README 和 `self_paging/eval/latency.py`。

## 其他已完成与历史结果

- `raw/faithful-baselines/`：修正基线后的 **112 条 T1 评测**，其中 Full 11、SFT 自主换页 55、StreamingLLM 26、H2O 20。用户切换到 LongBench 后停止，原计划 800 条未完成，不把不同覆盖的均值当同题比较。
- `control/post-sft-eval/`：早期 SFT v2 测评的 **298 条 summary 记录**及原始轨迹；其中旧系统基线协议存在提示词要求已禁用 KV 工具等问题，已标记 superseded。只作历史追溯，不作为修正后基线的最终成绩。
- `control/token-baselines-9c7dbd5/`：旧基线的预检失败、停止和协议审计记录。不能将预检或排队状态报告为成功评测。
- `historical-v1/`：此前 SFT v1/原始权重对照已落盘的轨迹补充快照。与 v2 的训练数据、提示词和执行协议不同，不能直接合并或据此声称压缩算法优越性。仓库先前的 `self_paging_20261116` 和 `self_paging_baselines_partial_20261116` 保留原记录；本次归档的协议审计限制也适用于引用这些旧对照。

## 复核与代码

在仓库根目录执行：

```bash
python3 experiments/self_paging_v2_20261117/summarize.py
```

脚本核验主清单 1,173 个捕获文件的 SHA、285 对测评结果与轨迹 SHA、样本唯一性以及成功换出计数，重建表格。原始大轨迹/JSONL 使用 gzip，解压后与服务器捕获内容一致；JSONL 仅包含完整可解析行。`manifest.json` 保存远端路径、捕获散列与压缩产物散列；`historical-manifest.json` 单独记录后补的 v1 历史快照。

`source/sft-v2/` 与 `source/longbench-latency/` 保留训练和计时评测时的源码。仓库当前修改的 8 个 Python 文件与正在运行的计时源码逐字节一致，见 `source-parity.json`。发布前重新执行相关基线 CPU 测试 **11 passed**，历史完整相关 CPU 套件 **60 passed、2 个无关训练测试 deselected**，CUDA 无淘汰 logits 一致与真实淘汰探针证据保留。验证通过不代表算法优越性或完整实验结束。

本次未启动 RL，未重跑已完成实验。运行中的测评继续，后续新增结果需另行归档。
