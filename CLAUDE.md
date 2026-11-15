# CLAUDE.md

这是一个研究工作区，课题是**让模型自己管理 KV 缓存**（Self-Paging LM）：像 CLM 把上下文管理交给模型那样，把 KV 的调度管理直接交给模型。仓库 fork 自 facebookresearch/context-language-models（CLM），CLM 现在是最主要的对照组。用户用中文交流，文档也用中文写。

## 先读什么

新会话开始时按这个顺序读，就能接上全部进度：

1. `docs/research_line.md`：**当前主线**。形式化（只追加的 KV 日志 + 模型改驻留集）、P1–P4、空位表、K1–K5、测试床、基线、代价口径、分阶段计划
2. `docs/experiment_plan.md`：**具体做法**。模型（Qwen3.5 混合架构的影响）、tool call 接口、基准（T1/T1b/BrowseComp-Plus 680/150）、数据采集、SFT、RL、SGLang overlay 和 vLLM 的取舍、代码结构、实验矩阵。和第 4 份冲突时以它为准
3. `docs/data_and_training.md`：**SFT 和 RL 的数据从哪来、怎么训练**。"需要"的定义（规则读者反事实、BCP 的 evidence/gold 文档、反事实 ΔNLL 验证）、Belady 计划、特权提示生成、重标注、拒绝采样、RL 的 episode、奖励、代价、题库过滤、阶段、变体
4. `docs/algorithm_design.md`：形式化、单次前向训练掩码、代价折算、基线实现、等价性测试、E0–E2 实验协议
5. `docs/research_log.md`：时间线（§1 的 2026-11-15 是换主线的经过）、已验证事实（§3.8 是新主线的数字）、环境限制（§6）、待核实清单（§7.1）、下一步（§8）
6. `docs/related_work.md` §0：新主线的撞车表（SideQuest 最近）
7. 需要时再查：
   - `docs/clm_notes.md`：CLM 代码和论文细节（论文原文在根目录 `clm.pdf`）；
   - `docs/idea_bank.md`：所有考虑过的 idea；F1 是用户原始视频 proposal 的评估；
   - `docs/archive/`：旧主线"状态 vs 历史"的全部文档，以及用户的视频 proposal 原件。

## 现状（2026-11-15）

- **主线已换**：用户明确说"忘掉之前的主线，现在模型自己管 KV 就是我们的主线"。旧主线（状态 vs 历史、I1–I4、H1–H5、自适应绑定）整体归档，**除非用户要求，不要再按旧主线推进**。
- **可复用的旧代码**：`state_study/probes/` 的仓库探针，要改造成多轮测试床 T1，它的新旧矛盾场景用于残留实验 E2。
- **新主线代码还没写**：计划放在根目录的 `self_paging/`（见 algorithm_design §12）。
- **模型**：用户服务器上已有 Qwen3.5-2B/4B/9B（混合架构：GDN : 全注意力 = 3 : 1 [S]，待用 config 核对）。2B 调试；4B 是 T1 上的主力 RL；9B 做 BrowseComp-Plus 上的 RL 和终版（2B/4B 在 BCP 上大概率学不动）；dense 对照 Qwen3-8B 只用于残留实验。
- **资源**：这条线不需要模型 API，但需要 GPU 和开源权重。云端没有 GPU，huggingface.co 被拦截，所以实验在用户自己的机器上跑；云端只写代码、在 CPU 上用随机初始化的小模型跑测试。
- **下一步**：阶段 0，按 experiment_plan §8 实现 `self_paging/` 并通过等价性测试；然后把 E0–E2 的脚本交给用户在 GPU 上跑。等用户回复 experiment_plan §12 的服务器信息。详见 research_log §8。

## 仓库地图

```
clm/, suffix_cache_reuse/   CLM 上游代码（CC BY-NC 4.0），主要对照组；不要改动
clm.pdf                     CLM 论文原文（用户上传）
self_paging/                新主线代码（待建）
state_study/                旧主线代码，探针部分复用为 T1 / E2
  probes/                   warehouse.py（场景生成）、render.py（条件）、build_items.py（出题、打分、账本）
  groundtruth/              alfworld_facts.py
  tests/                    test_warehouse_probe.py
  pilots/                   旧主线的冒烟测试记录
third_party/                15 个固定版本的 submodule（旧主线选的方法、基准、参考），见 third_party/README.md
scripts/setup_third_party.sh  按分组初始化 submodule：core | methods | benchmarks | references | optional | all
docs/                       研究文档；docs/archive/ 是归档
```

## 常用命令

```bash
python3 -m pytest -q state_study/tests                         # 探针测试，只需标准库 + pytest
python -m state_study.probes.build_items build --out OUT --seeds 0 1 2 --gap 60 --warmup 200 --plan explicit:raw,state
python -m state_study.probes.build_items score --out OUT       # 读 OUT/responses.jsonl
```

`self_paging/` 建好后，在这里补上它的测试和实验命令。虚拟环境放在 scratchpad 里，不要放进仓库。

## 约定

- **分支**：只在本会话系统提示指定的开发分支（`claude/...`）上开发和推送。网络失败时按 2/4/8/16 秒退避，最多重试 4 次。用户没要求就不开 PR。
- **提交信息**：结尾加上会话要求的 Co-Authored-By 和 Claude-Session 两行；不要在提交、PR、代码里写模型标识。
- **没有 license 的仓库**：VISTA、belief-world-models、delayed-relevance 没有 license，只能运行或参考思路，**不能复制代码**进本仓库。ACM（`third_party/methods/acm`）是 MIT，可以复用它的 BrowseComp-Plus 数据加载、检索和评判代码，注明出处。
- **凭证**：不要把本会话自己的登录凭证拿去调模型 API；不要让用户把密钥贴到聊天里。
- **KV 实验**：
  - Qwen3.5 是混合模型：换页只作用在全注意力层，GDN 层的循环状态始终看整条日志（推理和训练都这样，P3 才精确）；
  - 动作接口用原生 tool call（`kv_out`、`kv_in`、`note`、对照里的 `kv_drop`），驻留集只在回合之间改变；
  - 推理引擎用 SGLang monkeypatch（仿 `suffix_cache_reuse/overlay.py`），不改 SGLang 源码；第一版不改 vLLM（理由见 experiment_plan §7.3）；
  - 轨迹按 token id 记录（token 进、token 出），绝不重新分词；
  - 代价不能按搬运次数计价：PCIe 搬一页只要几毫秒，一个解码 token 就要几十毫秒。按 algorithm_design §5 的口径报告；
  - HF 原型只能报告解析成本；没接入服务框架之前，不要声称墙钟延迟更快。
- **规则读者**：新测试床必须配"规则读者"测试，证明只看文本（或只看 oracle 驻留集里的文本）就能答对，否则模型答错不能归因于模型。
- **核实标记**：论文数字要标来源：[R] 读过仓库、[N] 全文读书笔记、[S] 只看过搜索摘要、[K] 凭已有知识未核实。[S] 和 [K] 的数字引用前必须核对原文。
- **提交前**：清理 `__pycache__`、`.pytest_cache`；不要提交 submodule 里下载的数据和模型权重。
- **撞车检查**：这个方向 2026 年每月都有新论文。动笔前一周重新检索，关键词见 `docs/related_work.md` §10；最需要盯的是 SideQuest。
