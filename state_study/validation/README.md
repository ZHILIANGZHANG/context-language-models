# 2026-10-03 验证轮的代码

结论和解读见 [docs/validation_2026-10-03.md](../../docs/validation_2026-10-03.md)，记录见 [state_study/pilots/2026-10-03_validation](../pilots/2026-10-03_validation/)。

## 文件

| 文件 | 内容 | 来源 |
|---|---|---|
| `vdepth.py` | 版本深度探针：决策相关货架的状态翻转 f 次，事件数、token 数、答案都不随 f 变 | 自写 |
| `retro.py` | 事先更正 vs 事后更正，可选在更正前插入 THOUGHT 行 | 自写 |
| `echo.py` | 更正之后再跟 k 步"仿佛更正没发生"的 agent 回答 | 自写 |
| `ig.py` | 整合门控探针：两次调用的闭环（通知 → 下一托入库），历史与状态两种表示，说明书是否覆盖、是否要求只看规定字段、是否加整合提示 | 自写 |
| `*_build.py`、`*_score.py`、`ig_build.py`、`ig_score.py`、`rec.py` | 上面四个探针的盲化出题、打分、记录 | 自写 |
| `dr_common.py` | 读 delayed-relevance 轨迹，按它的 ReActRuntime 格式拼 prompt | 自写；运行时导入对方代码 |
| `dr_coherence.py` | 检查 delayed-relevance L2 更正通知与历史是否一致 | 同上 |
| `dr_l2.py` | L2 反事实重放（ORIG / COH / INT） | 同上 |
| `dr_l1.py` | L1 重放（BASE / CHECK / JIT 一次调用；PIN / FACT 两次调用） | 同上 |
| `dr_tables.py` | 从原轨迹重新统计 L1、L2 的表格，以及自由字段的"到达时是否写下"分解 | 同上 |
| `clm_cost_model.py` | 示意性成本模型：不管理、定时摘要、CLM 两种压缩日程，在 FLOPs、API 缓存计费、CLM 本地美元估计下的相对成本（不是测量） | 自写 |
| `dr_persistence.py`、`dr_adoption.py` | 显式状态闭环里信念偏差的持续时间、被拒后的采纳率、错误成串程度 | 同上 |

delayed-relevance **没有 license**：这些脚本只在运行时从 `third_party/references/delayed-relevance/src` 导入它的环境，读取它公开分支上的轨迹，从不复制它的代码或数据。脚本设置了 `sys.dont_write_bytecode`，不会在 submodule 里留下 `__pycache__`。

## 准备

```bash
# 轨迹：delayed-relevance 的 runs/table1-gemini-3-flash-preview-vertex 分支（2026-10-03 时为 f40247344a，
# results/ 下有全部 L1、L2 和 adj_* 轨迹）。只读，放在 scratchpad，不要放进仓库
mkdir -p $SCRATCH/drtraces
git -C third_party/references/delayed-relevance fetch --depth 1 origin runs/table1-gemini-3-flash-preview-vertex
git -C third_party/references/delayed-relevance archive FETCH_HEAD results/ | tar -x -C $SCRATCH/drtraces
T=$SCRATCH/drtraces/results
# 依赖：dr 包需要 anthropic、google-genai 等（见 research_log §6.3 的 venv-dr）
```

## 命令

```bash
cd state_study/validation
python -B dr_coherence.py                                  # 40 个种子里 4 个一致；L2 用过的 6 个全部矛盾
python -B dr_tables.py $T                                  # §2.2、§2.4、§2.5 的表
python -B dr_persistence.py $SCRATCH/drtraces              # §2.8
python -B dr_adoption.py $SCRATCH/drtraces

# L2 重放：建第 10 步 prompt → 子 agent 作答写到 OUT/outA/<id>.txt → 建第 11 步 → 作答写到 OUT/outB → 打分
python -B dr_l2.py build-a --traces $T --out OUT
python -B dr_l2.py build-b --traces $T --out OUT
python -B dr_l2.py score --out OUT

# L1 重放
python -B dr_l1.py build   --traces $T --out OUT           # BASE/CHECK/JIT，作答写到 OUT/outA
python -B dr_l1.py build-w --traces $T --out OUT           # PIN/FACT 通知那一步，作答写到 OUT/outW
python -B dr_l1.py build-d --traces $T --out OUT           # 决策那一步，作答写到 OUT/outD
python -B dr_l1.py score --out OUT

# 整合门控探针
python -B ig_build.py a OUT && python -B ig_build.py b OUT && python -B ig_score.py OUT

# 版本深度 / 事后更正 / 自我复述（SALT 任意；盲化映射存在 OUT/blind_map.json）
python -B vdepth_build.py OUT SALT && python -B vdepth_score.py OUT
```

子 agent 的作答协议（每题一个子 agent，Haiku，只许 Read 题目文件和 Write 一次答案）见 [docs/research_log.md](../../docs/research_log.md) §5。以上脚本在 2026-10-03 的 scratchpad 运行结果上做过回归：重新生成的 prompt 与当时逐字一致，分数一致。

规则读者测试：`python -m pytest -q state_study/tests/test_validation_probes.py`。
