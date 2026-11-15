# self_paging

新主线"让模型自己管理 KV 缓存"的代码。设计见 `docs/research_line.md`、`docs/experiment_plan.md`、`docs/data_and_training.md`、`docs/algorithm_design.md`。

## 已实现（阶段 0 的核心）

| 文件 | 内容 |
|---|---|
| `core/pages.py` | 页表：页、驻留集版本、`page_out` / `page_in` / `drop`、钉住页、预算检查 |
| `hf/logstore.py` | `PagedKV`：在 HF 解码器上实现自换页。每次前向把驻留页的 KV 按日志顺序拼成 cache，新 token 用全局位置；逻辑换页和物理换页（真的挪到内存）两种模式 |
| `hf/masks.py` | 单次前向训练用的掩码：稠密可见性矩阵、HF 用的加性 4D 掩码、FlexAttention 的 `mask_mod` |
| `tests/test_equivalence.py` | 在 Qwen3 的真实模型结构上（随机初始化的小配置，CPU）验证 P1–P3 |

## 测试

需要 torch 和 transformers（Qwen3 支持）。云端的做法是在 scratchpad 里建虚拟环境：

```bash
uv venv -p 3.11 <scratchpad>/venv-sp && . <scratchpad>/venv-sp/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install numpy pytest transformers
python -m pytest -q -p no:cacheprovider self_paging/tests
```

2026-11-15 在 torch 2.15.0+cpu、transformers 5.23.0 上全部通过（13 个）：

- 逐步换页推理 = 整条日志加掩码的一次前向：sdpa、float64 下误差 < 1e-9；eager 因为 HF 在 float32 里算 softmax，误差约 1e-7；
- 去掉驻留集掩码后结果明显不同（测试本身能失败）；
- 物理换页和逻辑换页逐位相同；换出再换回的 KV 逐位相同；
- FlexAttention 的 `mask_mod` 和稠密可见性矩阵一致；HF 自带的 `flex_attention` 实现直接接收我们的 BlockMask，训练不需要改模型代码；
- 非法动作（换出钉住页、换出正在写的页、换回不在内存里的页、超预算）都会报错。

## 下一步

见 `docs/experiment_plan.md` §8：协议和渲染（Qwen3 chat template、tool call）、HF agent 循环、T1 环境、系统基线和 Belady、E0–E2 脚本，然后是 SGLang overlay。
