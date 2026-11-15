# 归档：旧主线"状态 vs 历史"（2026-10-02 至 2026-11-15）

2026-11-15 用户决定换主线：**让模型自己管理 KV 缓存**（见 [`docs/research_line.md`](../../research_line.md)）。旧主线的四份文档原样移到这里，只作历史记录：

| 文件 | 内容 |
|---|---|
| [`research_line.md`](research_line.md) | 旧主线"写入时付代价，还是读取时付代价？"：误差分解、H1–H5、自适应绑定 |
| [`algorithm_design.md`](algorithm_design.md) | 探针、11 种条件、错误账本、真值、实验方案 |
| [`proposal.md`](proposal.md) | I1–I4 原始提案 |
| [`literature_map.md`](literature_map.md) | 状态派文献地图 |

仍在用的部分：
- `state_study/probes/` 的仓库探针，改造成新主线的合成测试床 T1；
- 新旧矛盾（原 I3）的场景，用于新主线的残留实验 E2；
- `docs/related_work.md` §1–§9、`docs/research_log.md` §2–§5 里的事实和教训仍然有效，只是不再围绕旧主线组织。

同目录上一级的 [`../video_kv_paging_proposal_2026-11-15.pdf`](../video_kv_paging_proposal_2026-11-15.pdf) 是用户 2026-11-15 上传的视频 KV 换页 proposal，它是新主线的起点，评估结论见 [`docs/idea_bank.md`](../../idea_bank.md) F1。
