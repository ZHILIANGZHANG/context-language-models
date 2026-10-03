# Third-party code and benchmarks

Everything here is a git submodule pinned to a fixed commit. Nothing is vendored: several
upstream repositories have no license, some are large, and pinning keeps every experiment
tied to an exact upstream version. Initialize what you need with
[`scripts/setup_third_party.sh`](../scripts/setup_third_party.sh):

```bash
scripts/setup_third_party.sh core      # the minimum for the first experiments
scripts/setup_third_party.sh all       # everything (BEAM alone is ~4 GB)
```

Pins were taken on 2026-10-03. Licenses were read from each repository's LICENSE file or
README at that commit.

## Methods (comparison arms and baselines)

| Path | Upstream | Pinned commit | License | Role in this study |
|---|---|---|---|---|
| `methods/pos` | [luoyu100/PoS](https://github.com/luoyu100/PoS) | `d6acb43` | MIT | **Experiment backbone.** Shared ReAct loop with a pluggable `ContextProvider` (`reset / update / get_context / get_result`), adapters for ALFWorld, LOCA-Bench, RCA-100 and ClinDiag. Also the PoS belief-state arm. Paper: arXiv 2610.01415 |
| `methods/skill-state-runtime` | [WUWeifeng710/skill-state-runtime](https://github.com/WUWeifeng710/skill-state-runtime) | `84b9804` | MIT | Unofficial re-implementation of SKILL.state (arXiv 2608.26263): schema-bound JSON state updated by patches, history discarded. The official code and SkillExecBench are not public |
| `methods/scroll` | [niceIrene/QwenPaw](https://github.com/niceIrene/QwenPaw/tree/scroll-research) (branch `scroll-research`) | `3db60c5` | Apache-2.0 | Scroll (arXiv 2608.21690): append-only event log + persistent Python kernel. Method code only, under `src/qwenpaw/agents/context/scroll/`; its benchmark adapters live in a private repo |
| `methods/vista` | [binyxu/VISTA](https://github.com/binyxu/VISTA) | `647a93d` | **none yet** | VISTA (arXiv 2606.30005): typed blocks + usage dashboard + recoverable archive. Has LOCA/BrowseComp-Plus/GAIA adapters. Run as a baseline only; ask the authors before reusing code |
| `methods/rlm` | [alexzhang13/rlm](https://github.com/alexzhang13/rlm) | `d04208a` | MIT | Recursive Language Models (arXiv 2512.24601): input held as a REPL variable |
| `methods/selfcompact` | [tianjianl/selfcompact](https://github.com/tianjianl/selfcompact) | `1295817` | MIT | SelfCompact (arXiv 2606.23525): rubric-gated, model-triggered compaction; a CLM baseline |
| `methods/acm` | [lixiaochuan2020/agentic-context-management](https://github.com/lixiaochuan2020/agentic-context-management) | `f06f90e` | MIT | ACM (arXiv 2607.23809): summarize/offload/query memory tools. The untrained `memtool` mode runs on API models; the paper's gains come from post-training |
| `methods/belief-world-models` | [skumar-ml/belief-world-models](https://github.com/skumar-ml/belief-world-models) (branch `master`) | `ac5bf2b` | **none** | Belief-Based World Models (arXiv 2609.00455). Reference; ALFWorld/ScienceWorld/BabyAI harness |

The CLM method itself is the code in [`../clm`](../clm) (CC BY-NC 4.0).

## Benchmarks

| Path | Upstream | Pinned commit | License | Role in this study |
|---|---|---|---|---|
| `benchmarks/loca-bench` | [hkust-nlp/LOCA-bench](https://github.com/hkust-nlp/LOCA-bench) | `8b6fac4` | MIT | Main shared battleground of the state-centric methods (PoS, Scroll and VISTA all report on it). Data generated locally; 15 task families × 5 seeds per length, 8K–256K. Some families ship a `groundtruth_workspace`. PoS vendors the same commit |
| `benchmarks/alfworld` | [alfworld/alfworld](https://github.com/alfworld/alfworld) | `aaba687` | MIT | Predictable-needs side; cheap; PDDL world facts give true state. PoS ships `raw`/`pos` configs (134 valid-unseen games) |
| `benchmarks/longmemeval` | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | `9e0b455` | MIT | Unpredictable-needs side (question asked after ingestion). Scroll reports on LongMemEval-S. **Data is on Hugging Face / Google Drive, not in the repo; download it yourself** |
| `benchmarks/supersede` | [Vrin-cloud/supersede](https://github.com/Vrin-cloud/supersede) | `677993d` | Apache-2.0 | LongMemEval knowledge-update (78 questions) with `full_context` vs bounded-notes rollouts; published full-context vs notes gaps to compare against |
| `benchmarks/state-bench` | [microsoft/STATE-Bench](https://github.com/microsoft/STATE-Bench) | `5644b18` | MIT | Optional. 450 tau-style tasks with final-DB-state checks; locked user simulator and judge |
| `benchmarks/beam` | [mohammadtavakoli78/BEAM](https://github.com/mohammadtavakoli78/BEAM) | `b2da22e` | MIT | Optional, **~4 GB**. 128K–10M-token chats; use the 128K/1M tiers only |

## References (read-only)

| Path | Upstream | Pinned commit | License | Why it is here |
|---|---|---|---|---|
| `references/delayed-relevance` | [JaviMaligno/delayed-relevance](https://github.com/JaviMaligno/delayed-relevance) | `53ceaf4` | **none** | Careful independent replication of SKILL.state with delayed-relevance and retroactive-invalidation probes and a cached-cost analysis. The design source for Idea 3. **No license: read it, do not copy code**; our warehouse environment is a fresh re-implementation from the SKILL.state paper |

## Not included, and why

- **Agent-BRACE, CompactionRL, From History to State, Context-as-a-Tool, ContextPilot**: their
  methods need RL training or self-hosted RL checkpoints.
- **InfiAgent (GPL-3.0), ContextPipe/Astra, TokenPilot/LightRSI, ESAA**: product runtimes, not
  research baselines we can drop into a shared loop.
- **BeliefMem, ContextRender, Schema, ContextEvo, official SKILL.state**: no public code found
  as of 2026-10-03.
- **StateMemBench, WorldLines, BeliefShift, SkillExecBench, DreamBench-SWE scoring oracles**:
  not released.

See [`docs/literature_map.md`](../docs/literature_map.md) for what each paper does.
