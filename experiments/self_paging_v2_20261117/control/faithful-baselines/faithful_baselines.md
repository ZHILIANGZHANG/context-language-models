# Qwen3-8B SFT: faithful StreamingLLM / H2O evaluation (2026-11-17)

The previous page/turn-level controls are superseded, not overwritten. They received a
system prompt requesting disabled memory tools. The previous token pager also pinned
structural pages, compressed only between turns, and omitted StreamingLLM position shift.
Those results cannot establish an advantage over the published algorithms.

## Pinned references and explicit portability choices

- StreamingLLM authors' repository `mit-han-lab/streaming-llm`, commit
  `2e5042606d69933d88fbf909bd77907456b9b4dd`, `kv_cache.py` and
  `pos_shift/modify_llama.py`: first four sink tokens plus recent tokens; store
  pre-RoPE keys, rotate retained keys at their cache positions, query at cache length.
- H2O authors' repository `FMInference/H2O`, commit
  `ac75c2a8a9e76832b2a4139b9363373b56336bfb`,
  `h2o_hf/utils_real_drop/modify_llama.py`, `H2OKVCache_LayerWise` and
  non-streaming `H2OLlamaAttention`: cumulative attention plus recent half;
  selected keys remain at their original global RoPE coordinates.

`hf/faithful_cache.py` ports these algorithms to Qwen3 QK normalization and GQA.
It is not an unmodified run of the authors' old Llama implementation. GQA averages
query-head probabilities within each KV-head group; cumulative scores use FP32.
Output attention uses SDPA in all arms; H2O computes matching causal attention
probabilities for its selection, without reading future queries. This is an
accuracy implementation, with no inference-speed or measured VRAM savings claim.

## Budget and execution

Keep the existing canonical `t1_budget` computed with the original environment
prompt before replacing system instructions, so all methods receive the same B.
B is maximum attended KV slots per KV head across all layers, including incoming
tokens; H2O/Streaming retain B-1 slots and reserve one incoming slot. This explicit
one-slot reservation keeps peak <= B, instead of hiding the B+1 transient of a
post-forward eviction implementation. H2O heavy slots=floor((B-1)/2), recent slots
=ceil((B-1)/2); Streaming sink=4, recent=B-5.

Prefill chunks never exceed remaining capacity. Once saturated, eviction occurs
after every token forward, including new environment tokens and generation.
All K/V are compacted physically. Evicted K/V are not archived, restored, replayed,
or recomputed. Headers/system/task/ack tokens are not exempt. Only Streaming's
four initial sink tokens receive special treatment. No external page-level
Recent/Belady policy, forced page eviction, or tool-body truncation is mixed in.

## Matched task protocol

Same frozen final SFT-v2-r64 adapter, seeds 100000..100049, four decisions/world,
ratios .25/.35/.40/.50/.60, deterministic greedy decoding, thinking disabled,
60 turns, 1024 generated tokens/turn, absolute log cap 32768. All environment
pages, gold decisions and digests remain unchanged; canonical budgets are paired.
Full, StreamingLLM and H2O have identical task-only system prompt/tools/rendering.
The Full arm retains the entire conversation (including acknowledgements).
The model arm retains its trained paging instructions and existing runner; its
dashboard, notes and control output consume its budget. System arms do not receive
misleading instructions to issue disabled memory calls or a false page-residency
dashboard. Invalid calls remain recorded and penalized uniformly (max 5).

This tests complete systems on equal GPU KV capacity. Model paging has a host
backing tier and note capability; StreamingLLM/H2O do not. It does not isolate
selection intelligence, prove physical memory/latency benefit, or remove the SFT
checkpoint's adaptation to its own memory-control interface.

## Acceptance (not a performance ordering gate)

CPU tests execute the pinned authors' cache selectors (H2O class extracted intact
by AST for old-Transformers import compatibility) on identical synthetic streams:
exact retention and K/V/score agreement; independent attention output and RoPE
checks before/after eviction; GQA; native no-eviction parity; no resurrection and
peak budget checks. Real SFT checkpoint CUDA probe checks <1e-4 max logit error and
exact top1 without eviction, then real eviction with finite logits and bound.
No rule requires H2O to beat Streaming or requires our method to win.

A real first-world canary for Full/model/Streaming/H2O precedes remaining jobs;
canary outputs are reused, not rerun. Acceptance checks artifact hashes, world
and budget identity, complete decision records, finite scores, no over-budget,
no system-forced eviction/truncation in automatic arms, and physical cache bounds.
Scores and invalid-tool counts are reported even when disappointing; no tuning on
test correctness. Final results include accuracy by decision type, actual peak
and mean decode KV, incomplete decisions, invalid tool calls, and world-paired
bootstrap intervals. Historical sampled (temperature .6) scores are not pooled.
