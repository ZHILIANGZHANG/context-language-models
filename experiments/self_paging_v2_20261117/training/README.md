# SFT v2 rebuild and autonomous launch — 2026-11-17

## User task
Pull latest **main**, stop current baseline evaluations, defer ALL RL, rebuild reliable/diverse SFT data under new guide, restart SFT from original Qwen3-8B, and verify actual formal training plus SwanLab cloud metrics. Fully authorized to fix ordinary issues. **No GitHub push.** Do not stop at queue creation or claim training started before real optimizer steps/cloud readback.

## Current authority (read live files first)
- SSH zhangcheng@10.120.16.212, working socket `/tmp/self-paging-evalstatus-20261116.sock`.
- H=/home/zhangcheng/self-paging-v2-20261117
- C=/data/zhangcheng/context-language-models-sft-v2-8fc082e
- N=/nas/Public/zhangcheng/self-paging-v2-20261117
- P=/data/zhangcheng/self-paging-20261116/venv/bin/python
- Unique controller **H/campaign_batched.py**, screen `self-paging-sft-v2-batch`, latest PID37796 UID1037 start_ticks1264066563 cwdC. Confirm identity.json against /proc; owns H/campaign.lock AND oldshared `/home/zhangcheng/self-paging-20261116/campaign.lock`. NEVER duplicate.
- Latest phase GPU_TEACHER, 1426 accepted trajectories preserved at replacement1794849069; active GPU0/1/2, all other cards foreign occupied untouched. Read H/health.json, error.json, jobs/*/{launch,exit,accepted}.json and rows.jsonl for actual progress.
- Current controller resumes completed cells by acceptance records; worker resumes accepted rows within incomplete cells; no successful accepted data rerun. STOP respected.

## Source and environment
- Latest origin/main **8fc082ee74093e64ef9a7887fbe73e41e0344621** fetched, merged into existing local checkout preserving unpublished result commit. Local merged HEAD e2409ca88ebaac66b2a63c1dd652d5a3bc08e618. No push. Explicit GitHub proxy127.0.0.1:7897 used because direct fetch stalled.
- New upstream implements digest page headers, actual structural-budget overhead, realizable-page-in filters, disabled KV-tool enforcement and corrected pager reservations.
- Remote84-file source manifest. Only test compatibility edit: old test100009 asserted unsupported>0 although new digest restoration repaired evidence. Removed that obsolete assertion, retained forced_evictions>0 and passes=False. Initial90pass1failure evidence preserved; full real-tokenizer rerun **91passed0failed0skipped**,240.77sec. No scientific source changes.
- Warm reuse exact old env pins torch2.11.0+cu129,transformers5.23.0,peft0.19.1,numpy2.3.5,triton3.6.0,pytest8.4.2. No installs or environment replacement.
- Original model `/nas/Public/zhangcheng/models/Qwen3-8B`. Teacher local cache `/data/zhangcheng/self-paging-v2-model-cache/Qwen3-8B`,16.397GB, all source-stream/local SHA equality verified with stable source stats, receipt H/model-cache-ready.json. Formal training still references original NAS base path.
- Source/helper SHA verified before every job launch. Update helpers-manifest.json for any justified helper edit; avoid racing controller launch. Do not weaken scientific acceptance.

## Data plan and gates
- 1000 train worlds seeds0:1000,100 heldout worlds90000:90100, varied T1Config fully recorded. No RL/evaluation seeds used.
- 13 conditions/world: unbudgeted1.0/paging; .6/.45/.35 × paging/notes/eager/mixed. **14300 candidates completed;8603 passed preliminary audits.** Candidate gzip/SHA and failed metadata retained.
- Exact target **2400train/240heldout**.20% no-paging,30% notes,50% paging/eager/mixed; within finite paging modes80%restore/20%out-only, mode allocations proportional to actual restore availability.40 nonempty cells. Rare heldout .6/paging and .35/mixed pairs have zero quota; do not claim complete Cartesian balance.
- Original8B rewrites substantive verified action reasons using unchanged QualityWriterV4; plain next_events deterministic. Each trajectory is executed and independently audited: grammar/environment reward, factual semantics, exact resident-set transitions, budget, header-realizable restoration, zero teacher fallback, <=32768tokens. Up to4 bounded prose draws/candidate, all failed evidence retained. Rejected examples included invented page49 instead of supplied40; never admitted via fallback.
- Finalizer rereads gzip+SHA and repeats all audits; asserts2640unique(seed,ratio,mode), disjoint train/heldout seeds/worldSHA, actual no-paging15–30%, each nonempty finite paging mode >60% with page-in, quarantine restores, in-only/out-only/combined turn classes, zero invalid/forced/truncation/fallback/unrealizable/unoffered. Reports exact diversity/lengths.
- Plain-next training loss retains25% deterministically, same prior recipe; inputs unchanged. No historical v1 trajectories reused.

## Generation throughput fixes / preserved evidence
- Serial teacher initially86accepted; switched to independent-episode batched short-prompt generation. CPU padding/isolation/EOS probe passed.
- Real-model variable-length greedy serial/batch exact token probes3 then32 passed. First probe's unrelated literal READY-word assertion failed despite exact equivalence; preserved batch-probe-first-attempt; corrected content check to nonempty/EOS, kept exact equivalence.
- Bounded batch concurrency32, no change to oracle/tool semantics/quality gates. All earlier controllers/workers identity-verified exited before replacement; accepted data preserved.
- Fixed FIFO head-of-line blocking: process completed candidates as available while reserving one acceptance slot per in-flight candidate.30 randomized completion/failure tests prove selected set remains first quota successful candidates. Current worker H/teacher_worker_batched.py. This is execution scheduling only.
- Historical evidence dirs: serial-teacher-evidence, batch8-teacher-evidence, fifo-teacher-evidence. Old identity files there are NOT active. Current root identity.json authoritative.

## Automatic continuation
After teacher cells all succeed, controller runs finalize_data.py -> N/accepted/{train,heldout}.jsonl and data-accepted.json. Then claims genuinely idle GPUs (currently3; minimum3 configured), creates training-plan.json, runs training_witness.py on real short/long samples: two LoRA updates, all-rank parameter SHA equality, validation and checkpoint exact state checks. Smoke checkpoints discarded.
Formal original-Qwen3-8B SFT: LoRA r64 alpha128 dropout0,lr1e-4,AdamW(beta.9/.95),wd0,clip1,warmup.03,cosine,1epoch,seed0,FlexAttention32768token packing,gradient_checkpointing,defaultgrad_accum1. Actual optimizer step count computed from packed dataset and GPU count. **Validate/save every50**, full240 heldout. Output N/checkpoints/sft-v2-r64; H/sft.jsonl; H/jobs/sft-quality (root sft-quality symlink contract).
SwanLab CPU relay: project `zhangcheng525/context-language-models` PRIVATE, group self-paging-v2-20261117, run Qwen3-8B-SFT-v2-digest-realizable. SDK0.10.3, read-only auth preflight already passed. Credential oldH/.swanlab-key0600, never print. Relay creates H/swanlab/run.json. After3 actual steps controller reads Api.run/series and compares cloud loss/lr/step to local log; **H/swanlab/cloud-verified.json required before reporting launch complete**. No new RL/evaluation queue appended.

## Old task stopped
OldH=/home/zhangcheng/self-paging-20261116: sft-controls/STOP and rl-profile64/STOP set. Old controller1536925/workers2028232,2299982,2342898 exited; locks free before new campaign.143 completed baseline rows preserved. RL profile had never started. NEVER resume old queues. No external messages/Parallel/automation/subagents.

Latest check1794850512:2051/2640 accepted,2200 completed attempts,15/40 cells complete,22 pending+3active. No controller error. Current root PID37796 remains authority. Finalizer now additionally records actual no-paging ratio, zero-error counters, per-turn memory action counts, token lengths and requires in-only/out-only/combined coverage. It has not yet run. Interim quarantine-restoration audit found17 quarantine-page restores and reference-reader repair examples; this is NOT model causal performance. Continue until final accepted data, training smoke, actual SFT steps and cloud metric readback.
