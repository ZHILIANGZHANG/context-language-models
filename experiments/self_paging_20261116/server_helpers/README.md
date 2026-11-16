# Deployed server helpers (frozen evidence)

These Python files are byte-identical to the collected server copies. They contain original machine paths and durable run-directory guards. Importing some worker scripts starts argument parsing/work: do not import or execute them merely to inspect this archive.

## Layout in the original deployment

- Control/scripts `H=/home/zhangcheng/self-paging-20261116`
- Data/checkpoints `N=/nas/Public/zhangcheng/self-paging-20261116`
- Interpreter `/data/zhangcheng/self-paging-20261116/venv/bin/python`
- SFT source `/data/zhangcheng/context-language-models-797680f-val50`
- Baseline/RL-preparation source `/data/zhangcheng/context-language-models-223cc65-rlprep`

The source snapshots and environment contract are documented in `../provenance/`. The credential file is intentionally absent. Obtain a local credential securely if reproducing on a separately configured machine; never commit it. The data/checkpoint artifacts are retained on the originating server, not in this Git archive.

## Pipeline entry points and status at collection

| Entry point | Purpose | Status |
|---|---|---|
| `build_quality_candidates.py`, `select_quality_cases.py`, `quality_pool_finalize.py` | CPU world candidates, quota/diversity checks | Completed |
| `run_quality_production.py`, `run_quality_supplement1.py` | Verified GPU prose rewriting, reject failed trajectories, targeted supplement | Completed |
| `assemble_quality_data.py`, `finalize_teacher_quality.py` | Final 2400/240 admission with coverage, SHA and semantic gates | Completed |
| `run_quality_val50.py` | Actual multi-rank smoke, fresh SFT, validation/checkpoint every 50, paired 200 evaluations | Completed; qualified results in parent README |
| `swanlab_val50_relay.py`, `swanlab_val50_verify.py` | Private online logging and independent metric readback | Completed |
| `build_rl_cpu.py` | Disjoint RL candidate pools and reference feasibility checks | Completed; GPU profile pending at collection |
| `queue_base_controls_three_gpu.py`, `base_controls_two_worker.py` | Two original-weight controls with dynamic idle-GPU dispatch | Active/unfinished; Full/tool-permission caveat applies |
| `queue_rl_profile.py`, `rl_profile_worker.py` | 64-condition, four-rollout GPU data diagnostic | Deferred behind baseline queue; not RL training |

`quality_writer.py`, `quality_writer_v3.py`, and corresponding probe scripts are included as referenced dependencies/history; actual final production uses V4 and rechecks reused trajectories. They are not additional runs to start. Superseded 800-condition and serial-first-world controllers are intentionally not presented as current entry points.

Reproduction requires rebuilding the recorded data inputs, assigning fresh control/output directories, restoring matching source/manifests, and passing the recorded environment/data gates. Existing run directories and `STOP` files must not be deleted to bypass guards. Copying a helper to a new folder is not sufficient to rerun the experiment safely.

All helper file hashes are in `../provenance/server-collection.json`. The main training plan, exact command arguments, and process exits are in `../sft/`; no command needs to be reconstructed from prose.
