# agent_trajectory_format (ATIF-CTX)

A trajectory format for agents that edit their own context.

Plain ATIF records an append-only list of steps and assumes the model's prompt at any step is
the replay of all earlier steps. That does not hold once the model rewrites its context.
ATIF-CTX keeps ATIF's steps and adds two things:

- **Intermediate segments.** The trajectory is a list of `segments`. Each segment starts with
  `input_context`, the exact message list the model saw at that point, and records the
  context edit that opened it (`branch`). Within a segment the context is append-only, so
  plain ATIF replay holds; across segments the recorded `input_context` is the ground truth.
- **Multi-agent settings.** A trajectory can carry `subtrajectories`: each sub-agent has its
  own agent, segments and metrics, plus links to the parent step that spawned it and the
  step where its result was folded back (`SpawnLink`). Sub-trajectories can nest.

`ClmAgent` writes `trajectory.ctx.json` in this format, and the FLOPs accounting in
`flops_metrics/` reads it.

```bash
python -m clm_harness.agent_trajectory_format.validator trajectory.ctx.json
```
