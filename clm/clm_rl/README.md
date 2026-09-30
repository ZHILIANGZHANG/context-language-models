# Reinforcement Learning for CLMs

RL training for CLM is not part of this repository. We trained with
[ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) (the Polar
rollout server and its `slime_bridge` to [Slime](https://github.com/THUDM/slime)).
Please refer to that codebase for the training setup: rollout server, gateway,
Slime installation, and launch scripts.

The one algorithmic change we made is to the **advantage function**. This directory
documents that change and provides it as patches.

## Dual-channel advantage

Standard GRPO broadcasts one group-normalized reward to every response token. We keep
that as the **task channel** and add an **efficiency channel** that only reaches the
tokens of context-management turns:

```
A[i, t] = r_i + w_eff * a_eff_i * m_i[t]
```

- `r_i`: group-normalized task reward of trajectory *i* (unchanged GRPO).
- `m_i[t]`: 1 if token *t* belongs to a turn whose only action edits the context
  mirror (`/tmp/.live_ctx/LIVE_CTX_MAIN.txt`), else 0.
- `a_eff_i`: a per-trajectory efficiency score computed within each rollout group,
  over the trajectories that solved the task and finished cleanly:

  ```
  a_eff_i = clip((mean_flops - flops_i) / mean_flops, -1, 1)
  ```

  `flops_i` is the trajectory's inference FLOPs with prefix-cache reuse accounted for,
  and `mean_flops` is the mean over the group's successful trajectories. Unsuccessful
  trajectories, and groups with fewer than two successes, get `a_eff = 0`.
- `w_eff`: weight of the efficiency channel (0.25 in our runs).

A context edit that makes a successful trajectory cheaper than its siblings is
reinforced on the tokens that made the edit. An edit that makes it more expensive is
discouraged there. Credit for the task outcome is unchanged. With `w_eff = 0`, or when
no turn edits the context, this reduces to plain GRPO.

## Patches

| patch | applies to | content |
|---|---|---|
| `patches/0001-slime_bridge-dual-channel-advantage.patch` | ProRL-Agent-Server (`8bc67cc`, also applies to current `stable`) | adds `src/slime_bridge/dual_channel_advantage.py`, the advantage hook above |
| `patches/0002-slime-forward-a_eff-and-role_masks.patch` | Slime (`bf9b1a3`) | passes `a_eff` and the per-token role masks from rollout samples to the trainer |

```bash
git -C ProRL-Agent-Server apply clm_rl/patches/0001-slime_bridge-dual-channel-advantage.patch
git -C slime apply clm_rl/patches/0002-slime-forward-a_eff-and-role_masks.patch
```

Enable the hook with these Slime arguments:

```bash
--advantage-estimator grpo \
--custom-advantage-function-path slime_bridge.dual_channel_advantage.custom_advantage_function
```

and set `POLAR_DUAL_CHANNEL_W_EFF=0.25` in the trainer environment.

The hook expects two inputs that the patches do not produce:

- `sample.metadata["a_eff"]`: the efficiency score above. Compute it in the reward
  post-processing step (`slime_bridge/reward_post_process.py`) after rewards are
  grouped. For each group, take the successful trajectories and compute their FLOPs
  and the clipped relative saving. Leave the task reward itself unscaled.
- `sample.metadata["polar"]["role_mask"]`: one 0/1 tag per response token. When
  building the trajectory, tag a turn with 1 if its assistant action is only a
  context edit, i.e. a bash command that writes the mirror file, and with 0 otherwise.
  Tag the tokens between turns with 0.

## Training configuration

| setting | value |
|---|---|
| base model | Qwen3.5-9B |
| algorithm | GRPO (token-mean loss, normalized advantages), plus DAPO dynamic sampling (drop groups with zero reward variance, over-sampling batch 16) |
| batch | 8 prompts × 32 samples per step |
| optimizer | Adam, lr 1e-6 constant, weight decay 0.1, betas (0.9, 0.98) |
| clipping | eps 0.2 / 0.28 |
| KL | 0.01, low-variance KL |
| context | 32K prompt, 32K response |
| efficiency weight | `w_eff = 0.25` |
| reward penalties | 0.3 for a trajectory with no tool call, 0.2 for a format failure (failed trajectories only) |
| training data | OpenResearcher `rl` split (3,040 prompts), one epoch |
