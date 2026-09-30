# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The contrastive note: evidence the proposer reads before writing candidate skills.

From the current skill's episodes on the training tasks, pick pairs of runs of the
same task where one did better than the other (higher reward, or the same reward at
lower cost), and write a summary table plus a step-indexed digest of each run in a
pair. Episodes are labelled E1, E2, ... so the proposer can cite them.
"""

from __future__ import annotations

import collections
from typing import Callable

from .tasks import Episode


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}g}" if abs(x) >= 1e4 or (0 < abs(x) < 1e-3) else f"{x:.{nd}f}"


def contrastive_pairs(episodes: list[Episode], k: int = 5) -> list[tuple[Episode, Episode, str]]:
    """Up to ``k`` (worse, better, kind) pairs, in this order of preference:

    1. ``reward``: two runs of the same task with different rewards;
    2. ``reward, across tasks``: the worst and best remaining runs of different tasks;
    3. ``cost``: two successful runs (reward > 0) of the same task with the same reward,
       the better one cheaper (this is what remains once every run succeeds).

    Within 1 and 3, tasks with the largest gap come first and slots go round robin
    over tasks, so a few tasks cannot fill every slot.
    """
    valid = [e for e in episodes if e.valid]
    by_task: dict[str, list[Episode]] = collections.defaultdict(list)
    for e in valid:
        by_task[e.task_id].append(e)
    used: set[int] = set()
    pairs: list[tuple[Episode, Episode, str]] = []

    def cost(e: Episode) -> float:
        return e.cost if e.cost is not None else 0.0

    def round_robin(queues: list[list[tuple[float, Episode, Episode]]], kind: str) -> None:
        for q in queues:
            q.sort(key=lambda x: x[0], reverse=True)
        queues = sorted((q for q in queues if q), key=lambda q: q[0][0], reverse=True)
        while len(pairs) < k and any(queues):
            for q in queues:
                while q and len(pairs) < k:
                    _, lo, hi = q.pop(0)
                    if id(lo) in used or id(hi) in used:
                        continue
                    pairs.append((lo, hi, kind))
                    used.update((id(lo), id(hi)))
                    break

    # 1. same task, different reward: worst runs against best runs
    queues = []
    for eps in by_task.values():
        s = sorted(eps, key=lambda e: (e.reward, -cost(e)))
        h = len(s) // 2
        q = [(hi.reward - lo.reward, lo, hi) for lo, hi in zip(s[:h], reversed(s[-h:]))
             if hi.reward > lo.reward]
        queues.append(q)
    round_robin(queues, "reward")

    # 2. across tasks: worst remaining against best remaining
    rest = sorted((e for e in valid if id(e) not in used), key=lambda e: (e.reward, -cost(e)))
    while len(pairs) < k and len(rest) >= 2 and rest[-1].reward > rest[0].reward:
        lo, hi = rest.pop(0), rest.pop(-1)
        if lo.task_id == hi.task_id:
            continue
        pairs.append((lo, hi, "reward, across tasks"))
        used.update((id(lo), id(hi)))

    # 3. same task and reward, successful runs: expensive against cheap
    queues = []
    for eps in by_task.values():
        groups: dict[float, list[Episode]] = collections.defaultdict(list)
        for e in eps:
            if e.reward > 0 and e.cost is not None and id(e) not in used:
                groups[e.reward].append(e)
        q = []
        for g in groups.values():
            s = sorted(g, key=cost, reverse=True)
            h = len(s) // 2
            q += [(cost(lo) - cost(hi), lo, hi) for lo, hi in zip(s[:h], reversed(s[-h:]))
                  if cost(lo) > cost(hi)]
        queues.append(q)
    round_robin(queues, "cost")
    return pairs


def build_note(
    episodes: list[Episode],
    digest: Callable[[Episode], str],
    *,
    k: int = 5,
) -> str:
    """Markdown note for the proposer: per-task table, pairs, and digests."""
    labels = {id(e): f"E{i}" for i, e in enumerate(episodes, 1)}
    valid = [e for e in episodes if e.valid]
    metrics = sorted({e.cost_metric for e in valid if e.cost_metric})
    lines = ["# Rollouts of the current skill on the training tasks", ""]
    acc = _mean([e.reward for e in valid])
    costs = [e.cost for e in valid if e.cost is not None]
    lines.append(f"episodes: {len(episodes)} ({len(episodes) - len(valid)} without a grade); "
                 f"mean reward {_fmt(acc)}; mean cost {_fmt(_mean(costs))} "
                 f"({', '.join(metrics) or 'not measured'})")
    lines += ["", "| task | episode | reward | cost | turns | context edits | nudges |",
              "|---|---|---|---|---|---|---|"]
    for e in episodes:
        x = e.extra or {}
        lines.append(f"| {e.task_id} | {labels[id(e)]} | {_fmt(e.reward, 2)} | {_fmt(e.cost)} "
                     f"| {x.get('total_iters', '-')} | {x.get('n_ctx_syncs', '-')} "
                     f"| {x.get('n_nudges', '-')} |")
    pairs = contrastive_pairs(episodes, k=k)
    lines += ["", f"## Contrastive pairs (n={len(pairs)})", "",
              "Each pair is a worse run and a better run, of the same task unless noted.", ""]
    for i, (lo, hi, kind) in enumerate(pairs, 1):
        lines.append(f"- pair {i} ({kind}): worse {labels[id(lo)]} (task {lo.task_id}, reward "
                     f"{_fmt(lo.reward, 2)}, cost {_fmt(lo.cost)}) vs better {labels[id(hi)]} "
                     f"(task {hi.task_id}, reward {_fmt(hi.reward, 2)}, cost {_fmt(hi.cost)})")
    for i, (lo, hi, _) in enumerate(pairs, 1):
        for tag, e in (("worse", lo), ("better", hi)):
            lines += ["", f"### pair {i}, {tag}: {labels[id(e)]} (task {e.task_id}, "
                      f"reward {_fmt(e.reward, 2)})", "", "```", digest(e), "```"]
    if not pairs:
        lines += ["", "No run did better than another run here; digests of up to "
                  f"{2 * k} runs follow.", ""]
        for e in valid[: 2 * k]:
            lines += ["", f"### {labels[id(e)]} (task {e.task_id}, reward {_fmt(e.reward, 2)})",
                      "", "```", digest(e), "```"]
    return "\n".join(lines) + "\n"
