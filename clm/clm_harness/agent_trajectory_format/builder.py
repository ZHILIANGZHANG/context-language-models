# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Incremental builder for ``ATIF-CTX-v1`` trajectories.

Drop-in for a live harness loop: open the first segment with the initial
messages, add a step per model turn, and call :meth:`branch` whenever a
context-management action rewrites the log. Step ids are assigned globally
(sequential from 1 across the whole trajectory); segments are numbered 1..N in
open order. Example::

    b = TrajectoryBuilder(Agent(name="clm-agent"), session_id=run_id,
                          token_counter=lambda m: tk.count_tokens(m)[0])
    b.open_segment(messages)                     # messages == [system, user]
    ...
    b.add_agent_step(message=msg, tool_calls=[ToolCall(...)],
                     observation=Observation(results=[ObservationResult(...)]),
                     metrics=Metrics(prompt_tokens=..., completion_tokens=...))
    ...
    # a context edit just mutated `messages` in place:
    b.branch(messages, trigger=BranchTrigger(kind="context_edit",
             source_step_id=edit_step.step_id, removed_chars=res.removed_chars,
             summary="collapsed turns #4..#9"))
    ...
    traj = b.build(final_metrics=FinalMetrics(total_cost_usd=cost))
    (logs_dir / "trajectory.ctx.json").write_text(json.dumps(traj.to_json_dict(), indent=2))

v1.1: a delegating agent opens a CHILD builder per sub-agent, which numbers its
own segments/steps from 1 and is attached on close::

    child = b.open_subtrajectory("subctx_2_1",
                                 SpawnLink(segment_id=1, step_id=spawn_step.step_id),
                                 Agent(name="worker"))
    child.open_segment(worker_messages); child.add_agent_step(...)
    b.close_subtrajectory("subctx_2_1", SpawnLink(segment_id=1, step_id=fold_step.step_id))
"""

from __future__ import annotations

from typing import Any, Callable

from .models import (
    SCHEMA_VERSION,
    Agent,
    BranchTrigger,
    ContextMessage,
    ContextStats,
    FinalMetrics,
    Observation,
    Segment,
    SpawnLink,
    Step,
    SubTrajectory,
    ToolCall,
    Trajectory,
)


def _as_context_message(m: Any) -> ContextMessage:
    if isinstance(m, ContextMessage):
        return m
    if isinstance(m, dict):
        return ContextMessage(**m)
    raise TypeError(f"context message must be a dict or ContextMessage, got {type(m).__name__}")


class TrajectoryBuilder:
    def __init__(
        self,
        agent: Agent | dict[str, Any],
        *,
        session_id: str | None = None,
        trajectory_id: str | None = None,
        schema_version: str = SCHEMA_VERSION,
        token_counter: Callable[[list[dict[str, Any]]], int] | None = None,
    ) -> None:
        self.agent = agent if isinstance(agent, Agent) else Agent(**agent)
        self.session_id = session_id
        self.trajectory_id = trajectory_id
        self.schema_version = schema_version
        self._token_counter = token_counter
        self._segments: list[Segment] = []
        self._cur: Segment | None = None
        self._n_segments = 0
        self._n_steps = 0
        # v1.1: sub-agent builders, keyed by sub_id, in open order.
        self._subs: dict[str, TrajectoryBuilder] = {}
        self._sub_links: dict[str, tuple[SpawnLink, SpawnLink | None]] = {}
        self._subtrajectories: list[SubTrajectory] = []

    # --- segments ----------------------------------------------------------
    def open_segment(
        self,
        input_context: list[Any],
        *,
        branch: BranchTrigger | dict[str, Any] | None = None,
        parent_segment_id: int | None = None,
        input_stats: ContextStats | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Segment:
        """Close the current segment (if any) and open a new one anchored on a
        snapshot of ``input_context``."""
        if self._cur is not None:
            self._segments.append(self._cur)
        self._n_segments += 1
        msgs = [_as_context_message(m) for m in input_context]
        trig = branch if branch is None or isinstance(branch, BranchTrigger) else BranchTrigger(**branch)
        self._cur = Segment(
            segment_id=self._n_segments,
            parent_segment_id=parent_segment_id,
            branch=trig,
            input_context=msgs,
            input_stats=input_stats or self._auto_stats(input_context),
            steps=[],
            extra=extra,
        )
        return self._cur

    def branch(
        self,
        input_context: list[Any],
        *,
        trigger: BranchTrigger | dict[str, Any],
        input_stats: ContextStats | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Segment:
        """Open a new segment caused by a context-management action; links
        ``parent_segment_id`` to the segment being closed."""
        if self._cur is None:
            raise RuntimeError("branch() called before open_segment()")
        return self.open_segment(
            input_context,
            branch=trigger,
            parent_segment_id=self._cur.segment_id,
            input_stats=input_stats,
            extra=extra,
        )

    # --- v1.1 sub-trajectories --------------------------------------------
    def open_subtrajectory(
        self,
        sub_id: str,
        spawn: SpawnLink | dict[str, Any],
        agent: Agent | dict[str, Any] | None = None,
    ) -> "TrajectoryBuilder":
        """Open a CHILD builder for a sub-agent launched at ``spawn`` (a step of
        THIS trajectory). The child numbers its own segments/steps from 1; call
        :meth:`close_subtrajectory` to attach it to ``subtrajectories``."""
        if sub_id in self._subs:
            raise ValueError(f"sub-trajectory {sub_id!r} is already open")
        if any(s.sub_id == sub_id for s in self._subtrajectories):
            raise ValueError(f"sub-trajectory {sub_id!r} was already closed")
        link = spawn if isinstance(spawn, SpawnLink) else SpawnLink(**spawn)
        child = TrajectoryBuilder(
            agent if agent is not None else self.agent,
            session_id=self.session_id,
            trajectory_id=sub_id,
            schema_version=self.schema_version,
            token_counter=self._token_counter,
        )
        self._subs[sub_id] = child
        self._sub_links[sub_id] = (link, None)
        return child

    def close_subtrajectory(
        self, sub_id: str, fold: SpawnLink | dict[str, Any] | None = None
    ) -> SubTrajectory:
        """Finalise the child builder and attach it to this trajectory."""
        child = self._subs.pop(sub_id, None)
        if child is None:
            raise KeyError(f"no open sub-trajectory {sub_id!r}")
        spawn, _ = self._sub_links.pop(sub_id)
        link = fold if fold is None or isinstance(fold, SpawnLink) else SpawnLink(**fold)
        built = child.build()
        sub = SubTrajectory(
            sub_id=sub_id,
            parent_trajectory_id=self.trajectory_id,
            spawn=spawn,
            fold=link,
            agent=built.agent,
            segments=built.segments,
            final_metrics=built.final_metrics,
            subtrajectories=built.subtrajectories,
        )
        self._subtrajectories.append(sub)
        return sub

    # --- steps -------------------------------------------------------------
    def add_step(self, step: Step | None = None, **fields: Any) -> Step:
        """Append a step to the current segment, assigning the next global
        ``step_id``. Pass a :class:`Step` or its fields as kwargs."""
        if self._cur is None:
            raise RuntimeError("add_step() called before open_segment()")
        if step is None:
            step = Step(step_id=0, **fields)
        self._n_steps += 1
        step.step_id = self._n_steps
        self._cur.steps.append(step)
        return step

    def add_user_step(self, message: Any, **fields: Any) -> Step:
        return self.add_step(source="user", message=message, **fields)

    def add_agent_step(
        self,
        *,
        message: Any = None,
        reasoning_content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        observation: Observation | None = None,
        metrics: Any = None,
        model_name: str | None = None,
        **fields: Any,
    ) -> Step:
        return self.add_step(
            source="agent",
            message=message,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            model_name=model_name or self.agent.model_name,
            **fields,
        )

    # --- finalise ----------------------------------------------------------
    def build(self, *, final_metrics: FinalMetrics | dict[str, Any] | None = None) -> Trajectory:
        if self._cur is not None:
            self._segments.append(self._cur)
            self._cur = None
        if final_metrics is None:
            fm: FinalMetrics | None = FinalMetrics(
                total_steps=self._n_steps, total_segments=len(self._segments)
            )
        elif isinstance(final_metrics, FinalMetrics):
            fm = final_metrics
            if fm.total_steps is None:
                fm.total_steps = self._n_steps
            if fm.total_segments is None:
                fm.total_segments = len(self._segments)
        else:
            fm = FinalMetrics(**final_metrics)
        return Trajectory(
            schema_version=self.schema_version,
            trajectory_id=self.trajectory_id,
            session_id=self.session_id,
            agent=self.agent,
            segments=self._segments,
            subtrajectories=list(self._subtrajectories),
            final_metrics=fm,
        )

    # --- helpers -----------------------------------------------------------
    def _auto_stats(self, messages: list[Any]) -> ContextStats:
        n_tokens = None
        if self._token_counter is not None:
            try:
                n_tokens = int(self._token_counter(messages))
            except Exception:
                n_tokens = None
        return ContextStats(n_messages=len(messages), n_tokens=n_tokens)
