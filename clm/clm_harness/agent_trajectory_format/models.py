# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Context-aware ATIF (``ATIF-CTX-v1.1``) — pydantic models.

A trajectory format written for CLM, for agents that edit their own context.
Compared with a plain step log, the ONE structural change is at the root:

    ATIF:      Trajectory.steps    : list[Step]        # append-only
    ATIF-CTX:  Trajectory.segments : list[Segment]     # append-only

Motivation. Plain ATIF assumes the live prompt at step *N* is the replay of
steps ``1..N-1`` (system + user + agent turns + observations). That assumption
breaks the moment an agent edits/compacts its OWN context (delete / collapse /
offload / rewrite), because the real prompt is no longer the sum of earlier
steps. Rather than invent a per-step "context diff", we cut the trajectory into
**segments**: a context-management action closes the current segment and opens a
new one, and the new segment stores, as ``input_context``, the exact message
list the agent starts from. So:

* WITHIN a segment the context is append-only -> ordinary ATIF replay holds.
* ACROSS segments the discontinuity is explicit and self-describing: each
  segment is self-contained from its ``input_context`` (never reconstruct a
  prompt across a segment boundary).

Reconstruction rule. The live context immediately before ``steps[k]`` of a
segment is ``input_context`` extended by materialising ``steps[:k]`` back into
OpenAI messages (see :func:`segment_context_before`). The authoritative record
of "what the model actually saw" is ``input_context`` (plus, when captured,
``Metrics.prompt_token_ids`` on each agent step).

v1.1: sub-trajectories. A delegating agent is not one chain. ``Trajectory`` gains
``subtrajectories: list[SubTrajectory]``; a :class:`SubTrajectory` is a full
trajectory (own ``agent`` / ``segments`` / ``final_metrics`` / nested
``subtrajectories``) plus ``sub_id`` and two :class:`SpawnLink` s addressed in
the PARENT: ``spawn`` (the parent step that launched the worker) and ``fold``
(the parent step that folded the result back, ``None`` if it never returned).
The segment chain stays strict WITHIN each object; a child's segments and step
ids are numbered from 1 independently of the parent. :meth:`Trajectory.graph_edges`
emits ``(from_node, to_node, kind)`` over nodes ``"main:seg3"`` /
``"subctx_2_1:seg1"``, with ``"spawn"``/``"fold"`` edges crossing between them —
that is what a graph renderer consumes. ``ATIF-CTX-v1`` files carry no
``subtrajectories`` and remain valid (see ``SUPPORTED_SCHEMA_VERSIONS``).

Dependency-light: pydantic + stdlib only (no harbor / litellm), so offline
tooling can build/validate trajectories without the serving stack.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "ATIF-CTX-v1.1"
# Versions this package reads. v1.1 only ADDS ``Trajectory.subtrajectories``, so
# every ``ATIF-CTX-v1`` file is a valid v1.1 file (and still validates).
SUPPORTED_SCHEMA_VERSIONS = ("ATIF-CTX-v1", "ATIF-CTX-v1.1")

# ATIF step provenance vs. raw OpenAI message roles (kept distinct on purpose).
Source = Literal["system", "user", "agent"]
Role = Literal["system", "user", "assistant", "tool"]


class _Base(BaseModel):
    """Shared config: unknown keys are preserved (round-trip friendly) and each
    model exposes ATIF's ``to_json_dict`` export (excludes ``None`` by default)."""

    # extra="allow" keeps unknown provider keys on round-trip; the empty
    # protected_namespaces lets us keep ATIF's ``model_name`` field name.
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def to_json_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)


# ---------------------------------------------------------------------------
# Step, tool-call, observation and metrics models
# ---------------------------------------------------------------------------
class ToolCall(_Base):
    tool_call_id: str
    function_name: str
    # ATIF uses a parsed object; a raw JSON string is also accepted verbatim.
    arguments: dict[str, Any] | str | None = None
    extra: dict[str, Any] | None = None


class ObservationResult(_Base):
    source_call_id: str | None = None
    content: str | list[Any] | None = None  # str, or content-parts (ATIF-v1.6)
    extra: dict[str, Any] | None = None


class Observation(_Base):
    results: list[ObservationResult] = Field(default_factory=list)
    extra: dict[str, Any] | None = None


class Metrics(_Base):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_usd: float | None = None
    logprobs: list[float] | None = None
    completion_token_ids: list[int] | None = None  # ATIF-v1.3
    prompt_token_ids: list[int] | None = None  # ATIF-v1.4 (the faithful prompt)
    extra: dict[str, Any] | None = None


class Step(_Base):
    """One interaction step. ``source`` is provenance; agent-only fields
    (``model_name``/``reasoning_content``/``tool_calls``/``metrics``) must only
    appear on ``source == 'agent'`` steps (enforced by the validator)."""

    step_id: int
    source: Source
    timestamp: str | None = None
    model_name: str | None = None
    message: str | list[Any] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None
    metrics: Metrics | None = None
    llm_call_count: int | None = None  # ATIF-v1.7
    extra: dict[str, Any] | None = None


class Agent(_Base):
    name: str
    version: str | None = None
    model_name: str | None = None
    tool_definitions: list[dict[str, Any]] | None = None  # ATIF-v1.5
    extra: dict[str, Any] | None = None


class FinalMetrics(_Base):
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_cached_tokens: int | None = None
    total_cost_usd: float | None = None
    total_steps: int | None = None
    total_segments: int | None = None  # ATIF-CTX addition
    extra: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# ATIF-CTX additions
# ---------------------------------------------------------------------------
class ContextMessage(_Base):
    """A single OpenAI-style message, stored VERBATIM as the agent saw it. This
    is the ground truth of the context (``extra='allow'`` keeps any provider
    fields on round-trip). ``tool_calls`` stays in raw OpenAI shape (a list of
    ``{"id", "type", "function": {"name", "arguments"}}``)."""

    role: Role
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    extra: dict[str, Any] | None = None


class ContextStats(_Base):
    """Cheap descriptors of a segment's ``input_context`` (for filtering/plots)."""

    n_messages: int | None = None
    n_tokens: int | None = None
    token_method: str | None = None
    extra: dict[str, Any] | None = None


class BranchTrigger(_Base):
    """Why a new segment was opened (i.e. the context-management action that made
    the context non-append-only). Absent on the first segment."""

    # context_edit | compaction | offload | reset | truncation | spawn | fold | ...
    # "spawn"/"fold" (v1.1) mark a segment boundary caused by delegating to / folding
    # back a sub-agent; such a branch points at the child via ``extra["sub_id"]``.
    kind: str = "context_edit"
    source_step_id: int | None = None  # step in the PARENT segment that triggered it
    tool_call_id: str | None = None
    removed_chars: int | None = None
    removed_tokens: int | None = None
    replacement_chars: int | None = None
    offload_files: list[str] | None = None
    summary: str | None = None  # human-readable, e.g. "collapsed turns #4..#9"
    extra: dict[str, Any] | None = None


class Segment(_Base):
    """An append-only run of ATIF steps sharing ONE base context.

    ``input_context`` is the literal message list at the segment's start; ``steps``
    are only the NEW turns generated on top of it (no duplication). The first
    segment has ``branch is None``; every later segment carries the
    :class:`BranchTrigger` that opened it and points ``parent_segment_id`` at the
    immediately preceding segment."""

    segment_id: int  # sequential from 1
    parent_segment_id: int | None = None
    branch: BranchTrigger | None = None
    input_context: list[ContextMessage]
    input_stats: ContextStats | None = None
    steps: list[Step] = Field(default_factory=list)
    extra: dict[str, Any] | None = None



class SpawnLink(_Base):
    """(v1.1) One end of a parent<->sub-agent link, addressed in the PARENT.

    ``segment_id``/``step_id`` name the parent step that launched the worker
    (``spawn``) or that folded its result back into the parent context
    (``fold``). ``path`` is the file that carried it across (the ``SUBCTX_<slot>``
    brief on spawn, the ``finished_subagents/...`` dir on fold)."""

    segment_id: int
    step_id: int
    tool_call_id: str | None = None
    time_s: float | None = None
    path: str | None = None
    summary: str | None = None
    extra: dict[str, Any] | None = None


class SubTrajectory(_Base):
    """(v1.1) A sub-agent's own trajectory, hung off its parent.

    Structurally a :class:`Trajectory` (own ``agent``, ``segments``,
    ``final_metrics``, ``extra``, and its own nested ``subtrajectories``) plus the
    two links that place it in the parent's timeline. The strict segment chain is
    enforced WITHIN this object, independently of the parent's chain; step ids are
    likewise the child's own, numbered from 1."""

    sub_id: str  # e.g. "subctx_2_1" (slot 2, launch 1)
    parent_trajectory_id: str | None = None
    spawn: SpawnLink
    fold: SpawnLink | None = None
    agent: Agent
    segments: list[Segment] = Field(default_factory=list)
    final_metrics: FinalMetrics | None = None
    subtrajectories: list["SubTrajectory"] = Field(default_factory=list)
    extra: dict[str, Any] | None = None


class Trajectory(_Base):
    """Root object: an append-only log of :class:`Segment` s."""

    schema_version: str = SCHEMA_VERSION
    trajectory_id: str | None = None
    session_id: str | None = None
    agent: Agent
    segments: list[Segment] = Field(default_factory=list)
    subtrajectories: list[SubTrajectory] = Field(default_factory=list)  # v1.1
    final_metrics: FinalMetrics | None = None
    extra: dict[str, Any] | None = None

    def graph_edges(self) -> list[tuple[str, str, str]]:
        """(v1.1) The trajectory as a DAG, for a graph renderer.

        Nodes are ``"<who>:seg<N>"`` (``who`` is ``"main"`` for the root and the
        ``sub_id`` for a sub-trajectory). Edges:

        * consecutive segments inside one object -> the branch kind that opened
          the later segment (``"context_edit"`` when unlabelled);
        * the parent segment holding ``spawn.step_id`` -> the child's first
          segment, kind ``"spawn"``;
        * the child's last segment -> the parent segment holding
          ``fold.step_id``, kind ``"fold"`` (only when ``fold`` is set).
        """
        return _graph_edges(self, "main")


def _graph_edges(obj: Any, who: str) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for prev, seg in zip(obj.segments, obj.segments[1:]):
        kind = seg.branch.kind if seg.branch is not None else "context_edit"
        edges.append((f"{who}:seg{prev.segment_id}", f"{who}:seg{seg.segment_id}", kind))
    for sub in getattr(obj, "subtrajectories", None) or []:
        if not sub.segments:
            continue
        edges.append(
            (f"{who}:seg{sub.spawn.segment_id}", f"{sub.sub_id}:seg{sub.segments[0].segment_id}", "spawn")
        )
        edges.extend(_graph_edges(sub, sub.sub_id))
        if sub.fold is not None:
            edges.append(
                (f"{sub.sub_id}:seg{sub.segments[-1].segment_id}", f"{who}:seg{sub.fold.segment_id}", "fold")
            )
    return edges


# ---------------------------------------------------------------------------
# Reconstruction helpers (best-effort inverse; input_context is authoritative)
# ---------------------------------------------------------------------------
def _tool_call_to_openai(tc: ToolCall) -> dict[str, Any]:
    args = tc.arguments
    if not isinstance(args, str):
        args = json.dumps(args or {})
    return {
        "id": tc.tool_call_id,
        "type": "function",
        "function": {"name": tc.function_name, "arguments": args},
    }


def messages_from_step(step: Step) -> list[dict[str, Any]]:
    """Materialise one ATIF step back into the OpenAI message(s) it represents:
    a user/system step -> one message; an agent step -> the assistant turn plus
    one ``tool`` message per observation result."""
    if step.source in ("system", "user"):
        return [{"role": step.source, "content": step.message}]

    assistant: dict[str, Any] = {"role": "assistant", "content": step.message or ""}
    if step.reasoning_content:
        assistant["reasoning_content"] = step.reasoning_content
    if step.tool_calls:
        assistant["tool_calls"] = [_tool_call_to_openai(tc) for tc in step.tool_calls]
    out = [assistant]
    if step.observation:
        for r in step.observation.results:
            out.append(
                {"role": "tool", "tool_call_id": r.source_call_id, "content": r.content}
            )
    return out


def segment_context_before(segment: Segment, step_index: int) -> list[dict[str, Any]]:
    """The live context the agent saw right before ``segment.steps[step_index]``:
    ``input_context`` extended by materialising the earlier steps in the segment.
    Pass ``len(segment.steps)`` for the context at the end of the segment."""
    msgs = [m.to_json_dict() for m in segment.input_context]
    for step in segment.steps[:step_index]:
        msgs.extend(messages_from_step(step))
    return msgs
