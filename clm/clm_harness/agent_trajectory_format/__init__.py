# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``ATIF-CTX-v1.1`` — a context-editing-friendly trajectory format.

Plain ATIF logs an append-only list of steps and assumes the live prompt is the
replay of all prior steps. That breaks for agents that edit/compact their own
context. This format changes exactly one thing at the root — ``steps`` becomes
``segments`` — and adds two fields to each :class:`Segment`:

* ``input_context`` — the literal OpenAI message list the agent starts the
  segment with (the ground truth of what the model saw), and
* ``branch`` — the :class:`BranchTrigger` (the context-management action) that
  opened the segment.

A context-management action closes the current segment and opens a new one, so a
trajectory is an append-only log of segments; within a segment the context is
append-only and ordinary ATIF replay holds. See :mod:`.models` for the full
spec and :func:`.models.segment_context_before` for reconstruction.

v1.1: sub-trajectories
----------------------
An agent that delegates (the CLM harness's ``SUBCTX_<slot>`` workers) is no
longer one chain. v1.1 adds ``Trajectory.subtrajectories``: a list of
:class:`SubTrajectory`, each a full trajectory in its own right (own ``agent``,
``segments``, ``final_metrics``, and nested ``subtrajectories``) plus two
:class:`SpawnLink` s addressed in the PARENT — ``spawn`` (the parent step that
launched the worker) and ``fold`` (the parent step that read its result back,
``None`` while the result never returned).

Graph semantics. :meth:`Trajectory.graph_edges` flattens the whole thing into
``(from_node, to_node, kind)`` triples over nodes named ``"<who>:seg<N>"``
(``who`` = ``"main"`` for the root, the ``sub_id`` for a child):

* consecutive segments inside one object -> the branch kind that opened the
  later segment (``"context_edit"`` by default);
* parent segment holding ``spawn.step_id`` -> child's first segment, ``"spawn"``;
* child's last segment -> parent segment holding ``fold.step_id``, ``"fold"``.

Backward compatible: an ``ATIF-CTX-v1`` file has no ``subtrajectories`` and
validates unchanged (both version strings are accepted, see
``models.SUPPORTED_SCHEMA_VERSIONS``). The strict segment chain
(``parent_segment_id == segment_id - 1``) is enforced within each object's own
segments; a child's chain and step ids are independent of its parent's.

Public API::

    from clm_harness.agent_trajectory_format import (
        TrajectoryBuilder, Trajectory, Segment, Step, ToolCall, Observation,
        ObservationResult, Metrics, FinalMetrics, Agent, ContextMessage,
        BranchTrigger, ContextStats, SpawnLink, SubTrajectory,
        validate_trajectory, SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS,
    )
"""

from __future__ import annotations

from .builder import TrajectoryBuilder
from .models import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    Agent,
    BranchTrigger,
    ContextMessage,
    ContextStats,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Segment,
    SpawnLink,
    Step,
    SubTrajectory,
    ToolCall,
    Trajectory,
    messages_from_step,
    segment_context_before,
)
from .validator import TrajectoryValidator, validate_trajectory

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Agent",
    "BranchTrigger",
    "ContextMessage",
    "ContextStats",
    "FinalMetrics",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Segment",
    "SpawnLink",
    "Step",
    "SubTrajectory",
    "ToolCall",
    "Trajectory",
    "TrajectoryBuilder",
    "TrajectoryValidator",
    "messages_from_step",
    "segment_context_before",
    "validate_trajectory",
]
