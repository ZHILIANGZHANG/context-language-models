# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Validator for ``ATIF-CTX-v1`` trajectories.

Runs the ATIF-v1.7 structural checks (schema/agent presence, sequential step
ids, agent-only fields, tool-call references, ISO-8601 timestamps) plus the
segment invariants this format adds:

* segment ids are sequential from 1;
* the first segment has no ``branch``/``parent_segment_id``; every later segment
  has a ``branch`` and ``parent_segment_id == segment_id - 1``;
* each segment's ``input_context`` is non-empty;
* ``BranchTrigger.source_step_id`` (when set) references a step in the parent
  segment.

v1.1 adds sub-trajectories. The chain rules above are enforced WITHIN each
object's own ``segments`` (a sub-trajectory's chain is independent of its
parent's), plus:

* ``sub_id`` is unique among the siblings hung off one parent;
* ``spawn`` (and ``fold`` when present) reference an existing
  ``(segment_id, step_id)`` pair in the PARENT;
* nested sub-trajectories are checked recursively.

A v1.0 (``ATIF-CTX-v1``) file has no ``subtrajectories`` and validates unchanged.

Usage::

    python -m clm_harness.agent_trajectory_format.validator trajectory.ctx.json
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from .models import Trajectory

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


def _is_iso8601(value: str) -> bool:
    if not _ISO_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


class TrajectoryValidator:
    def __init__(self) -> None:
        self._errors: list[str] = []

    def get_errors(self) -> list[str]:
        return list(self._errors)

    # --- entry point -------------------------------------------------------
    def validate(self, obj: Any) -> bool:
        self._errors = []
        data = self._load(obj)
        if data is None:
            return False
        try:
            traj = Trajectory.model_validate(data)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                self._errors.append(f"{loc}: {err['msg']}")
            return False
        self._check_semantics(traj)
        return not self._errors

    # --- helpers -----------------------------------------------------------
    def _load(self, obj: Any) -> Any:
        if isinstance(obj, Trajectory):
            return obj.to_json_dict(exclude_none=False)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            if os.path.exists(obj):
                try:
                    with open(obj, encoding="utf-8") as fh:
                        return json.load(fh)
                except (OSError, json.JSONDecodeError) as exc:
                    self._errors.append(f"could not read/parse file: {exc}")
                    return None
            try:
                return json.loads(obj)
            except json.JSONDecodeError as exc:
                self._errors.append(f"not a valid path or JSON string: {exc}")
                return None
        self._errors.append(f"unsupported input type: {type(obj).__name__}")
        return None

    def _check_semantics(self, traj: Trajectory) -> None:
        if not traj.schema_version:
            self._errors.append("trajectory.schema_version: required field is missing")
        self._check_object(traj, "trajectory")

    def _check_object(self, obj: Any, base: str) -> None:
        """Segment-chain + step checks for one Trajectory or SubTrajectory, then
        recurse into its sub-trajectories (each has its OWN chain and step ids)."""
        if not obj.agent.name:
            self._errors.append(f"{base}.agent.name: required field is missing")
        if not obj.segments:
            self._errors.append(f"{base}.segments: at least one segment is required")
            return

        seen_call_ids: set[str] = set()
        expected_step_id = 1
        for s_idx, seg in enumerate(obj.segments):
            sbase = f"{base}.segments.{s_idx}"
            expected_seg = s_idx + 1
            if seg.segment_id != expected_seg:
                self._errors.append(
                    f"{sbase}.segment_id: expected {expected_seg} (sequential from 1), got {seg.segment_id}"
                )
            if s_idx == 0:
                if seg.branch is not None:
                    self._errors.append(f"{sbase}.branch: first segment must not have a branch")
                if seg.parent_segment_id is not None:
                    self._errors.append(f"{sbase}.parent_segment_id: first segment must not have a parent")
            else:
                if seg.branch is None:
                    self._errors.append(f"{sbase}.branch: required on every non-first segment")
                if seg.parent_segment_id != seg.segment_id - 1:
                    self._errors.append(
                        f"{sbase}.parent_segment_id: expected {seg.segment_id - 1}, got {seg.parent_segment_id}"
                    )
            if not seg.input_context:
                self._errors.append(f"{sbase}.input_context: must be a non-empty message list")

            parent_step_ids = {st.step_id for st in obj.segments[s_idx - 1].steps} if s_idx > 0 else set()
            if seg.branch is not None and seg.branch.source_step_id is not None:
                if seg.branch.source_step_id not in parent_step_ids:
                    self._errors.append(
                        f"{sbase}.branch.source_step_id: {seg.branch.source_step_id} "
                        "does not reference a step in the parent segment"
                    )

            expected_step_id = self._check_steps(seg, sbase, expected_step_id, seen_call_ids)

        self._check_subtrajectories(obj, base)

    # --- v1.1 sub-trajectories --------------------------------------------
    def _check_subtrajectories(self, obj: Any, base: str) -> None:
        subs = getattr(obj, "subtrajectories", None) or []
        if not subs:
            return
        # (segment_id, step_id) pairs the child links may point at.
        pairs = {(seg.segment_id, st.step_id) for seg in obj.segments for st in seg.steps}
        seen_ids: set[str] = set()
        for i, sub in enumerate(subs):
            sb = f"{base}.subtrajectories.{i}"
            if not sub.sub_id:
                self._errors.append(f"{sb}.sub_id: required field is missing")
            elif sub.sub_id in seen_ids:
                self._errors.append(f"{sb}.sub_id: duplicate sub_id {sub.sub_id!r}")
            else:
                seen_ids.add(sub.sub_id)
            for name in ("spawn", "fold"):
                link = getattr(sub, name, None)
                if link is None:
                    continue
                if (link.segment_id, link.step_id) not in pairs:
                    self._errors.append(
                        f"{sb}.{name}: (segment_id={link.segment_id}, step_id={link.step_id}) "
                        "does not reference a step in the parent trajectory"
                    )
            self._check_object(sub, sb)

    def _check_steps(
        self, seg: Any, base: str, expected_step_id: int, seen_call_ids: set[str]
    ) -> int:
        for st_idx, step in enumerate(seg.steps):
            sbase = f"{base}.steps.{st_idx}"
            if step.step_id != expected_step_id:
                self._errors.append(
                    f"{sbase}.step_id: expected {expected_step_id} (sequential from 1), got {step.step_id}"
                )
            expected_step_id += 1

            if step.timestamp is not None and not _is_iso8601(step.timestamp):
                self._errors.append(f"{sbase}.timestamp: not an ISO-8601 timestamp: {step.timestamp!r}")

            if step.source != "agent":
                for fld in ("model_name", "reasoning_content", "tool_calls", "metrics"):
                    if getattr(step, fld) is not None:
                        self._errors.append(
                            f"{sbase}.{fld}: agent-only field present on a '{step.source}' step"
                        )

            for tc in step.tool_calls or []:
                if tc.tool_call_id:
                    seen_call_ids.add(tc.tool_call_id)

            if step.observation is not None:
                for r_idx, res in enumerate(step.observation.results):
                    cid = res.source_call_id
                    if cid is not None and cid not in seen_call_ids:
                        self._errors.append(
                            f"{sbase}.observation.results.{r_idx}.source_call_id: "
                            f"{cid!r} does not reference any prior tool_call_id"
                        )
        return expected_step_id


def validate_trajectory(obj: Any) -> bool:
    """Convenience wrapper: ``True`` iff ``obj`` is a valid ATIF-CTX trajectory."""
    return TrajectoryValidator().validate(obj)


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m clm_harness.agent_trajectory_format.validator <trajectory.json>")
        return 2
    path = argv[0]
    validator = TrajectoryValidator()
    if validator.validate(path):
        print(f"\u2713 Trajectory is valid: {path}")
        return 0
    print(f"\u2717 Trajectory validation failed: {path}\n")
    errors = validator.get_errors()
    print(f"Found {len(errors)} error(s):")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
