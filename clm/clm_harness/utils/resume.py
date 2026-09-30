# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Warm restart of a trial, and the periodic checkpoints that make one possible.

A long open-problem run can be cut off with budget left (wall clock, lease, preemption).
Two halves are needed to continue it, and both live on the host:

* ``artifacts/workspace/`` -- the sandbox filesystem (scores/, archive/, programs, notes);
* the live context -- ``agent/trajectory.ctx.json`` (ATIF-CTX keeps every segment's
  ``input_context`` verbatim, so a context that survived its own compactions is recoverable
  exactly), or ``agent/messages.json`` from a checkpoint if the run never got to write one.

Both are replayed host-side rather than through the task instructions, because a model
asked to retype several KB of state truncates it.

Waiting until the trial ends to save either half is not enough: a sandbox VM can become
unreachable mid-run, after which nothing can be downloaded from it. ``Checkpointer``
therefore mirrors both halves to the host on a timer, into the very paths ``ResumeState``
reads, so an interrupted run resumes from at most one interval ago instead of from nothing.

Shared by every harness -- each one accepts a ``resume_from`` kwarg, calls ``restore()`` in
``setup()`` and ``replay()`` where it builds its opening messages, and calls the
checkpointer's ``maybe()`` once per turn.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from ..agent_trajectory_format.models import Trajectory, segment_context_before
from . import tokens as tk

logger = logging.getLogger(__name__)

WORKSPACE = "/workspace"
CHECKPOINT_MESSAGES = "messages.json"

# A restore runs inside the agent-setup timeout, and an agent that scores in a loop can leave a
# workspace with too many files to upload in time. Past this many files the upload is thinned
# to the newest ``PER_DIR_CAP`` of each directory. Score files are a record rather than state -- the host
# artifact keeps all of them, and the agent's own state (programs, archive, notes) is small.
MAX_RESTORE_FILES = int(os.environ.get("RESUME_MAX_FILES", "20000"))
PER_DIR_CAP = int(os.environ.get("RESUME_PER_DIR_CAP", "2000"))

# Appended after the replayed context. The files come back, but processes do not: a run
# resumed mid-flight would otherwise poll for background optimisers and workers that died
# with the old sandbox, and wait on output that will never arrive.
RESUME_MARKER = "SESSION RESUMED."
RESUME_NOTE = (
    "SESSION RESUMED. Your sandbox was restarted and this context was restored from the "
    "previous session, so /workspace is exactly as you left it -- scores, archive, notes "
    "and programs are all there, and your submission budget continues from what you have "
    "already spent (the evaluator counts the restored score files).\n\n"
    "What did NOT survive: every process is gone. Background optimisers, long-running jobs "
    "and any sub-agents you had launched are dead, and their unwritten results are lost. Do "
    "not wait on them or poll for their output. Re-check the state on disk before trusting "
    "anything in the context above about work that was still in progress, then carry on.\n\n"
    "You have NOT finished. If the context above ends with you wrapping up or trying to "
    "submit, that attempt is void -- the session was cut short, not completed. Do not submit "
    "or stop now: check what the evaluator reports as your remaining budget, and only finish "
    "when it tells you the budget is exhausted.\n\n"
    "Do NOT re-run the one-time setup block from the task description. Everything it creates "
    "already exists, and re-running it overwrites your restored files with the task's blank "
    "starting state -- including the program file, which currently holds your best result, "
    "not the seed. Inspect the files before writing to them."
)


# Used when the previous session's files survived but its conversation did not.
COLD_CONTEXT_NOTE = (
    "SESSION RESUMED. /workspace already holds substantial work from your previous sessions -- "
    "scores, an archive of scored candidates, notes and programs -- but the conversation from "
    "those sessions could not be recovered, so you are reading this with no memory of it.\n\n"
    "Start by reading the state on disk (the score files, the archive ledger and any notes) "
    "before you write anything. The program file holds your best result so far, not the seed, "
    "and your submission budget continues from what those score files already spent.\n\n"
    "Do NOT re-run the one-time setup block from the task description: everything it creates "
    "already exists, and re-running it would overwrite your own best work with the task's "
    "blank starting state. No processes survived, so nothing is running for you to wait on."
)


TRIM_NOTE = (
    "\n\nOne more thing: the restored context above is only the most recent part of the "
    "previous session -- its first {n} turns were dropped so you have room to work in this "
    "one. Nothing important is lost: your scores, archive, notes and programs are all on "
    "disk. Read them rather than relying on memory of the early exploration."
)

def _trim_to_budget(
    messages: list[dict[str, Any]], budget_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    """Keep the newest messages that fit; returns (kept, n_dropped).

    Measured the same way the harnesses measure their own budget, which means tiktoken over
    every text that reaches the prompt -- including the bash command in
    ``tool_calls[*].function.arguments``. A chars/4 pass over ``content`` alone reads a
    bash-heavy transcript at roughly half its size, so the replay would not fit the window.
    """
    sizes = [tk.count_tokens([m])[0] for m in messages]
    total, cut = sum(sizes), 0
    while cut < len(messages) and total > budget_tokens:
        total -= sizes[cut]
        cut += 1
    # A tool result whose request was just dropped is orphaned; APIs that pair them reject
    # that, and a model reading it has no idea what produced it.
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1
    return messages[cut:], cut


def _drop_unanswered_call(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drop trailing tool calls the previous run never got an answer to.

    A run cut off mid-turn -- a timeout, a lost sandbox, a killed job -- saves a context whose
    last message is the assistant asking for a command that never came back. Replaying that
    puts an unanswered ``tool_calls`` in front of the resume note, which OpenAI-shaped APIs
    reject outright (HTTP 400). The command may well have run, but its effect is in the workspace the agent gets back, so the honest
    thing is to forget the asking rather than invent a reply.
    """
    kept, dropped = list(messages), 0
    while kept and kept[-1].get("role") == "assistant" and kept[-1].get("tool_calls"):
        kept.pop()
        dropped += 1
    return kept, dropped


def _clear(path: Path) -> None:
    """Remove a directory tree even where it is not writable.

    A workspace that passed through a read-only archive comes back with read-only directories,
    and a plain rmtree cannot empty those: it leaves them, the checkpoint swap fails with
    ENOTEMPTY, and every checkpoint from then on fails the same way. The tree is on its way
    out, so its permissions are not worth preserving.
    """
    def force(func, target, _exc):
        os.chmod(os.path.dirname(target) or target, 0o700)
        try:
            os.chmod(target, 0o700)
        except OSError:
            pass
        func(target)

    try:
        try:
            shutil.rmtree(path, onexc=force)          # python >= 3.12
        except TypeError:
            shutil.rmtree(path, onerror=force)
    except (OSError, FileNotFoundError):
        pass


def _thin(ws: Path) -> Path:
    """``ws`` itself when it is a normal size, or a hardlinked copy of it that will upload.

    Only the newest ``PER_DIR_CAP`` files of each directory are linked, and what was left
    behind is named in ``RESTORE_TRUNCATED.json`` at the root so the resumed agent can see that
    its score directory is a sample rather than assume the evaluator lost its work. Hardlinks
    cost no space and no copying, so this stays cheap on a workspace that is already pathological.
    """
    n = 0
    for _, _, files in os.walk(ws):
        n += len(files)
        if n > MAX_RESTORE_FILES:
            break
    if n <= MAX_RESTORE_FILES:
        return ws

    staged = ws.with_name(".workspace.thinned")
    _clear(staged)
    dropped: dict[str, int] = {}
    for root, _, files in os.walk(ws):
        rel = Path(root).relative_to(ws)
        out = staged / rel
        out.mkdir(parents=True, exist_ok=True)
        keep = files
        if len(files) > PER_DIR_CAP:
            keep = sorted(files, key=lambda f: os.path.getmtime(os.path.join(root, f)))[-PER_DIR_CAP:]
            dropped[str(rel)] = len(files) - len(keep)
        for f in keep:
            try:
                os.link(os.path.join(root, f), out / f)
            except OSError:  # crosses a device, or is not a regular file
                shutil.copy2(os.path.join(root, f), out / f)
    (staged / "RESTORE_TRUNCATED.json").write_text(
        json.dumps({"total_files": n, "per_dir_cap": PER_DIR_CAP, "dropped_per_dir": dropped},
                   indent=2) + "\n", encoding="utf-8")
    logger.warning(
        "workspace has %d+ files; restoring the newest %d per directory (dropped: %s)",
        n, PER_DIR_CAP, ", ".join(f"{k}:{v}" for k, v in dropped.items()) or "none",
    )
    return staged


class Checkpointer:
    """Mirrors the sandbox and the live context to the host every ``interval_s`` seconds.

    Writes into the trial's own ``artifacts/workspace/`` and ``agent/messages.json`` rather
    than a private location, which is what ``ResumeState`` (and the callers that pick a trial
    to resume) already look for. A checkpointed trial is therefore resumable by exactly the
    same path as one that finished cleanly, with no caller changes: whatever the trial ends
    up writing there itself simply overwrites the last checkpoint.

    Inert unless ``interval_s`` is positive.
    """

    def __init__(
        self,
        logs_dir: Path | str,
        interval_s: float = 0.0,
        source: str = WORKSPACE,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.interval_s = float(interval_s)
        self.source = source
        self.workspace_dir = self.logs_dir.parent / "artifacts" / "workspace"
        self.n_saved = 0
        self._last = time.monotonic()  # first save is one full interval in, not at turn 1

    def _live_source(self, environment) -> str:
        """The sandbox's REAL working dir, resolved at call time.

        Tasks may set their own [environment] workdir (task.toml -> env._workdir,
        e.g. /home/workspace). The environment is the authority on where the agent
        actually works, per task; the constant is only the fallback for envs that
        don't expose a workdir.
        """
        wd = getattr(environment, "_workdir", None)
        return str(wd) if wd else self.source

    def _log_save(self, ok: bool, seconds: float, messages: int) -> None:
        """Append one line per attempt to ``agent/checkpoints.jsonl``.

        A checkpoint runs inside the turn loop, so it spends the run's wall clock. How much is
        worth knowing per run rather than assuming: a workspace grows over days, and a
        checkpoint that has quietly become expensive -- or that has been failing since the
        sandbox went unreachable -- should be visible in the artifacts, not inferred from gaps
        between turn timestamps.
        """
        try:
            n_files = n_bytes = 0
            if ok:
                for p in self.workspace_dir.rglob("*"):
                    if p.is_file():
                        n_files += 1
                        n_bytes += p.stat().st_size
            rec = {
                "t": time.time(), "n": self.n_saved, "ok": ok,
                "seconds": round(seconds, 2), "messages": messages,
                "files": n_files, "bytes": n_bytes,
            }
            out = self.logs_dir / "checkpoints.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001 - instrumentation must never cost a run
            pass

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0

    async def maybe(self, environment: Any, messages: list[dict[str, Any]]) -> bool:
        """Checkpoint if one is due. Cheap enough to call every turn."""
        if not self.enabled or time.monotonic() - self._last < self.interval_s:
            return False
        return await self.save(environment, messages)

    async def save(self, environment: Any, messages: list[dict[str, Any]]) -> bool:
        """Returns whether a complete checkpoint landed.

        Never raises: a checkpoint is insurance, and losing the run it protects because the
        insurance failed would be perverse. The clock is reset either way, so a sandbox that
        has stopped answering does not turn every turn into a doomed download attempt.
        """
        self._last = time.monotonic()
        t0 = time.monotonic()
        staging = self.workspace_dir.with_name(".workspace.partial")
        try:
            _clear(staging)
            staging.mkdir(parents=True, exist_ok=True)
            await environment.download_dir(source_dir=self._live_source(environment),
                                           target_dir=str(staging))
            # Swap only once the download is complete, so a failure part-way through leaves
            # the previous checkpoint intact instead of a half-written workspace.
            previous = self.workspace_dir.with_name(".workspace.superseded")
            _clear(previous)
            self.workspace_dir.parent.mkdir(parents=True, exist_ok=True)
            if self.workspace_dir.exists():
                os.replace(self.workspace_dir, previous)
            os.replace(staging, self.workspace_dir)
            _clear(previous)
            # The context is written after the workspace so it can never describe files that
            # were not captured -- a context ahead of its workspace would have the agent act
            # on results that no longer exist on disk.
            self._write_messages(messages)
        except Exception as exc:  # noqa: BLE001 - see docstring
            _clear(staging)
            logger.warning("checkpoint %d failed (%s: %s)", self.n_saved + 1, type(exc).__name__, exc)
            self._log_save(False, time.monotonic() - t0, len(messages))
            return False
        self.n_saved += 1
        seconds = time.monotonic() - t0
        logger.info(
            "checkpoint %d: %s + %d messages -> %s (%.1fs)",
            self.n_saved, self.source, len(messages), self.workspace_dir.parent, seconds,
        )
        self._log_save(True, seconds, len(messages))
        return True

    def _write_messages(self, messages: list[dict[str, Any]]) -> None:
        out = self.logs_dir / CHECKPOINT_MESSAGES
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.partial")
        tmp.write_text(json.dumps(messages, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, out)


class ResumeState:
    """Restores a previous trial into a fresh one. Inert unless ``resume_from`` is set."""

    def __init__(self, resume_from: str | Path | None = None) -> None:
        self.trial_dir: Path | None = None
        self.context_missing = False  # set when only the workspace could be recovered
        if resume_from:
            self.trial_dir = Path(resume_from).expanduser()
            if not self.trial_dir.is_dir():
                raise ValueError(f"resume_from is not a directory: {self.trial_dir}")

    @property
    def enabled(self) -> bool:
        return self.trial_dir is not None

    # ------------------------------------------------------------- workspace
    async def restore(self, environment: Any, target: str = "") -> None:
        """Push the cached workspace back into the sandbox's REAL workdir (the
        env's per-task workdir when it exposes one; /workspace fallback)."""
        if self.trial_dir is None:
            return
        if not target:
            wd = getattr(environment, "_workdir", None)
            target = str(wd) if wd else WORKSPACE
        ws = self.trial_dir / "artifacts" / "workspace"
        if not ws.is_dir():
            raise FileNotFoundError(
                f"{self.trial_dir} has no artifacts/workspace/ to resume from — the trial "
                "predates workspace caching, or ran without it in task.toml"
            )
        staged = _thin(ws)
        try:
            await environment.upload_dir(source_dir=str(staged), target_dir=target)
        finally:
            if staged != ws:
                _clear(staged)

    # --------------------------------------------------------------- context
    def prior_context(self) -> list[dict[str, Any]]:
        """The live context the previous run held at cutoff: the last segment's
        ``input_context`` with that segment's own steps materialised back onto it.

        The trajectory is only written when a run ends on its own terms, so a run that was
        cut off has none; its last checkpoint's ``messages.json`` is the same context, taken
        a little earlier, and is used as the fallback.
        """
        if self.trial_dir is None:
            return []
        traj = self.trial_dir / "agent" / "trajectory.ctx.json"
        if not traj.is_file():
            msgs = self.trial_dir / "agent" / CHECKPOINT_MESSAGES
            if msgs.is_file():
                return json.loads(msgs.read_text(encoding="utf-8"))
            # A trial that died before its first turn has a workspace but no context. Refusing
            # to resume it would make every later restart of the chain fail the same way. The
            # files are the part worth keeping -- come back with them and a fresh context.
            if (self.trial_dir / "artifacts" / "workspace").is_dir():
                logger.warning(
                    "%s has a workspace but no context; resuming the files with a fresh context",
                    self.trial_dir,
                )
                self.context_missing = True
                return []
            raise FileNotFoundError(
                f"no agent/trajectory.ctx.json or agent/{CHECKPOINT_MESSAGES} under {self.trial_dir}"
            )
        tr = Trajectory.model_validate_json(traj.read_text(encoding="utf-8"))
        if not tr.segments:
            return []
        last = tr.segments[-1]
        return [dict(m) for m in segment_context_before(last, len(last.steps))]

    def replay(
        self,
        messages: list[dict[str, Any]],
        protect: int = 2,
        budget_tokens: int | None = None,
        max_ratio: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Extend a harness's opening messages with the previous run's context.

        The caller's own prefix (system + task) is kept rather than the old one, since
        budget, step count and prompt may all differ this time; the same number of leading
        messages is dropped from the replay so the prefix is not duplicated. Harnesses that
        pin a prefix keep pinning exactly ``protect`` messages, so nothing else changes.

        Earlier notes are dropped, since only the newest one describes the sandbox the agent
        is now in; otherwise a long chain replays a note for every restart it has ever had.

        With ``budget_tokens``, the replay is capped at ``max_ratio`` of it, keeping the most
        recent turns. A context that survived to the end of a run is often near that run's
        limit, and replaying it whole leaves the resumed agent no room to work. The dropped turns are the ones already distilled into
        the workspace the agent gets back anyway -- scores, archive and notes.
        """
        prior = [
            m for m in self.prior_context()
            if not str(m.get("content") or "").startswith(RESUME_MARKER)
        ]
        if not prior:
            # Files without a conversation: say so, or the agent reads a workspace full of work
            # it does not remember doing and starts over on top of it.
            return [*messages, {"role": "user", "content": COLD_CONTEXT_NOTE}] \
                if self.context_missing else messages
        tail, dropped = prior[protect:], 0
        if budget_tokens:
            tail, dropped = _trim_to_budget(tail, int(max_ratio * budget_tokens))
        tail, unanswered = _drop_unanswered_call(tail)
        if unanswered:
            logger.warning(
                "dropped %d trailing tool call(s) the previous run never answered", unanswered
            )
        note = RESUME_NOTE if not dropped else RESUME_NOTE + TRIM_NOTE.format(n=dropped)
        return [*messages, *tail, {"role": "user", "content": note}]

    # ------------------------------------------------------------ provenance
    def write_manifest(self, logs_dir: Path | str) -> None:
        """Record what this run was resumed from, so a chain of restarts stays traceable."""
        if self.trial_dir is None:
            return
        out = Path(logs_dir) / "resumed_from.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        prior = self.prior_context()
        out.write_text(
            json.dumps(
                {
                    "resumed_from": str(self.trial_dir),
                    "replayed_messages": len(prior),
                    "replayed_chars": sum(len(str(m.get("content") or "")) for m in prior),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
