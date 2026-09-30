# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Task sources: the general interface between the evolution loop and the tasks.

The loop only needs three things from a set of tasks: the list of task ids, a way to
run one task with a given skill, and the result of that run (reward, cost, and where
the trajectory is). ``TaskSource`` is that interface. ``HarborTaskSource`` implements it
for Harbor task directories run with ``ClmAgent``; plug in your own subclass for
anything else (see ``load_task_source``).
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARBOR_SCRIPT = REPO_ROOT / "clm" / "examples" / "run_harbor.sh"


@dataclass
class Episode:
    """One run of one task with one skill.

    ``reward`` is None when the run produced no grade (crash, infrastructure error);
    such an episode is invalid, not a zero. ``cost`` is None when it was not measured.
    ``cost_metric`` names the unit of ``cost`` (e.g. ``usd``, ``flops``, ``tokens``);
    costs with different metrics are never compared.
    """

    task_id: str
    rep: int
    reward: float | None
    cost: float | None = None
    cost_metric: str | None = None
    trajectory_path: str | None = None
    log_dir: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.reward is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Episode":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


class TaskSource(ABC):
    """A set of tasks to evaluate a skill on.

    Subclasses implement ``task_ids`` and ``run``. ``run`` must not raise for a failed
    task: return an ``Episode`` with ``reward=None`` and ``error`` set instead.
    ``skill_dir`` is a directory holding a ``SKILL.md``, or None to run without a skill.
    ``out_dir`` is a fresh directory the run may write to.
    """

    @abstractmethod
    def task_ids(self) -> list[str]: ...

    @abstractmethod
    def run(self, task_id: str, skill_dir: Path | None, out_dir: Path, rep: int) -> Episode: ...

    def describe(self) -> str:
        """Optional text about the tasks, shown to the proposer."""
        return ""

    def digest(self, episode: Episode, max_chars: int = 6000) -> str:
        """Readable summary of an episode's trajectory, shown to the proposer."""
        return digest_trajectory(episode.trajectory_path, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Harbor + ClmAgent
# ---------------------------------------------------------------------------
def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-") or "x"


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def harbor_reward(trial_dir: Path) -> float | None:
    """Reward of a Harbor trial: ``result.json`` first, then ``verifier/reward.{json,txt}``."""
    res = _read_json(trial_dir / "result.json") or {}
    rewards = (res.get("verifier_result") or {}).get("rewards") or {}
    if rewards:
        v = rewards.get("reward", next(iter(rewards.values())))
        return float(v) if v is not None else None
    rj = _read_json(trial_dir / "verifier" / "reward.json")
    if isinstance(rj, dict) and rj.get("reward") is not None:
        return float(rj["reward"])
    rt = trial_dir / "verifier" / "reward.txt"
    if rt.is_file():
        try:
            return float(rt.read_text().strip())
        except ValueError:
            return None
    return None


def clm_cost(agent_dir: Path, metric: str = "auto") -> tuple[float | None, str | None]:
    """Cost of a ClmAgent run from its log directory.

    ``usd``: ``usage.json`` ``cost_usd``. ``flops``: prefix-reuse (cache-aware) FLOPs from
    ``trajectory.ctx.json`` (needs a FLOPs model size, see ``flops_model_key``).
    ``tokens``: prompt + completion tokens from ``usage.json``. ``auto`` picks usd when
    it is positive, else flops when present, else tokens.
    """
    usage = _read_json(agent_dir / "usage.json") or {}
    usd = usage.get("cost_usd")
    tokens = None
    if usage.get("prompt_tokens") is not None:
        tokens = float(usage.get("prompt_tokens") or 0) + float(usage.get("completion_tokens") or 0)
    flops = None
    ctx = _read_json(agent_dir / "trajectory.ctx.json") or {}
    extra = (ctx.get("final_metrics") or {}).get("extra") or {}
    for key in ("compute_cost", "kv_cache_flops"):
        v = (extra.get(key) or {}).get("cache_aware_flops")
        if v is not None:
            flops = float(v)
            break
    if metric == "usd":
        return (float(usd) if usd is not None else None), "usd"
    if metric == "flops":
        return flops, "flops"
    if metric == "tokens":
        return tokens, "tokens"
    if usd:
        return float(usd), "usd"
    if flops is not None:
        return flops, "flops"
    return tokens, ("tokens" if tokens is not None else None)


class HarborTaskSource(TaskSource):
    """Harbor task directories, each run with ``ClmAgent`` through ``clm/examples/run_harbor.sh``.

    ``spec`` is a Harbor task directory (has ``task.toml``), a directory of task
    directories, or a text file with one task directory per line. The skill reaches the
    agent through ClmAgent's ``skill_dirs`` kwarg. ``config`` is passed to the script
    (``bcp``, ``edgebench`` or a YAML path); ``harbor_args`` are appended to
    ``harbor trial start``.
    """

    def __init__(
        self,
        spec: str | Path,
        *,
        model: str,
        api_base: str,
        config: str = "",
        harbor_env: str | None = None,
        cost_metric: str = "auto",
        harbor_args: list[str] | None = None,
        script: str | Path = DEFAULT_HARBOR_SCRIPT,
        timeout_s: float | None = None,
        description: str = "",
    ) -> None:
        self.tasks = self._resolve(Path(spec).expanduser())
        if not self.tasks:
            raise FileNotFoundError(f"no Harbor tasks (task.toml) found in {spec}")
        self.model, self.api_base, self.config = model, api_base, config
        self.harbor_env = harbor_env
        self.cost_metric = cost_metric
        self.harbor_args = list(harbor_args or [])
        self.script = Path(script).expanduser().resolve()
        self.timeout_s = timeout_s
        self.description = description

    @staticmethod
    def _resolve(spec: Path) -> dict[str, Path]:
        if spec.is_file():
            dirs = [Path(ln.strip()).expanduser() for ln in spec.read_text().splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
        elif (spec / "task.toml").is_file():
            dirs = [spec]
        elif spec.is_dir():
            dirs = sorted(d for d in spec.iterdir() if (d / "task.toml").is_file())
        else:
            raise FileNotFoundError(f"task spec not found: {spec}")
        out: dict[str, Path] = {}
        for d in dirs:
            d = d.resolve()
            tid = d.name
            while tid in out:
                tid += "_"
            out[tid] = d
        return out

    def task_ids(self) -> list[str]:
        return list(self.tasks)

    def describe(self) -> str:
        return self.description

    def run(self, task_id: str, skill_dir: Path | None, out_dir: Path, rep: int) -> Episode:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        trial_name = _safe(f"{task_id}__r{rep}")
        trial_dir = out_dir / trial_name
        if trial_dir.exists():  # leftover of an interrupted run
            shutil.rmtree(trial_dir)
        cmd = [str(self.script), str(self.tasks[task_id]), self.config,
               "--trial-name", trial_name]
        if skill_dir is not None:
            cmd += ["--agent-kwarg", f"skill_dirs={Path(skill_dir).resolve()}"]
        cmd += self.harbor_args
        env = dict(os.environ, API_BASE=self.api_base, MODEL=self.model,
                   TRIALS_DIR=str(out_dir.resolve()))
        if self.harbor_env:
            env["HARBOR_ENV"] = self.harbor_env
        error = None
        try:
            with open(out_dir / f"{trial_name}.log", "w") as log:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=self.timeout_s, check=False).returncode
            if rc != 0:
                error = f"harbor exited with code {rc}"
        except subprocess.TimeoutExpired:
            error = f"timed out after {self.timeout_s}s"
        agent_dir = trial_dir / "agent"
        reward = harbor_reward(trial_dir)
        cost, metric = clm_cost(agent_dir, self.cost_metric)
        traj = next((str(p) for p in (agent_dir / "trajectory.ctx.json",
                                      agent_dir / "trajectory.json") if p.is_file()), None)
        usage = _read_json(agent_dir / "usage.json") or {}
        keep = ("total_iters", "n_lm_calls", "n_ctx_syncs", "n_ctx_rejected", "n_nudges",
                "n_retry_on_limit", "budget_finalized", "prompt_tokens", "completion_tokens")
        return Episode(
            task_id=task_id, rep=rep, reward=reward, cost=cost, cost_metric=metric,
            trajectory_path=traj, log_dir=str(trial_dir),
            error=error if reward is None else None,
            extra={k: usage[k] for k in keep if k in usage},
        )


# ---------------------------------------------------------------------------
# Trajectory digest (for the proposer)
# ---------------------------------------------------------------------------
def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(_text(p.get("text") if isinstance(p, dict) else p) for p in x)
    return json.dumps(x, default=str)


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else f"{s[: n // 2]} ...[{len(s) - n} chars]... {s[-n // 2:]}"


def _args(a: Any) -> str:
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return a
    if isinstance(a, dict) and "command" in a:
        return str(a["command"])
    return json.dumps(a, default=str)


def digest_trajectory(path: str | None, max_chars: int = 6000, per_item: int = 400) -> str:
    """Step-indexed text digest of a trajectory file.

    Understands ClmAgent's ``trajectory.ctx.json`` (segments; a new segment marks a
    context edit), plain ATIF (``steps``), and an OpenAI message list. Anything else is
    shown as raw text.
    """
    if not path or not Path(path).is_file():
        return "(no trajectory)"
    raw = Path(path).read_text(errors="replace")
    try:
        data = json.loads(raw)
    except ValueError:
        return _clip(raw, max_chars)
    lines: list[str] = []

    def step_lines(steps: list[dict]) -> None:
        for s in steps:
            sid = s.get("step_id")
            if s.get("source") != "agent":
                if s.get("source") == "user" and sid and sid > 2:
                    lines.append(f"step {sid} [user] {_clip(_text(s.get('message')), per_item)}")
                continue
            msg = _text(s.get("message"))
            if msg.strip():
                lines.append(f"step {sid} [say] {_clip(msg, per_item)}")
            for tc in s.get("tool_calls") or []:
                lines.append(f"step {sid} [cmd] {_clip(_args(tc.get('arguments')), per_item)}")
            for r in (s.get("observation") or {}).get("results") or []:
                out = _text(r.get("content"))
                lines.append(f"step {sid} [out {len(out)} chars] {_clip(out, per_item)}")

    if isinstance(data, dict) and "segments" in data:
        for seg in data["segments"]:
            st = seg.get("input_stats") or {}
            if seg.get("segment_id", 1) > 1:
                lines.append(f"--- context edit: segment {seg.get('segment_id')} starts with "
                             f"{st.get('n_messages')} messages, {st.get('n_tokens')} tokens ---")
            step_lines(seg.get("steps") or [])
    elif isinstance(data, dict) and "steps" in data:
        step_lines(data["steps"])
    elif isinstance(data, list):
        for i, m in enumerate(data):
            role = m.get("role") if isinstance(m, dict) else "?"
            if role in ("system",):
                continue
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    lines.append(f"msg {i} [cmd] {_clip(_args(fn.get('arguments')), per_item)}")
                if _text(m.get("content")).strip():
                    lines.append(f"msg {i} [say] {_clip(_text(m.get('content')), per_item)}")
            else:
                c = _text(m.get("content") if isinstance(m, dict) else m)
                lines.append(f"msg {i} [{role} {len(c)} chars] {_clip(c, per_item)}")
    else:
        return _clip(raw, max_chars)
    return _clip("\n".join(lines), max_chars) if lines else "(empty trajectory)"


# ---------------------------------------------------------------------------
# Loading a source from the command line
# ---------------------------------------------------------------------------
def load_task_source(spec: str, **harbor_kwargs: Any) -> TaskSource:
    """``module:attr`` imports a user TaskSource (a class or factory called with no
    arguments, or an instance); anything else is a Harbor task spec."""
    if re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", spec) and not Path(spec).exists():
        mod, attr = spec.split(":")
        obj = getattr(importlib.import_module(mod), attr)
        src = obj() if callable(obj) and not isinstance(obj, TaskSource) else obj
        if not isinstance(src, TaskSource):
            raise TypeError(f"{spec} did not produce a TaskSource")
        return src
    return HarborTaskSource(spec, **harbor_kwargs)
