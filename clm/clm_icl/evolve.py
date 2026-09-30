# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""In-context skill evolution for CLM.

Each round:

  1. run the current skill (the incumbent) on the training tasks;
  2. build a contrastive note from better vs worse runs of the same task;
  3. ask a proposer model for N candidate skills (SKILL.md);
  4. validate them (malformed candidates are dropped, never repaired);
  5. evaluate every valid candidate on the development tasks;
  6. keep the best candidate if it passes the gate against the incumbent.

The proposer only sees training rollouts. Selection uses the development tasks, which
default to the training tasks when no separate set is given. All state lives in the run
directory; running the same command again resumes where it stopped, reusing finished
episodes and proposals.

    python -m clm_icl.evolve --tasks path/to/tasks --model openai/qwen36-27b \\
        --api-base http://localhost:8000/v1 --proposer-model anthropic/<model> \\
        --rounds 5 --candidates 4 --out runs/evo1
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .note import build_note
from .propose import Proposer, litellm_proposer, parse_proposal, system_prompt, user_message
from .tasks import Episode, TaskSource, load_task_source

BASE = "base"  # the incumbent before any round: no skill, or the --init-skill


def log(msg: str) -> None:
    print(f"[clm_icl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Scores and the gate
# ---------------------------------------------------------------------------
@dataclass
class Score:
    n: int
    n_valid: int
    acc: float | None
    cost: float | None
    cost_metric: str | None
    # mean graded reward per task; the unit for standard errors
    task_rewards: dict[str, float] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.n > 0 and self.n_valid == self.n


def score(episodes: list[Episode]) -> Score:
    """Mean reward over graded episodes; mean cost only if every graded episode has a
    cost in one metric (otherwise None, meaning not comparable)."""
    valid = [e for e in episodes if e.valid]
    acc = sum(e.reward for e in valid) / len(valid) if valid else None
    metrics = {e.cost_metric for e in valid}
    cost = metric = None
    if valid and len(metrics) == 1 and all(e.cost is not None for e in valid):
        cost, metric = sum(e.cost for e in valid) / len(valid), metrics.pop()
    per_task: dict[str, list[float]] = {}
    for e in valid:
        per_task.setdefault(e.task_id, []).append(e.reward)
    task_rewards = {t: sum(v) / len(v) for t, v in per_task.items()}
    return Score(len(episodes), len(valid), acc, cost, metric, task_rewards)


def _sem(xs: list[float]) -> float:
    """Standard error of the mean (sample standard deviation); 0 for fewer than 2."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def diff_se(a: Score, b: Score) -> tuple[float, str]:
    """Standard error of acc(a) - acc(b), with tasks as the unit.

    Paired when both skills were graded on the same set of tasks: the standard error of
    the per-task differences. Otherwise unpaired: the two standard errors of the mean
    combined, sqrt(se_a^2 + se_b^2).
    """
    ta, tb = a.task_rewards or {}, b.task_rewards or {}
    if ta and set(ta) == set(tb):
        return _sem([ta[t] - tb[t] for t in ta]), "paired"
    return math.hypot(_sem(list(ta.values())), _sem(list(tb.values()))), "unpaired"


def gate(cand: Score, inc: Score, eps: float = 1e-9) -> tuple[bool, str]:
    """Gate on the development tasks, with d = acc(candidate) - acc(incumbent) and SE the
    standard error of d (see ``diff_se``):

    - d > SE: pass (higher accuracy);
    - d < -SE: fail (lower accuracy);
    - otherwise a tie: pass only if the candidate's mean cost is lower, in the same metric.

    The candidate must be graded on every development episode.
    """
    if not cand.complete:
        return False, f"only {cand.n_valid}/{cand.n} episodes graded"
    if inc.acc is None:
        return True, "incumbent has no accuracy"
    d = cand.acc - inc.acc
    se, kind = diff_se(cand, inc)
    acc_txt = f"acc {cand.acc:.4f} vs {inc.acc:.4f}, diff {d:+.4f}, SE {se:.4f} ({kind})"
    if d > se + eps:
        return True, f"higher accuracy: {acc_txt}"
    if d < -se - eps:
        return False, f"lower accuracy: {acc_txt}"
    if cand.cost is None or inc.cost is None or cand.cost_metric != inc.cost_metric:
        return False, f"tie within 1 SE, cost not comparable: {acc_txt}"
    cost_txt = f"cost {cand.cost:.4g} vs {inc.cost:.4g} {cand.cost_metric}"
    if cand.cost < inc.cost:
        return True, f"tie within 1 SE, lower cost: {acc_txt}; {cost_txt}"
    return False, f"tie within 1 SE, not cheaper: {acc_txt}; {cost_txt}"


def pick(passing: list[tuple[str, Score]]) -> str:
    """Most accurate passing candidate; candidates within one SE of it count as tied,
    and the cheapest of those wins."""
    best = max(passing, key=lambda x: x[1].acc)[1]
    tied = [(sid, s) for sid, s in passing if best.acc - s.acc <= diff_se(best, s)[0] + 1e-9]
    inf = float("inf")
    return min(tied, key=lambda x: (x[1].cost if x[1].cost is not None else inf, -x[1].acc))[0]


def pareto(scores: dict[str, Score]) -> list[str]:
    """Skills no other skill matches or beats on both accuracy and cost."""
    pts = {k: s for k, s in scores.items() if s.acc is not None and s.cost is not None}
    out = []
    for k, s in pts.items():
        dominated = any(o.cost_metric == s.cost_metric and o.acc >= s.acc and o.cost <= s.cost
                        and (o.acc > s.acc or o.cost < s.cost) for j, o in pts.items() if j != k)
        if not dominated:
            out.append(k)
    return sorted(out, key=lambda k: pts[k].cost)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class Evolution:
    def __init__(
        self,
        run_dir: str | Path,
        train: TaskSource,
        proposer: Proposer,
        *,
        dev: TaskSource | None = None,
        n_candidates: int = 4,
        reps: int = 1,
        n_pairs: int = 5,
        workers: int = 1,
        patience: int = 0,
        init_skill: str | Path | None = None,
    ) -> None:
        self.dir = Path(run_dir)
        self.train, self.dev = train, dev or train
        self.dev_split = "dev" if dev is not None else "train"
        self.proposer = proposer
        self.n_candidates, self.reps, self.n_pairs = n_candidates, reps, n_pairs
        self.workers, self.patience = max(1, workers), patience
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load()
        self.state["dev_equals_train"] = dev is None
        if dev is None:
            log("WARNING: no --dev-tasks given; selection uses the training tasks; "
                "pass --dev-tasks for an unbiased gate")
        self._save()
        if BASE not in self.state["skills"]:
            if init_skill:
                src = Path(init_skill)
                src = src if src.is_dir() else src.parent
                shutil.copytree(src, self.dir / "skills" / BASE, dirs_exist_ok=True)
            self.state["skills"][BASE] = {"round": 0, "parent": None, "status": "incumbent"}
            self._save()

    # -- state ---------------------------------------------------------------
    def _load(self) -> dict:
        p = self.dir / "state.json"
        if p.is_file():
            return json.loads(p.read_text())
        return {"incumbent": BASE, "round": 0, "no_improve": 0, "skills": {}, "history": []}

    def _save(self) -> None:
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1) + "\n")
        tmp.replace(self.dir / "state.json")

    def skill_dir(self, sid: str) -> Path | None:
        d = self.dir / "skills" / sid
        return d if (d / "SKILL.md").is_file() else None

    def skill_text(self, sid: str) -> str:
        d = self.skill_dir(sid)
        return (d / "SKILL.md").read_text() if d else ""

    # -- running tasks ---------------------------------------------------------
    def _episode(self, split: str, src: TaskSource, sid: str, tid: str, rep: int) -> Episode:
        cache = self.dir / "episodes" / split / sid / f"{tid}__r{rep}.json"
        if cache.is_file():
            cached = Episode.from_dict(json.loads(cache.read_text()))
            # A graded episode (any reward, including 0) is final. An ungraded one
            # (crash, infrastructure error) is run again.
            if cached.valid:
                return cached
        try:
            ep = src.run(tid, self.skill_dir(sid), self.dir / "trials" / split / sid, rep)
        except Exception as exc:  # a broken task must not stop the loop
            ep = Episode(task_id=tid, rep=rep, reward=None, error=f"{type(exc).__name__}: {exc}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ep.to_dict(), indent=1) + "\n")
        return ep

    def evaluate(self, split: str, src: TaskSource, sids: list[str]) -> dict[str, list[Episode]]:
        jobs = [(sid, tid, rep) for sid in sids for tid in src.task_ids() for rep in range(self.reps)]
        log(f"{split}: {len(jobs)} episodes for {', '.join(sids)}")
        with ThreadPoolExecutor(self.workers) as pool:
            eps = list(pool.map(lambda j: self._episode(split, src, *j), jobs))
        out: dict[str, list[Episode]] = {sid: [] for sid in sids}
        for (sid, _, _), ep in zip(jobs, eps):
            out[sid].append(ep)
        return out

    def dev_score(self, sid: str) -> Score:
        info = self.state["skills"][sid]
        if "dev" not in info or not Score(**info["dev"]).complete:
            info["dev"] = asdict(score(self.evaluate(self.dev_split, self.dev, [sid])[sid]))
            self._save()
        return Score(**info["dev"])

    # -- one round -------------------------------------------------------------
    def run_round(self, k: int) -> dict:
        inc = self.state["incumbent"]
        rdir = self.dir / "rounds" / f"r{k}"
        rdir.mkdir(parents=True, exist_ok=True)
        inc_score = self.dev_score(inc)

        # 1-2. rollouts of the incumbent on the training tasks, and the note
        train_eps = self.evaluate("train", self.train, [inc])[inc]
        note = build_note(train_eps, self.train.digest, k=self.n_pairs)
        (rdir / "note.md").write_text(note)

        # 3-4. propose and validate
        resp_path = rdir / "proposer_response.txt"
        if not resp_path.is_file():
            system = system_prompt(self.n_candidates)
            user = user_message(self.skill_text(inc), note, self.train.describe())
            (rdir / "proposer_request.txt").write_text(user)
            log(f"round {k}: asking the proposer for {self.n_candidates} candidates")
            try:
                resp = self.proposer(system, user)
            except Exception as exc:  # stop without finishing the round; a rerun retries
                (rdir / "proposer_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
                raise SystemExit(f"round {k}: proposer call failed ({type(exc).__name__}); "
                                 f"see {rdir / 'proposer_error.txt'}. Rerun to retry.")
            resp_path.write_text(resp)
        prop = parse_proposal(resp_path.read_text(), max_skills=self.n_candidates)
        (rdir / "analysis.md").write_text(prop.analysis + "\n")
        cands = []
        known = {self.skill_text(s).strip(): s for s in self.state["skills"]
                 if self.skill_dir(s) and not s.startswith(f"r{k}-")}
        for i, (slug, text) in enumerate(prop.skills, 1):
            if text.strip() in known:
                prop.rejected.append(f"skill {slug}: same text as {known[text.strip()]}")
                continue
            sid = f"r{k}-{i}-{slug}"
            d = self.dir / "skills" / sid
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(text)
            self.state["skills"].setdefault(sid, {"round": k, "parent": inc, "status": "candidate"})
            cands.append(sid)
        log(f"round {k}: {len(cands)} valid candidates, {len(prop.rejected)} rejections")

        # 5. evaluate candidates on the development tasks
        todo = [s for s in cands if "dev" not in self.state["skills"][s]
                or not Score(**self.state["skills"][s]["dev"]).complete]
        if todo:
            for sid, eps in self.evaluate(self.dev_split, self.dev, todo).items():
                self.state["skills"][sid]["dev"] = asdict(score(eps))
            self._save()

        # 6. gate against the incumbent
        results, passing = [], []
        for sid in cands:
            s = Score(**self.state["skills"][sid]["dev"])
            ok, why = gate(s, inc_score)
            self.state["skills"][sid].update(status="passed" if ok else "rejected", gate=why)
            results.append({"skill": sid, **asdict(s), "passed": ok, "reason": why})
            if ok:
                passing.append((sid, s))
        winner = pick(passing) if passing else None
        if winner:
            self.state["skills"][inc]["status"] = "retired"
            self.state["skills"][winner]["status"] = "incumbent"
            self.state["incumbent"] = winner
            self.state["no_improve"] = 0
        else:
            self.state["no_improve"] += 1
        rec = {"round": k, "incumbent_before": inc, "incumbent_score": asdict(inc_score),
               "candidates": results, "rejected_at_validation": prop.rejected,
               "winner": winner, "incumbent_after": self.state["incumbent"]}
        (rdir / "gate.json").write_text(json.dumps(rec, indent=1) + "\n")
        self.state["history"].append(rec)
        self.state["round"] = k
        self._save()
        log(f"round {k}: " + (f"new incumbent {winner}" if winner else f"kept {inc}"))
        return rec

    def run(self, rounds: int) -> str:
        start = self.state["round"] + 1
        for k in range(start, rounds + 1):
            if self.patience and self.state["no_improve"] >= self.patience:
                log(f"stopping: {self.patience} rounds without improvement")
                break
            self.run_round(k)
        scores = {sid: Score(**v["dev"]) for sid, v in self.state["skills"].items() if "dev" in v}
        self.state["frontier"] = pareto(scores)
        self._save()
        best = self.state["incumbent"]
        out = self.dir / "best_skill"
        shutil.rmtree(out, ignore_errors=True)
        if self.skill_dir(best):
            shutil.copytree(self.skill_dir(best), out)
        log(f"done: incumbent {best} {scores.get(best)}; frontier {self.state['frontier']}")
        return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m clm_icl.evolve",
        description="Evolve a context-management skill (SKILL.md) for ClmAgent from its own "
                    "rollouts. Rerun the same command to resume.")
    t = ap.add_argument_group("tasks")
    t.add_argument("--tasks", required=True,
                   help="training tasks: a Harbor task dir, a dir of task dirs, a file listing "
                        "task dirs, or module:attr naming a TaskSource")
    t.add_argument("--dev-tasks", help="development tasks for selection (default: --tasks)")
    t.add_argument("--task-description", default="",
                   help="text or a file describing the tasks, shown to the proposer")
    a = ap.add_argument_group("agent (Harbor task sources)")
    a.add_argument("--model", help="agent model, a litellm name (e.g. openai/qwen36-27b)")
    a.add_argument("--api-base", default=None, help="agent endpoint (default: $API_BASE)")
    a.add_argument("--config", default="", help="bcp | edgebench | a config YAML for run_harbor.sh")
    a.add_argument("--harbor-env", default=None, help="Harbor environment (default: docker)")
    a.add_argument("--cost-metric", default="auto", choices=["auto", "usd", "flops", "tokens"])
    a.add_argument("--harbor-arg", action="append", default=[],
                   help="extra argument for harbor trial start (repeatable)")
    a.add_argument("--trial-timeout", type=float, default=None, help="seconds per trial")
    p = ap.add_argument_group("proposer")
    p.add_argument("--proposer-model", required=True,
                   help="litellm model name; use the agent's model for self-evolution")
    p.add_argument("--proposer-api-base", default=None)
    p.add_argument("--proposer-temperature", type=float, default=0.4,
                   help="pass a negative value to omit it")
    p.add_argument("--proposer-max-tokens", type=int, default=16000)
    lp = ap.add_argument_group("loop")
    lp.add_argument("--rounds", type=int, default=3)
    lp.add_argument("--candidates", type=int, default=4)
    lp.add_argument("--reps", type=int, default=1, help="runs per task and skill")
    lp.add_argument("--pairs", type=int, default=5, help="contrastive pairs in the note")
    lp.add_argument("--workers", type=int, default=1, help="episodes run in parallel")
    lp.add_argument("--patience", type=int, default=0,
                    help="stop after this many rounds without a new incumbent (0: never)")
    lp.add_argument("--init-skill", default=None,
                    help="starting skill dir (or SKILL.md); default: no skill")
    lp.add_argument("--out", required=True, help="run directory")
    return ap


def main(argv: list[str] | None = None) -> int:
    import os

    args = build_parser().parse_args(argv)

    def text_or_file(v: str) -> str:
        return Path(v).read_text() if v and Path(v).is_file() else v

    harbor = dict(model=args.model, api_base=args.api_base or os.environ.get("API_BASE", ""),
                  config=args.config, harbor_env=args.harbor_env, cost_metric=args.cost_metric,
                  harbor_args=args.harbor_arg, timeout_s=args.trial_timeout,
                  description=text_or_file(args.task_description))

    def source(spec: str) -> TaskSource:
        if ":" not in spec or Path(spec).exists():
            if not args.model or not harbor["api_base"]:
                raise SystemExit("Harbor task sources need --model and --api-base (or $API_BASE)")
        return load_task_source(spec, **harbor)

    train = source(args.tasks)
    dev = source(args.dev_tasks) if args.dev_tasks else None
    proposer = litellm_proposer(
        args.proposer_model, api_base=args.proposer_api_base,
        temperature=None if args.proposer_temperature < 0 else args.proposer_temperature,
        max_tokens=args.proposer_max_tokens)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1) + "\n")
    evo = Evolution(run_dir, train, proposer, dev=dev, n_candidates=args.candidates,
                    reps=args.reps, n_pairs=args.pairs, workers=args.workers,
                    patience=args.patience, init_skill=args.init_skill)
    evo.run(args.rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
