# In-Context Learning for CLMs

A CLM agent's context-management behavior can be steered with a skill: a `SKILL.md` text
appended to its system prompt. `clm_icl` improves that skill from the agent's own runs,
with the model weights and the harness fixed. Each round:

1. run the current skill on the training tasks;
2. build a note that contrasts better and worse runs (a success and a failure of the same
   task, or two successes where one was cheaper), with a step-indexed digest of each run;
3. ask a proposer model for N candidate skills, each a full rewrite that cites the steps it
   targets and predicts its effect on accuracy and cost;
4. validate the candidates (frontmatter, non-empty body, cited evidence; malformed ones are
   dropped, not repaired);
5. run every candidate on the development tasks;
6. keep the best candidate if it passes the gate; it becomes the current skill.

The proposer can be a stronger model than the agent (assisted evolution) or the agent's
own model (self-evolution). It only reads training runs. Evaluate the final skill once on
a separate test set yourself.

## The gate

On the development tasks, let d be the candidate's mean reward minus the current skill's,
and SE the standard error of d, with tasks as the unit (reward averaged over repeats of a
task). When both skills were graded on the same tasks, SE is paired: the standard error
of the per-task differences. Otherwise it combines the two standard errors of the mean,
sqrt(SE_candidate^2 + SE_current^2). A candidate replaces the current skill if

- d > SE (more accurate), or
- |d| <= SE (a tie) and its mean cost is lower, in the same cost metric,

and every one of its development episodes was graded. Among the candidates that pass,
those within one SE of the most accurate count as tied, and the cheapest of them wins.
Each round's `gate.json` gives d, SE, the SE kind and the costs for every candidate. At
the end, `state.json` also lists the Pareto frontier (accuracy against cost) over every
skill evaluated.

Without `--dev-tasks`, selection uses the training tasks, which the proposer has seen;
the run prints a warning and records `dev_equals_train: true` in `state.json`.

## Run on Harbor tasks

Skills reach `ClmAgent` through its `skill_dirs` kwarg; each episode is one
`clm/examples/run_harbor.sh` trial.

```bash
python -m clm_icl.evolve \
  --tasks path/to/train_tasks --dev-tasks path/to/dev_tasks \
  --model openai/qwen36-27b --api-base http://localhost:8000/v1 \
  --proposer-model anthropic/<model> \
  --rounds 5 --candidates 4 --reps 2 --workers 8 \
  --out runs/evo1
```

Run it with `clm/` on `PYTHONPATH` (or after `pip install -e .`). `--tasks` is a Harbor
task directory, a directory of task directories, or a file with one task directory per
line. Without `--dev-tasks`, selection uses the training tasks. `--config` passes `bcp`,
`edgebench` or a YAML file to the script; `--harbor-arg` adds `harbor trial start`
arguments. `--cost-metric` picks the cost: `usd` (`usage.json`), `flops` (prefix-reuse
FLOPs from `trajectory.ctx.json`, which needs `cost_metric=flops` and `flops_model_key`
in the config), `tokens`, or `auto`. `--init-skill` starts from an existing skill instead
of none; `--task-description` gives the proposer a short text about the tasks. For
self-evolution, pass the agent's model as `--proposer-model` (and its endpoint as
`--proposer-api-base`).

Rerun the same command to resume: graded episodes, proposals and scores are reused; episodes that ended without a grade are run again.

## Your own tasks

Anything that can run a task with a skill and return a reward can drive the loop.
Subclass `TaskSource`:

```python
from pathlib import Path
from clm_icl import Episode, TaskSource

class MyTasks(TaskSource):
    def task_ids(self) -> list[str]:
        return ["task-a", "task-b"]

    def run(self, task_id: str, skill_dir: Path | None, out_dir: Path, rep: int) -> Episode:
        # run the agent with skill_dir/SKILL.md (None = no skill), write logs to out_dir
        return Episode(task_id=task_id, rep=rep, reward=1.0, cost=0.02, cost_metric="usd",
                       trajectory_path=str(out_dir / "trajectory.json"))
```

and pass it as `--tasks mymodule:MyTasks`. Return `reward=None` for a run that produced no
grade; it counts as missing, not as zero. Override `digest(episode)` to control what the
proposer reads of a trajectory (the default reads ClmAgent's `trajectory.ctx.json`, ATIF,
or an OpenAI message list), and `describe()` to describe the tasks. From Python, use
`Evolution(run_dir, train_source, proposer, dev=dev_source).run(rounds)`; a proposer is
any function `(system, user) -> text`.

## Run directory

```
state.json            incumbent, per-skill scores and gate results, round history, frontier
skills/<id>/SKILL.md  every candidate (base = the starting skill, if any)
rounds/r<k>/          note.md, proposer_request.txt, proposer_response.txt, analysis.md, gate.json
episodes/<split>/<id>/  one JSON per episode (reward, cost, trajectory path)
trials/<split>/<id>/  Harbor trial outputs
best_skill/SKILL.md   the final incumbent
```

## Files

| file | role |
|---|---|
| `evolve.py` | the loop (`Evolution`), the gate, and the CLI |
| `tasks.py` | `TaskSource`, `Episode`, `HarborTaskSource`, trajectory digest |
| `note.py` | contrastive pair selection and the note |
| `propose.py` | proposer call (litellm), response parsing and validation |
| `proposer_prompt.md` | the proposer's system prompt |
