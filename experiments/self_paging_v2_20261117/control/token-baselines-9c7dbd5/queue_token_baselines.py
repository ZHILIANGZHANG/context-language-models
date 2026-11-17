"""Queue the matched StreamingLLM/H2O T1 comparison after the SFT evaluation."""
from __future__ import annotations

import contextlib
import fcntl
import importlib.metadata
import itertools
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from post_sft_eval import BASE, CHECKPOINT, H, OLD, alive, free_gpus, identity, read, rows, sha256, write

R = H / "token-baselines-9c7dbd5"
UPSTREAM = H / "post-sft-eval"
SOURCE = Path("/data/zhangcheng/context-language-models-token-baselines-9c7dbd5")
OUTPUT = Path("/nas/Public/zhangcheng/self-paging-v2-20261117/token-baselines-9c7dbd5")
COMMIT = "9c7dbd582610203cdf4ef80747424b93b5ec3022"
FIRST, LAST = 100000, 100050
METHODS = ("full", "model", "streaming", "h2o")
RATIOS = (0.25, 0.35, 0.4, 0.5, 0.6)
SANITY_LENGTHS, SANITY_RATIOS, SANITY_ITEMS = (4000, 8000), (0.1, 0.25), 50
SANITY_METHODS = ("full", "streaming", "h2o")


def check_stop():
    for root in (H, UPSTREAM, R):
        if (root / "STOP").exists():
            raise RuntimeError(f"STOP at {root}")


def health(phase, **details):
    write(R / "health.json", dict(phase=phase, time=time.time(), **details))


def preflight():
    manifest = read(R / "source-manifest.json")
    assert manifest["commit"] == COMMIT
    for name, expected in manifest["files"].items():
        assert sha256(SOURCE / name) == expected, name
    for package, expected in read(OLD / "env-spec.json")["pins"].items():
        assert importlib.metadata.version(package) == expected, package
    cpu = read(R / "cpu-tests.json")
    assert cpu["commit"] == COMMIT and cpu["passed"] >= 24 and cpu["failed"] == cpu["skipped"] == 0
    write(R / "plan.json", dict(commit=COMMIT, methods=METHODS, ratios=RATIOS,
          seeds=[FIRST, LAST], decisions_per_episode=4, temperature=0, checkpoint=str(CHECKPOINT),
          comparison_episodes=(LAST - FIRST) * (1 + 3 * len(RATIOS)), source=str(SOURCE),
          output=str(OUTPUT), depends_on=str(UPSTREAM / "complete.json")))


def wait_for_dependency():
    marker = UPSTREAM / "complete.json"
    controller = read(UPSTREAM / "identity.json")
    while not marker.exists():
        check_stop()
        if (UPSTREAM / "error.json").exists():
            raise RuntimeError(f"Parent evaluation reported an error: {UPSTREAM / 'error.json'}")
        if not alive(controller):
            raise RuntimeError("Parent evaluation exited without a completion record")
        state = read(UPSTREAM / "health.json")
        health("WAIT_TASK_ACCURACY", upstream_phase=state["phase"], upstream_pid=controller["pid"])
        time.sleep(15)
    result = read(marker)
    assert (result["status"], result["episodes"], result["decisions"]) == (
        "T1_EVALUATION_COMPLETE", 500, 2000)
    assert sha256(UPSTREAM / "summary.json") == result["summary_sha256"]
    assert sha256(UPSTREAM / "records.jsonl.gz") == result["records_sha256"]
    checkpoint = result["checkpoint"]
    assert Path(checkpoint["path"]) == CHECKPOINT and checkpoint["completed_steps"] == 516
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        assert sha256(CHECKPOINT / name) == checkpoint["sha256"][name], name
    write(R / "dependency-accepted.json", result)


def lock_campaign(stack):
    for path in (H / "campaign.lock", OLD / "campaign.lock"):
        handle = stack.enter_context(path.open("a"))
        while True:
            check_stop()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                health("WAIT_CAMPAIGN_LOCK", lock=str(path))
                time.sleep(10)


def command(module, model, *arguments):
    return [sys.executable, "-m", f"self_paging.eval.{module}", "--model", str(model),
            "--tokenizer", BASE, *map(str, arguments)]


def run_jobs(phase, jobs, active):
    pending = list(jobs)
    while pending or active:
        check_stop()
        for gpu, (proc, process_id, directory, log) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            del active[gpu]
            write(directory / "exit.json", dict(returncode=proc.returncode, identity=process_id, time=time.time()))
            if proc.returncode != 0:
                raise RuntimeError(f"Worker failed; see {directory / 'stdout.log'}")
        if pending:
            first = free_gpus()
            time.sleep(3)
            for gpu in sorted((first & free_gpus()) - active.keys()):
                if not pending:
                    break
                check_stop()
                name, args = pending.pop(0)
                directory = R / name
                directory.mkdir()
                log = (directory / "stdout.log").open("w")
                proc = subprocess.Popen(args, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                        env=os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)},
                                        start_new_session=True)
                process_id = identity(proc.pid)
                active[gpu] = proc, process_id, directory, log
                write(directory / "launch.json", dict(identity=process_id, gpu=gpu, args=args, time=time.time()))
        health(phase, pending=[name for name, _ in pending],
               active={gpu: item[1] for gpu, item in active.items()})
        if pending or active:
            time.sleep(10)


def sanity_jobs():
    return [(f"sanity-{length}", command("token_sanity", BASE,
            "--out", OUTPUT / f"sanity-{length}.jsonl", "--modes", "oneshot",
            "--methods", *SANITY_METHODS, "--lengths", length,
            "--budgets", *SANITY_RATIOS, "--n", SANITY_ITEMS)) for length in SANITY_LENGTHS]


def accept_sanity():
    rates = {}
    for length in SANITY_LENGTHS:
        data = list(rows(OUTPUT / f"sanity-{length}.jsonl"))
        groups = {(m, b): [] for m in SANITY_METHODS
                  for b in ((1.0,) if m == "full" else SANITY_RATIOS)}
        for row in data:
            assert row["mode"] == "oneshot" and row["length"] == length
            groups[row["method"], row["ratio"]].append(row)
        for (method, ratio), group in groups.items():
            assert sorted(r["item"] for r in group) == list(range(SANITY_ITEMS))
            rates[f"{length}/{method}@{ratio}"] = sum(r["correct"] for r in group) / len(group)
    full = statistics.mean(v for k, v in rates.items() if "/full@" in k)
    h2o = statistics.mean(v for k, v in rates.items() if "/h2o@" in k)
    streaming = statistics.mean(v for k, v in rates.items() if "/streaming@" in k)
    passed = full >= h2o > streaming
    write(R / "sanity-result.json", dict(passed=passed, cells=rates, full=full, h2o=h2o,
          streaming=streaming, rule="mean Full >= mean H2O > mean StreamingLLM"))
    if not passed:
        raise RuntimeError("The server pass-key test did not reproduce the registered relative trend")


def comparison_job(lo, hi):
    name = f"t1-{lo}-{hi}"
    args = command("control_eval", CHECKPOINT, "--task", "t1", "--seeds", lo, hi,
                   "--methods", *METHODS, "--ratios", *RATIOS, "--n-decisions", 4,
                   "--out", OUTPUT / name)
    return name, args


def verify_shard(lo, hi, scores, totals, worlds, budgets):
    directory = OUTPUT / f"t1-{lo}-{hi}"
    for seed, method in itertools.product(range(lo, hi), METHODS):
        for ratio in ((1.0,) if method == "full" else RATIOS):
            check_stop()
            name = f"{seed}-{method}-{ratio}"
            row = read(directory / f"{name}.json")
            raw_path = directory / f"{name}.trajectory.json"
            assert (row["seed"], row["method"], row["ratio"]) == (seed, method, ratio)
            assert sha256(raw_path) == row["trajectory_sha256"]
            raw = read(raw_path)
            assert len(raw["decisions"]) == 4
            correct = sum(d["gold"] == d["predicted"] for d in raw["decisions"])
            assert row["reward"] == raw["reward"] == correct / 4
            assert row["tokens"] <= 32768 and row["meta"]["over_budget"] == 0
            assert worlds.setdefault(seed, raw["environment_pages"]) == raw["environment_pages"]
            group = method, ratio
            assert seed not in scores.setdefault(group, {})
            scores[group][seed] = correct / 4
            numbers = totals.setdefault(group, {"kinds": {}, "stats": {}})
            for decision in raw["decisions"]:
                kind = numbers["kinds"].setdefault(decision["kind"], [0, 0])
                kind[0] += int(decision["gold"] == decision["predicted"])
                kind[1] += 1
            peak = row["peak_resident"]
            if method == "full":
                assert row["budget"] is None
            else:
                budget = row["budget"]
                assert budget == budgets.setdefault((seed, ratio), budget)
                assert peak <= budget
            if method != "model":
                assert all(row["meta"].get(k, 0) == 0 for k in ("kv_out", "kv_in", "notes"))
            cost = row["cost_batch32"]
            for key, value in (("peak_resident", peak), ("mean_resident_decode", cost["mean_resident_decode"]),
                               ("cost_batch32", cost["cost"]), ("cost_batch1", row["cost_batch1"]["cost"]),
                               ("keep_changes", raw.get("keep_changes", 0)),
                               *((k, row["meta"].get(k, 0)) for k in (
                                   "forced_evictions", "truncated_tokens", "invalid", "kv_in", "kv_out"))):
                assert math.isfinite(value)
                numbers["stats"].setdefault(key, []).append(value)


def finalize(ranges):
    scores, totals, worlds, budgets = {}, {}, {}, {}
    for lo, hi in ranges:
        verify_shard(lo, hi, scores, totals, worlds, budgets)
    assert set(worlds) == set(range(FIRST, LAST))
    groups, paired = {}, {}
    for (method, ratio), values in scores.items():
        assert set(values) == set(worlds)
        acc = totals[method, ratio]
        groups[f"{method}@{ratio}"] = dict(episodes=len(values), accuracy=statistics.mean(values.values()),
            correct=int(4 * sum(values.values())), decisions=4 * len(values), by_decision_type=acc["kinds"],
            **{f"mean_{k}" if not k.startswith("mean_") else k: statistics.mean(v)
               for k, v in acc["stats"].items()})
        if method in ("streaming", "h2o"):
            delta = [scores["model", ratio][seed] - values[seed] for seed in sorted(values)]
            rng = random.Random(0)
            draws = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(4000))
            paired[f"model-minus-{method}@{ratio}"] = dict(delta=statistics.mean(delta),
                ci95=[draws[100], draws[3899]], unit="world")
    write(R / "summary.json", dict(groups=groups, paired_accuracy_delta=paired))
    write(R / "complete.json", dict(status="COMPLETE", episodes=sum(len(v) for v in scores.values()),
          summary_sha256=sha256(R / "summary.json"), time=time.time()))
    health("COMPLETE")


def main():
    active = {}
    with contextlib.ExitStack() as stack:
        handle = stack.enter_context((R / "controller.lock").open("a"))
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (R / "identity.json").exists(), "Controller was already started"
        write(R / "identity.json", identity(os.getpid()))
        try:
            check_stop()
            preflight()
            wait_for_dependency()
            lock_campaign(stack)
            OUTPUT.mkdir(exist_ok=False)
            run_jobs("SANITY", sanity_jobs(), active)
            accept_sanity()
            first = FIRST, FIRST + 1
            remainder = [(FIRST + 1, FIRST + 5)] + [(lo, lo + 5) for lo in range(FIRST + 5, LAST, 5)]
            run_jobs("CANARY", [comparison_job(*first)], active)
            verify_shard(*first, {}, {}, {}, {})
            run_jobs("T1_COMPARISON", [comparison_job(*r) for r in remainder], active)
            health("VERIFY_RESULTS")
            finalize([first, *remainder])
        except BaseException:
            write(R / "error.json", dict(traceback=traceback.format_exc(), time=time.time()))
            for process, process_id, directory, log in active.values():
                if process.poll() is None and alive(process_id):
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                log.close()
                write(directory / "exit.json", dict(returncode=process.returncode, time=time.time()))
            raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--preflight"]:
        preflight()
        print(read(R / "plan.json"))
    else:
        main()
