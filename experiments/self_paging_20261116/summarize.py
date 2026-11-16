"""Recompute the completed SFT tables from archived evidence; CPU, stdlib only.

Run from any directory: python3 /path/to/this/summarize.py
This verifies provenance and accounting, not the validity of the nominal Full control.
"""
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from self_paging.core.costs import episode_cost


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    collection = read_json(HERE / "provenance/server-collection.json")
    for name, spec in collection["files"].items():
        path = HERE / name
        data = gzip.decompress(Path(str(path) + ".gz").read_bytes()) if name.startswith("raw/") else path.read_bytes()
        assert len(data) == spec["bytes"]
        assert hashlib.sha256(data).hexdigest() == spec["sha256"], name

    # Publication-time source checks. For subsequent code revisions, use this commit.
    for manifest_name in ("rl-prep-source-manifest.json", "val50-source-manifest.json"):
        manifest = read_json(HERE / "provenance" / manifest_name)
        for name, sha in manifest.items():
            override = HERE / "provenance/sft_source_override" / name
            path = override if manifest_name.startswith("val50") and override.exists() else ROOT / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, (manifest_name, name)

    summaries = {}
    for path in sorted((HERE / "sft").glob("eval-*/summary.jsonl")):
        assert read_json(path.parent / "exit.json")["returncode"] == 0
        for row in rows(path):
            key = (row["method"], row["ratio"], row["seed"])
            assert key not in summaries
            summaries[key] = row
    expected = {(m, r, s) for m, r in [("full", 1.0), ("model", .25), ("model", .4), ("model", .6)] for s in range(100000, 100050)}
    assert set(summaries) == expected
    records = {}
    world_pages = {}
    decision_rows = []
    for path in sorted((HERE / "raw").glob("*.jsonl.gz")):
        for line in gzip.decompress(path.read_bytes()).splitlines():
            rec = json.loads(line)
            key = (rec["method"], rec["ratio"], rec["seed"])
            assert key not in records
            records[key] = rec
            summary = summaries[key]
            assert len(rec["decisions"]) == 4
            correct = sum(d["gold"] == d["predicted"] for d in rec["decisions"])
            assert rec["reward"] == summary["reward"] == correct / 4
            assert len(rec["tokens"]) == summary["tokens"] <= 32768
            assert rec["budget"] == summary["budget"]
            assert rec["peak_resident"] == summary["peak_resident"]
            assert rec["meta"]["over_budget"] == 0
            if rec["budget"] is not None:
                assert rec["peak_resident"] <= rec["budget"]
            for field, value in rec["meta"].items():
                if field in summary:
                    assert summary[field] == value, (key, field)
            for field, value in episode_cost(rec).items():
                assert math.isclose(summary[field], value, rel_tol=1e-10, abs_tol=1e-8), (key, field)
            pages = json.dumps(rec["environment_pages"], sort_keys=True)
            assert world_pages.setdefault(rec["seed"], pages) == pages
            for decision in rec["decisions"]:
                decision_rows.append(dict(method=key[0], ratio=key[1], seed=key[2], **decision,
                                          correct=decision["gold"] == decision["predicted"]))
    assert set(records) == expected
    by = defaultdict(list)
    for key, row in summaries.items():
        by[key[:2]].append(row)
    aggregated = []
    for (method, ratio), group in sorted(by.items()):
        recs = [records[(method, ratio, row["seed"])] for row in group]
        correct = sum(d["gold"] == d["predicted"] for rec in recs for d in rec["decisions"])
        item = dict(method=method, ratio=ratio, episodes=len(group), decisions=200,
                    correct=correct, accuracy=correct / 200,
                    perfect_episodes=sum(row["reward"] == 1 for row in group),
                    strict_full_history_control=False if method == "full" else None)
        for field in ["peak_resident", "cost"]:
            item["mean_" + field] = sum(row[field] for row in group) / len(group)
        for field in ["invalid", "kv_errors", "forced_evictions", "truncated_tokens", "kv_in", "kv_out", "notes", "over_budget"]:
            item[field] = sum(row[field] for row in group)
            item[field + "_episodes"] = sum(row[field] > 0 for row in group)
        aggregated.append(item)
    logs = rows(HERE / "sft/sft.jsonl")
    train = [r for r in logs if "step" in r]
    val = [r for r in logs if "eval_loss" in r]
    assert [r["step"] for r in train] == list(range(431))
    assert [r["completed_steps"] for r in val] == [50, 100, 150, 200, 250, 300, 350, 400, 431]
    assert all(math.isfinite(r["loss"]) for r in train)
    assert all(math.isfinite(r["eval_loss"]) for r in val)
    cloud = read_json(HERE / "sft/swanlab/cloud-verified.json")
    assert cloud["validation_matching_steps"] == [r["eval_step"] for r in val]
    assert cloud["local_latest"] == 430 and set(cloud["matching_steps"]) == {0, 430}
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    for name, table in [("sft_summary", aggregated), ("decisions", decision_rows), ("validation", val)]:
        (out / (name + ".json")).write_text(json.dumps(table, indent=2) + "\n")
        with (out / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    report = dict(status="ARCHIVE_ACCOUNTING_VERIFIED_WITH_FULL_CONTROL_QUALIFICATION",
                  episodes=200, decisions=800, optimizer_updates=431, validation_points=9,
                  raw_files=10, source_manifests_verified=True,
                  warning="Nominal Full executes model-issued KV tools; not a strict complete-history control.")
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
