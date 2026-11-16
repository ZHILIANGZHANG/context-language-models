"""Verify a frozen partial baseline snapshot and recompute paired SFT tables.

No GPU or third-party dependencies. Does not read live server results.
"""
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SFT = HERE.parent / "self_paging_20261116"
sys.path.insert(0, str(ROOT))
from self_paging.core.costs import episode_cost


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    meta = load(HERE / "provenance/collection.json")
    for name, spec in meta["files"].items():
        raw = (HERE / name).read_bytes()
        assert len(raw) == spec["bytes"]
        assert hashlib.sha256(raw).hexdigest() == spec["sha256"], name
    for name, sha in load(HERE / "provenance/source-manifest.json").items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    baseline = rows(HERE / "summary.jsonl")
    sft = [row for p in sorted((SFT / "sft").glob("eval-*/summary.jsonl")) for row in rows(p) if row["method"] == "model"]
    sft_index = {(r["seed"], r["ratio"]): r for r in sft}
    sft_raw = {}
    for p in sorted((SFT / "raw").glob("*.jsonl.gz")):
        data = gzip.decompress(p.read_bytes())
        spec = load(SFT / "provenance/server-collection.json")["files"]["raw/" + p.name[:-3]]
        assert hashlib.sha256(data).hexdigest() == spec["sha256"]
        for line in data.splitlines():
            r = json.loads(line)
            if r["method"] == "model":
                sft_raw[(r["seed"], r["ratio"])] = r
    keys = set()
    decisions = []
    for row in baseline:
        key = (row["seed"], row["method"], row["ratio"])
        assert key not in keys
        keys.add(key)
        assert row["model_path"].endswith("/Qwen3-8B")
        raw = (HERE / "raw" / Path(row["raw_path"]).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        rec = json.loads(gzip.decompress(raw))
        assert (rec["seed"], rec["method"], rec["ratio"]) == key
        assert len(rec["decisions"]) == 4
        correct = sum(d["gold"] == d["predicted"] for d in rec["decisions"])
        assert row["reward"] == rec["reward"] == correct / 4
        paired = sft_index[(key[0], key[2])]
        reference = sft_raw[(key[0], key[2])]
        assert row["budget"] == rec["budget"] == paired["budget"]
        assert rec["environment_pages"] == reference["environment_pages"]
        assert [(d['page'], d['gold']) for d in rec['decisions']] == [(d['page'], d['gold']) for d in reference['decisions']]
        assert row["peak_resident"] == rec["peak_resident"] <= row["budget"]
        assert len(rec["tokens"]) == row["tokens"] <= 32768
        for field, value in rec["meta"].items():
            if field in row:
                assert row[field] == value, (key, field)
        for field, value in episode_cost(rec).items():
            assert math.isclose(row[field], value, rel_tol=1e-10, abs_tol=1e-8), (key, field)
        for d in rec["decisions"]:
            decisions.append(dict(seed=key[0], method=key[1], ratio=key[2], **d,
                                  correct=d["gold"] == d["predicted"]))
    assert len(baseline) == meta["completed_episodes"]
    conditions = [(m, b) for m in ["recent", "quest"] for b in [.25, .4, .6]]
    assert keys <= {(s, m, b) for s in range(100000, 100050) for m, b in conditions}
    common = [s for s in range(100000, 100050) if all((s, m, b) in keys for m, b in conditions)]

    def aggregate(data, method, ratio):
        group = [r for r in data if r["method"] == method and r["ratio"] == ratio]
        count = len(group)
        correct = sum(round(r["reward"] * 4) for r in group)
        protocol = sum((Counter(r.get("protocol_counts", {})) for r in group), Counter())
        out = dict(method=method, ratio=ratio, worlds=count, correct=correct, decisions=4 * count,
                   accuracy=correct / (4 * count) if count else None,
                   perfect_worlds=sum(r["reward"] == 1 for r in group))
        for field in ["invalid", "kv_errors", "kv_in", "kv_out", "notes", "forced_evictions", "truncated_tokens", "over_budget"]:
            out[field] = sum(r.get(field, 0) for r in group)
        out.update(environment_calls=protocol["environment_calls"],
                   environment_errors=protocol["environment_errors"],
                   environment_error_rate=protocol["environment_errors"] / protocol["environment_calls"] if protocol["environment_calls"] else None,
                   parsed_memory_calls=protocol["memory_calls"])
        # SFT's historical summary lacks this separate protocol replay counter.
        if method == "model":
            for field in ["environment_calls", "environment_errors", "environment_error_rate", "parsed_memory_calls"]:
                out[field] = None
        return out

    all_available = [aggregate(baseline, m, b) for m, b in conditions]
    paired = [aggregate([r for r in baseline if r["seed"] in common], m, b) for m, b in conditions]
    paired += [aggregate([r for r in sft if r["seed"] in common], "model", b) for b in [.25, .4, .6]]
    verification = dict(status="PARTIAL_SNAPSHOT_ACCOUNTING_VERIFIED_NOT_FINAL_BASELINE_ACCEPTANCE",
                        completed_episodes=len(baseline), target_episodes=300,
                        common_worlds=common, paired_decisions_per_condition=4 * len(common),
                        raw_hashes_answers_costs_and_world_pairing_verified=True,
                        model_memory_calls=sum(r.get("protocol_counts", {}).get("memory_calls", 0) for r in baseline),
                        capture_time_shanghai=meta["capture_time_shanghai"],
                        caveat="Base task/protocol failures confound attribution to the KV policy. The disabled-tool execution bug remains in source.")
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    for name, table in [("all_completed", all_available), ("paired_common_worlds", paired), ("decisions", decisions)]:
        (out / (name + ".json")).write_text(json.dumps(table, indent=2) + "\n")
        with (out / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    for r in paired:
        print(r["method"], r["ratio"], f'{r["correct"]}/{r["decisions"]}', r["environment_error_rate"])


if __name__ == "__main__":
    main()
