"""eval/t1_audit.py: the recency shortcut and the trajectory audit (model-free, toy tokenizer)."""
import json

from self_paging.envs.t1_warehouse import T1Config, generate, rule_read
from self_paging.eval.t1_audit import recency_reader, shortcut, trajectories


def test_recency_reader_assumes_unseen_shelves_occupied():
    text = "[step 1] x\n    ACTION: STORE I-1 -> S7\n    RESULT: OK"
    assert rule_read([text]) == 1 and recency_reader([text]) == 8
    out = shortcut(range(100000, 100030), T1Config())
    assert set(out) <= {"explicit", "implicit", "delayed"} and all(0 <= v["last5"] <= 1 for v in out.values())


def test_trajectory_audit_on_an_oracle_episode(tmp_path):
    from self_paging.agent.episode import EpisodeConfig, ScriptedPolicy, run_episode
    from self_paging.core.log import TokenLog
    from self_paging.data.build_t1_sft import t1_budget
    from self_paging.envs.t1_warehouse import WarehouseEnv
    from self_paging.oracle.t1_oracle import T1Oracle
    from self_paging.train.common import load_tokenizer

    tok = load_tokenizer("toy")
    env = WarehouseEnv(100003, T1Config())
    b = t1_budget(env, tok, 0.4)
    rec = run_episode(env, TokenLog(b), ScriptedPolicy(tok, T1Oracle("paging", seed=1)), tok,
                      EpisodeConfig(budget=b, max_turns=120))
    rec.update(seed=100003, ratio=0.4)
    path = tmp_path / "100003-model-0.4.trajectory.json"
    path.write_text(json.dumps(rec))
    out = trajectories([str(path)], T1Config())
    n = sum(out["decisions"].values())
    assert n == len(generate(100003, T1Config()).decisions)
    # the oracle answers from a sufficient resident set every time
    assert all(k.endswith("sufficient/right") and "insufficient" not in k for k in out["decisions"])
    assert out["fill_median_turns_with_kv_out"]["0.4"] > out["fill_median_other_turns"]["0.4"]
