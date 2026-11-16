import copy
import pytest
from self_paging.envs.longmemeval import LongMemEvalEnv, validate_item
from self_paging.envs.longmemeval_judge import get_anscheck_prompt


def item():
    return dict(question_id="test", question_type="single-session-user", question="Which city?",
                answer="SECRET_REFERENCE", question_date="2025/01/03",
                haystack_session_ids=["s1", "s2"], haystack_dates=["2025/01/01", "2025/01/02"],
                answer_session_ids=["s2"], haystack_sessions=[
                    [dict(role="user", content="I visited Rome.", has_answer=False)],
                    [dict(role="user", content="I moved to Oslo.", has_answer=True)]])


def test_question_and_annotations_not_disclosed_during_history():
    env = LongMemEvalEnv(item(), lambda *_: True)
    assert "Which city?" not in env.system + env.task
    assert env.step("answer", {"answer": "premature"})[0].startswith("ERROR")
    for i in range(2):
        text, meta, done = env.step("next_session", {})
        assert "Which city?" not in text and "SECRET_REFERENCE" not in text
        assert "has_answer" not in text and "answer_session_ids" not in text
        assert meta == {"env_page": i} and not done
    text, meta, done = env.step("next_session", {})
    assert "Which city?" in text and meta == {"env_page": 2} and not done
    assert "SECRET_REFERENCE" not in text
    assert env.evidence_pages() == [1]


def test_question_first_uses_identical_sessions():
    a = LongMemEvalEnv(item(), lambda *_: False)
    b = LongMemEvalEnv(item(), lambda *_: False, question_first=True)
    assert "Which city?" in b.task
    for _ in range(3):
        assert a.step("next_session", {}) == b.step("next_session", {})


def test_judge_only_called_once_after_answer():
    calls = []
    def judge(it, response):
        calls.append((it["question_id"], response)); return True
    env = LongMemEvalEnv(item(), judge)
    assert env.score() == 0 and not calls
    for _ in range(3): env.step("next_session", {})
    assert not env.step("answer", {"answer": ""})[2]
    assert env.step("answer", {"answer": "Oslo"})[2]
    assert env.score() == env.score() == 1 and calls == [("test", "Oslo")]


def test_gold_annotation_cannot_change_observations():
    x = item(); y = copy.deepcopy(x)
    y["answer"] = "DIFFERENT_REFERENCE";y["answer_session_ids"] = ["s1"]
    a, b = LongMemEvalEnv(x, None), LongMemEvalEnv(y, None)
    assert (a.system, a.task, a.tools) == (b.system, b.task, b.tools)
    for _ in range(3): assert a.step("next_session", {}) == b.step("next_session", {})


def test_invalid_alignment_and_missing_evidence_fail():
    x = item();x["haystack_dates"].pop()
    with pytest.raises(ValueError): validate_item(x)
    x = item();x["answer_session_ids"] = ["missing"]
    with pytest.raises(ValueError): validate_item(x)


def test_identical_official_distractor_duplicates_are_preserved():
    x = item()
    x["haystack_sessions"].append(copy.deepcopy(x["haystack_sessions"][0]))
    x["haystack_session_ids"].append("s1");x["haystack_dates"].append(x["haystack_dates"][0])
    env = LongMemEvalEnv(x, None)
    pages = [env.step("next_session", {}) for _ in range(3)]
    assert pages[0][0] == pages[2][0]
    assert [p[1]["env_page"] for p in pages] == [0, 1, 2]
    x["haystack_sessions"][-1][0]["content"] = "conflict"
    with pytest.raises(ValueError): validate_item(x)


def test_official_question_type_specific_judge_prompts():
    assert "off-by-one" in get_anscheck_prompt("temporal-reasoning", "q", "a", "r")
    assert "updated answer" in get_anscheck_prompt("knowledge-update", "q", "a", "r")
    assert "Rubric:" in get_anscheck_prompt("single-session-preference", "q", "a", "r")
    assert "unanswerable" in get_anscheck_prompt("anything", "q", "a", "r", abstention=True)
