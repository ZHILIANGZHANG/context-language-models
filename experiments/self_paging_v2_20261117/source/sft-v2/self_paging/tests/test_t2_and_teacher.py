"""BrowseComp-Plus environment logic with a fake searcher and judge, and the hinted teacher."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from self_paging.agent.episode import EpisodeConfig, ScriptedPolicy, run_episode  # noqa: E402
from self_paging.core.log import TokenLog  # noqa: E402
from self_paging.core.protocol import Tok, render_tool_call  # noqa: E402
from self_paging.data.reject_sample import accept  # noqa: E402
from self_paging.envs.t1_warehouse import T1Config, WarehouseEnv  # noqa: E402
from self_paging.envs.t2_bcp import BCPEnv, parse_judgement  # noqa: E402
from self_paging.oracle.hinted_teacher import HintedTeacher  # noqa: E402
from self_paging.oracle.t1_oracle import T1Oracle  # noqa: E402
from self_paging.testing.toy_tokenizer import ToyTokenizer  # noqa: E402
from self_paging.train.common import tiny_qwen3  # noqa: E402

TOK = Tok(ToyTokenizer())


class FakeSearcher:
    docs = {"d1": "The bridge was built in 1932 by Ana Ruiz. " * 20, "d2": "Unrelated text. " * 50}

    def search(self, q):
        return [("d1", self.docs["d1"]), ("d2", self.docs["d2"])]

    def doc(self, docid):
        return self.docs.get(docid)


def judge(question, response, correct):
    return correct.lower() in response.lower()


def scripted_bcp(st):
    turn = st.turn
    if turn == 0:
        return render_tool_call("search", {"query": "bridge builder"})
    if turn == 1:
        tool_pages = [p for p, pg in st.log.table.pages.items() if pg.kind == "tool"]
        return (render_tool_call("kv_out", {"pages": str(tool_pages[0])}) + "\n"
                + render_tool_call("get_document", {"docid": "d1"}))
    return render_tool_call("answer", {"answer": "Ana Ruiz"})


def test_bcp_episode_with_fakes():
    env = BCPEnv({"question": "Who built the bridge?", "answer": "Ana Ruiz"}, FakeSearcher(), judge, TOK,
                 snippet_new=40, snippet_seen=10, doc_max=60)
    rec = run_episode(env, TokenLog(None), ScriptedPolicy(TOK, scripted_bcp), TOK, EpisodeConfig(max_turns=5))
    assert rec["reward"] == 1.0
    assert rec["meta"]["kv_out"] == 1 and rec["meta"]["invalid"] == 0
    assert sum(rec["ctrl_mask"]) > 0
    text = TOK.decode(rec["tokens"])
    assert "docid=d1" in text and "Answer recorded." in text


def test_bcp_multiple_candidates_are_wrong():
    env = BCPEnv({"question": "q", "answer": "Ana Ruiz"}, FakeSearcher(), judge, TOK)
    env.step("answer", {"answer": "Ana Ruiz or Bo Li"})
    assert env.score() == 0.0


def test_bcp_judge_failures_are_not_scored_wrong_and_answer_is_terminal():
    with pytest.raises(ValueError):
        parse_judgement("Service unavailable")
    with pytest.raises(ValueError):
        parse_judgement("correct: yes\ncorrect: no")
    assert parse_judgement("reasoning: equivalent\ncorrect: yes\nconfidence: 100")
    assert not parse_judgement("correct: no")
    calls=[]
    env=BCPEnv({"question":"q","answer":"a"},FakeSearcher(),
               lambda *args: calls.append(args) or True,TOK)
    env.step("answer",{"answer":""})
    assert not env.done
    env.step("answer",{"answer":"a"})
    env.step("answer",{"answer":"different"})
    assert env.final=="a" and env.score()==env.score()==1.
    assert len(calls)==1


def test_hinted_teacher_keeps_oracle_calls():
    model = tiny_qwen3(TOK.vocab_size).eval()
    env = WarehouseEnv(0, T1Config(n_decisions=1, warmup=4, gap=(2, 3), events_per_page=(2, 3)))
    oracle = T1Oracle("paging")
    teacher = HintedTeacher(model, TOK, oracle, max_new_tokens=8)
    rec = run_episode(env, TokenLog(None), ScriptedPolicy(TOK, teacher), TOK, EpisodeConfig(max_turns=40))
    assert rec["reward"] == 1.0  # tool calls come from the oracle, prose from the (random) teacher
    assert accept(rec)


def test_privileged_hint_does_not_nest_user_inside_unfinished_assistant():
    from self_paging.core.protocol import IM_START,IM_END,NO_THINK
    model=tiny_qwen3(TOK.vocab_size).eval()
    prompts=[]
    def generate(ids,**kwargs):
        text=TOK.decode(ids[0].tolist());prompts.append(text)
        split=text.rindex(IM_START+'user\n[Private note')
        assert text[:split].endswith(IM_END+'\n')
        assert not text[:split].endswith(NO_THINK)
        suffix=TOK.encode('I will follow the current warehouse events.')+[TOK.im_end]
        return torch.cat([ids,torch.tensor([suffix],device=ids.device)],dim=1)
    model.generate=generate
    env=WarehouseEnv(0,T1Config(n_decisions=1,warmup=4,gap=(2,3),events_per_page=(2,3)))
    teacher=HintedTeacher(model,TOK,T1Oracle('paging'))
    rec=run_episode(env,TokenLog(None),ScriptedPolicy(TOK,teacher),TOK,EpisodeConfig(max_turns=40))
    assert rec['reward']==1 and prompts
    assert '[Private note' not in TOK.decode(rec['tokens'])
