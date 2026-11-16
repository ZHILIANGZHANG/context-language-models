import pytest
import torch

from self_paging.agent.episode import EpisodeConfig, EpisodeState, ScriptedPolicy, _lru_room, run_episode
from self_paging.core.log import TokenLog
from self_paging.core.protocol import Tok, render_tool_call
from self_paging.envs.t2_bcp import BCPEnv
from self_paging.eval.long_eval import DeferredJudge, make_result, validate_items
from self_paging.eval.judge_results import parse_answer, pass_at_k, summarize
from self_paging.testing.toy_tokenizer import ToyTokenizer
from self_paging.hf.logstore import PagedKV
from self_paging.train.batch import pack,token_logprobs
from self_paging.train.common import tiny_qwen3


class Searcher:
    def search(self, query):
        return [('a', 'x' * 5000)]


def test_absolute_log_cap_requests_final_answer_without_exposing_omitted_document():
    tok = Tok(ToyTokenizer())
    judge = DeferredJudge()
    item = dict(id=1, question='What is the code?', answer='abc')
    env = BCPEnv(item, Searcher(), judge, tok, snippet_new=5000)
    cfg = EpisodeConfig(max_new_tokens=200, max_log_tokens=5000, answer_reserve=1000)
    def policy(st):
        if st.meta.get('log_limit_notice'):
            assert 'x' * 100 not in tok.decode(st.log.tokens)
            return render_tool_call('answer', {'answer': 'abc'})
        return render_tool_call('search', {'query': 'code'})
    rec = run_episode(env, TokenLog(), ScriptedPolicy(tok, policy), tok, cfg)
    assert len(rec['tokens']) <= 5000
    assert rec['meta']['log_limit_notice'] == rec['meta']['log_limit_reached'] == 1
    assert rec['env_page'] == {} and env.final == 'abc'
    row = make_result('bcp', item, env, rec, judge, cfg)
    assert row['reward'] is None and row['judge_status'] == 'pending'
    assert rec['reward'] is None
    assert len(rec['tokens']) == len(rec['loss_mask']) == len(rec['page_of'])


def test_duplicate_question_ids_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        validate_items([{'id': 1}, {'id': '1'}], 'bcp')


def test_judgement_and_pass_at_k_reject_missing_or_ambiguous_evidence():
    assert parse_answer('bcp', 'reasoning: same\ncorrect: yes\nconfidence: 100')
    assert not parse_answer('longmemeval', '<think>reasoning</think>No.')
    with pytest.raises(ValueError):
        parse_answer('bcp', 'correct: yes\ncorrect: no')
    assert pass_at_k(8, 2, 1) == .25 and pass_at_k(8, 2, 8) == 1
    row = dict(dataset='bcp', method='full', budget=None, question_id='x', sample=0, reward=None)
    with pytest.raises(ValueError, match='pending'):
        summarize([row])
    rows = [row | {'sample': i, 'reward': float(i<2)} for i in range(8)]
    assert summarize(rows)[0]['pass_at_k'] == {'1': .25, '8': 1}
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(rows + [rows[0]])


def test_lru_reloaded_old_page_is_newer_than_unaccessed_page():
    tok=Tok(ToyTokenizer())
    log=TokenLog(budget=50)
    pinned=log.open('system',pinned=True);log.feed(pinned,[1]*10)
    old=log.open('tool');log.feed(old,[1]*10)
    other=log.open('tool');log.feed(other,[1]*10)
    current=log.open('assistant');log.feed(current,[1]*5)
    st=EpisodeState(None,log,tok,EpisodeConfig(budget=50,overflow='lru'))
    st.last_access[old]=log.log_len
    assert _lru_room(st,20)
    assert log.table.pages[other].where=='host'
    assert log.table.pages[old].where=='gpu'
    assert log.table.pages[pinned].where==log.table.pages[current].where=='gpu'
    assert st.meta['lru_evictions']==1


def test_thought_pages_do_not_execute_planned_tool_calls():
    tok=Tok(ToyTokenizer())
    judge=DeferredJudge()
    class NeverSearch:
        def search(self,*args):raise AssertionError('A thought was executed as an action')
    env=BCPEnv(dict(id=1,question='What?',answer='abc'),NeverSearch(),judge,tok)
    text='<think>I could use '+render_tool_call('search',{'query':'wrong'})+'</think>\n'+render_tool_call('answer',{'answer':'abc'})
    rec=run_episode(env,TokenLog(),ScriptedPolicy(tok,lambda st:text),tok,EpisodeConfig(thinking=True))
    assert env.final=='abc' and len(judge.calls)==1
    thought=[p for p in rec['pages'] if p['kind']=='think']
    assert len(thought)==1
    p=thought[0]
    raw=tok.decode(rec['tokens'][p['start']:p['start']+p['length']])
    assert raw.startswith('<think>') and raw.endswith('</think>')
    assert sum(rec['ctrl_mask'][p['start']:p['start']+p['length']])==0
    assert rec['meta']['invalid']==0


def test_thought_split_logprobs_match_single_pass_visibility():
    tok=Tok(ToyTokenizer());model=tiny_qwen3(tok.vocab_size).eval()
    env=BCPEnv(dict(id=1,question='What?',answer='abc'),Searcher(),DeferredJudge(),tok)
    text='<think>The evidence gives abc.</think>\n'+render_tool_call('answer',{'answer':'abc'})
    rec=run_episode(env,PagedKV(model),ScriptedPolicy(tok,lambda st:text),tok,EpisodeConfig(thinking=True))
    p=pack([rec]);lp=token_logprobs(model,p,attn='sdpa')
    selected=p.loss_mask.bool()
    torch.testing.assert_close(lp[selected],torch.tensor(rec['logprobs'])[selected],rtol=1e-4,atol=1e-4)
