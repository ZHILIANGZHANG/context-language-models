import json
import os
import pytest
from self_paging.data.rl_preparation import (config, decode_config, describe, reference,
    clean_reference, split_audit, select_training, select_profile)
from self_paging.train.common import load_tokenizer
from self_paging.envs.t1_warehouse import T1Config

@pytest.fixture(scope='module')
def tok():
    path = os.environ.get('QWEN3_TOKENIZER', '/nas/Public/zhangcheng/models/Qwen3-8B')
    if not os.path.isdir(path):
        pytest.skip('needs the real Qwen3 tokenizer (set QWEN3_TOKENIZER)')
    return load_tokenizer(path)

def test_split_and_determinism(tok):
    assert split_audit()['seed_overlap']==0
    a,c=describe(10000,'train',tok);b,_=describe(10000,'train',tok)
    assert a==b and decode_config(a['t1_config'])==c
    assert a['max_turns']>a['n_pages']+a['n_decisions']

@pytest.mark.parametrize('seed',[10000,10001,10002])
def test_reference_and_budget(seed,tok):
    cfg=T1Config(n_decisions=2,warmup=6,gap=(3,5),events_per_page=(3,6))
    full=reference(seed,cfg,tok,1.,'full',80)
    assert clean_reference(full)
    for method in ['recent','belady']:
        r=reference(seed,cfg,tok,.6,method,80)
        assert 0<=r['reward']<=1 and r['peak_resident']<=r['budget']
        assert len(r['tokens'])==len(r['page_of'])==len(r['loss_mask'])
    # A failing heuristic must never be classified as a mathematical impossibility.
    assert not clean_reference(full|{'reward':.5})
    assert not clean_reference(full|{'meta':full['meta']|{'forced_evictions':1}})

def test_quota_selection():
    rows=[dict(split='train',ratio=r,n_decisions=n,prompt_id=f'{r}-{n}-{i}',train_eligible=True,sensitive=i%2==0)
          for r in [1.,.6,.4,.25] for n in range(2,7) for i in range(40)]
    s,g=select_training(rows,200)
    assert not any(g.values()) and len(s)==200
    assert sum(r['ratio']==1. for r in s)==20
    assert len({r['prompt_id'] for r in select_profile(s)})==64
    t,g=select_training([r for r in rows if r['ratio']!=.25],200)
    assert g['0.25']==60 and len(t)==140

def test_full_length_guard_not_silent_truncation(tok):
    a,c=describe(210000,'ood',tok)
    assert c.n_decisions>=7 and a['max_log_tokens']==32768
    assert a['page_tokens']==sum(a['page_lengths'])

def test_ood_v2_fits_the_32k_scope(tok):
    # v1 put the full-context reference over 32K on almost every world (1/300 supported)
    for seed in (210000, 210001):
        a, c = describe(seed, 'ood', tok)
        assert a['ood_version'] == 2 and a['n_decisions'] >= 7 and a['page_tokens'] <= 20000
        full = reference(seed, c, tok, 1., 'full', a['max_turns'])
        assert clean_reference(full) and len(full['tokens']) <= 32768
