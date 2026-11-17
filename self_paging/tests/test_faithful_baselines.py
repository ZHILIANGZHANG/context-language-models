"""Author-code witnesses and independent attention oracle; no score-order assertions."""
import ast
import importlib.util
import os
from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
from self_paging.hf.faithful_cache import FaithfulKV,retention_indices
from self_paging.hf.logstore import PagedKV
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config,SYSTEM


def reference_classes():
    if 'FAITHFUL_REFERENCES' not in os.environ:
        pytest.skip('needs the pinned authors repositories (FAITHFUL_REFERENCES)')
    root=Path(os.environ['FAITHFUL_REFERENCES'])
    spec=importlib.util.spec_from_file_location('authors_streaming',root/'streaming/streaming_llm/kv_cache.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    source=ast.parse((root/'h2o/h2o_hf/utils_real_drop/modify_llama.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='H2OKVCache_LayerWise')
    scope={'torch':torch};exec(compile(ast.Module(body=[cls],type_ignores=[]),'author_h2o_class','exec'),scope)
    return mod.StartRecentKVCache,scope['H2OKVCache_LayerWise']


def small(groups=2):
    torch.manual_seed(71)
    cfg=Qwen3Config(vocab_size=79,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                    num_attention_heads=2*groups,num_key_value_heads=2,head_dim=8,
                    max_position_embeddings=1024,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return Qwen3ForCausalLM(cfg).eval()


def test_prompt_contract():
    env=WarehouseEnv(100000,T1Config(n_decisions=4))
    assert env.system_for_tools(True)==SYSTEM==env.system
    prompt=env.system_for_tools(False)
    for x in ('kv_in','kv_out','kv_drop','note','[KV]'):assert x not in prompt
    assert 'next_events' in prompt and 'store' in prompt


@pytest.mark.parametrize('method',['streaming','h2o'])
def test_eviction_matches_author_code(method):
    S,H=reference_classes();torch.manual_seed(2)
    cap=15;official=S(4,cap-4) if method=='streaming' else H(cap//2,cap-cap//2)
    ks=vs=score=None;orig=None;heads=2
    for step in range(80):
        k=torch.randn(1,heads,1,8);v=torch.randn_like(k)
        ks=k if ks is None else torch.cat((ks,k),2);vs=v if vs is None else torch.cat((vs,v),2)
        weights=torch.rand(1,heads,1,ks.shape[-2]).softmax(-1)
        if method=='h2o':
            update=weights.sum(0).sum(1)
            if score is not None:update[:,:-1]+=score
            score=update
        idx=retention_indices(method,score,ks.shape[-2],cap,heads,ks.device)
        gi=idx[None,:,:,None].expand(1,heads,-1,8)
        if method=='streaming':ref=official([(ks,vs)])[0]
        else:ref=official((ks,vs),weights)
        ks,vs=ks.gather(2,gi),vs.gather(2,gi)
        assert torch.equal(ks,ref[0]) and torch.equal(vs,ref[1])
        if method=='h2o':
            score=score.gather(1,idx);assert torch.equal(score,official.hh_score)


@pytest.mark.parametrize('method',['streaming','h2o'])
@pytest.mark.parametrize('groups',[1,2])
def test_layer_against_author_cache_and_independent_attention(method,groups):
    S,H=reference_classes();model=small(groups);module=model.model.layers[0].self_attn
    log=FaithfulKV(model,method,16)
    refcache=None;positions=None
    official=S(4,11) if method=='streaming' else H(7,8)
    try:
        for t in range(45):
            hidden=torch.randn(1,1,32)
            shape=(1,1,-1,8)
            q=module.q_norm(module.q_proj(hidden).view(shape)).transpose(1,2)
            k=module.k_norm(module.k_proj(hidden).view(shape)).transpose(1,2)
            v=module.v_proj(hidden).view(shape).transpose(1,2)
            size=0 if refcache is None else refcache[0].shape[-2]
            p=torch.tensor([[size if method=='streaming' else t]])
            rope=model.model.rotary_emb(hidden,p)
            log.log_len=t
            log.native_mask=torch.zeros(1,1,1,size+1)
            log.stream_rope=model.model.rotary_emb(hidden,torch.arange(size+1)[None])
            actual,_=log._attention(0,module,hidden,rope)
            q,rotated=apply_rotary_pos_emb(q,k,*rope)
            if method=='h2o':k=rotated
            k=k if refcache is None else torch.cat((refcache[0],k),2)
            v=v if refcache is None else torch.cat((refcache[1],v),2)
            ak=k
            if method=='streaming':_,ak=apply_rotary_pos_emb(k,k,*log.stream_rope)
            w=(q@ak.repeat_interleave(groups,1).transpose(-1,-2)*module.scaling).softmax(-1)
            expected=module.o_proj((w@v.repeat_interleave(groups,1)).transpose(1,2).reshape(1,1,-1))
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            if method=='streaming':refcache=official([(k,v)])[0]
            else:refcache=official((k,v),w.view(1,2,groups,1,size+1).mean(2))
            torch.testing.assert_close(log.state[0][0],refcache[0],atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(log.state[0][1],refcache[1],atol=1e-6,rtol=1e-5)
            if method=='h2o':torch.testing.assert_close(log.state[0][2],official.hh_score,atol=1e-6,rtol=1e-5)
    finally:log.close()


@pytest.mark.parametrize('method',['streaming','h2o'])
def test_no_eviction_matches_native(method):
    model=small();ids=[list(range(10)),[10,11,12],[13]]
    native=PagedKV(model);p=native.open('tool');ref=[native.feed(p,x) for x in ids]
    log=FaithfulKV(model,method,100);p=log.open('tool')
    try:
        for x,y in zip(ids,ref):torch.testing.assert_close(log.feed(p,x),y,atol=2e-6,rtol=1e-5)
        assert log.evictions==0
    finally:log.close()


@pytest.mark.parametrize('method',['streaming','h2o'])
def test_chunking_budget_and_irreversibility(method):
    model=small();tokens=list(range(48))
    log=FaithfulKV(model,method,16,record_trace=True);p=log.open('tool')
    try:
        output=log.feed(p,tokens)
        assert output.shape==(48,79) and log.peak_resident==16
        assert all(s[0].shape[-2]<=15 for s in log.state)
        assert log.evictions==33
        previous={}
        for e in log.trace:
            key=e['layer'];now=[set(x) for x in e['positions']]
            if key in previous:
                old_at,old=previous[key]
                for a,b in zip(now,old):assert a.issubset(b|set(range(old_at,e['at'])))
            previous[key]=(e['at'],now)
        if method=='streaming':assert log.state[0][3][0,:4].tolist()==[0,1,2,3]
    finally:log.close()


@pytest.mark.parametrize('method',['streaming','h2o'])
def test_space_prefill_keeps_budget_and_needs_few_forwards(method):
    """prefill='space' (authors' evict_for_space): a saturated cache reads a long page in a few
    chunked forwards instead of one forward per token; the peak stays <= B and nothing returns."""
    model=small();tokens=list(range(70))*2
    runs={}
    for mode in ('token','space'):  # one FaithfulKV at a time: it patches the model's attention
        log=FaithfulKV(model,method,24,record_trace=True,prefill=mode,chunk=8)
        try:
            out=log.feed(log.open('tool'),tokens)
            assert out.shape==(140,79) and torch.isfinite(out).all()
            assert log.peak_resident<=24 and all(s[0].shape[-2]<=23 for s in log.state)
            p=log.open('assistant');log.feed(p,[5]);log.feed(p,[6])  # decoding: one token per forward
            assert log.peak_resident<=24
            runs[mode]=(len(log.feed_stats),log.trace,log.state[0][3].clone())
        finally:
            log.close()
    assert runs['space'][0]<=len(tokens)//8+4 < runs['token'][0]
    previous={}
    for e in runs['space'][1]:
        now=[set(x) for x in e['positions']]
        if e['layer'] in previous:
            old_at,old=previous[e['layer']]
            for a,b in zip(now,old):assert a.issubset(b|set(range(old_at,e['at'])))
        previous[e['layer']]=(e['at'],now)
    if method=='streaming':  # same retention rule: identical final slots
        assert torch.equal(runs['space'][2],runs['token'][2]) and runs['space'][2][0,:4].tolist()==[0,1,2,3]
