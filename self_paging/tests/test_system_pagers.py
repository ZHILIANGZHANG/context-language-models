import torch

from self_paging.agent.episode import EpisodeConfig,EpisodeState
from self_paging.agent.pagers import QuestPagePager,AttnEvictPager
from self_paging.hf.logstore import PagedKV
from self_paging.hf.observed import ObservedPagedKV
from self_paging.train.common import tiny_qwen3


def populate(log):
    a=log.open('system',pinned=True);log.feed(a,[1,2,3])
    b=log.open('tool');log.feed(b,[4,5,6,7])
    c=log.open('tool');log.feed(c,[8,9,10,11])
    log.page_out([b])
    d=log.open('assistant');out=log.feed(d,[12,13])
    return a,b,c,d,out


def test_query_telemetry_preserves_logits_and_attention_mass():
    model=tiny_qwen3(40).eval()
    reference=PagedKV(model)
    observed=ObservedPagedKV(model,attention_mass=True,window=3)
    *_,expected=populate(reference)
    a,b,c,d,actual=populate(observed)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert len(observed.attention_window)==3
    for row in observed.attention_window:
        assert abs(sum(row.values())-1)<1e-6
    # The final two queries could not attend to the paged-out record.
    assert all(b not in row for row in list(observed.attention_window)[-2:])
    assert set(observed.keymean_scores())=={a,b,c,d}
    assert all(torch.isfinite(q).all() for q in observed.recent_queries.values())


def test_system_pagers_retrieval_vs_irreversible_eviction():
    model=tiny_qwen3(40).eval()
    log=ObservedPagedKV(model,budget=14,attention_mass=True)
    a,b,c,d,_=populate(log)
    st=EpisodeState(None,log,None,EpisodeConfig(budget=14,struct_reserve=0,min_gen_room=0))
    log.keymean_scores=lambda:{a:0,b:10,c:-10,d:0}
    QuestPagePager()(st,5)
    assert log.table.pages[b].where=='gpu' and log.table.pages[c].where=='host'
    assert log.table.pages[a].where==log.table.pages[d].where=='gpu'
    log.attention_scores=lambda:{a:1,b:0,d:1}
    AttnEvictPager()(st,10)
    assert log.table.pages[b].where=='dropped'


def test_backing_store_owns_independent_storage_and_replays_exactly():
    model=tiny_qwen3(40).eval()
    a=PagedKV(model)
    b=PagedKV(model,store_device='cpu')
    aa=populate(a);bb=populate(b)
    torch.testing.assert_close(aa[-1],bb[-1],rtol=0,atol=0)
    for _,cache in b.kv.items():
        for layer,(k,v) in enumerate(cache):
            assert k.device.type==v.device.type=='cpu'
            assert k.untyped_storage().data_ptr()!=b._live.layers[layer].keys.untyped_storage().data_ptr()
    for log,record in [(a,aa),(b,bb)]:
        log.page_in([record[1]])
    p=a.open('assistant');q=b.open('assistant')
    torch.testing.assert_close(a.feed(p,[14]),b.feed(q,[14]),rtol=0,atol=0)


def test_recorded_mass_matches_native_eager_attention_with_position_gaps():
    model=tiny_qwen3(40,attn='eager').eval()
    captured={}
    handles=[]
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            lambda module,args,output,i=i: captured.__setitem__(i,output[1].detach())))
    observed=ObservedPagedKV(model,attention_mass=True,window=3)
    try:
        a,b,c,d,_=populate(observed)
    finally:
        for handle in handles:handle.remove()
    expected=sum(w.float().mean((0,1)) for w in captured.values())/len(captured)
    actual=list(observed.attention_window)[-2:]
    for j,row in enumerate(actual):
        for pid,lo,hi in [(a,0,3),(c,3,7),(d,7,9)]:
            assert abs(row[pid]-float(expected[j,lo:hi].sum()))<1e-6


def test_chunked_cpu_backing_preserves_global_positions_and_final_logits():
    model=tiny_qwen3(40).eval()
    ref=PagedKV(model)
    bounded=PagedKV(model,store_device='cpu',prefill_chunk=2,last_logits_only=True)
    a=populate(ref);b=populate(bounded)
    torch.testing.assert_close(a[-1][-1],b[-1][-1],rtol=1e-5,atol=1e-6)
    assert bounded.tokens==ref.tokens and bounded.page_of==ref.page_of
    for log,record in [(ref,a),(bounded,b)]:log.page_in([record[1]])
    p=ref.open('assistant');q=bounded.open('assistant')
    torch.testing.assert_close(ref.feed(p,[14,15,16,17])[-1],bounded.feed(q,[14,15,16,17])[-1],rtol=1e-5,atol=1e-6)
