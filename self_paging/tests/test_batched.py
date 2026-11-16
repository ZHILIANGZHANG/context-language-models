import torch
import pytest

from self_paging.hf.batched import BatchCoordinator,BatchedLog
from self_paging.hf.logstore import PagedKV
from self_paging.train.common import tiny_qwen3


def scenario(log,offset):
    outputs=[]
    a=log.open('system',pinned=True);outputs.append(log.feed(a,[1,2,3+offset])[-1])
    b=log.open('tool');outputs.append(log.feed(b,list(range(4,8+offset)))[-1])
    c=log.open('assistant');outputs.append(log.feed(c,[10+offset,11])[-1])
    log.page_out([b]);d=log.open('tool');outputs.append(log.feed(d,[12,13,14])[-1])
    log.page_in([b]);e=log.open('assistant');outputs.append(log.feed(e,[15])[-1])
    return torch.stack(outputs),log.record()


def test_batched_private_caches_padding_and_global_positions_match_serial():
    model=tiny_qwen3(40).eval()
    reference=[scenario(PagedKV(model),i) for i in range(3)]
    coordinator=BatchCoordinator(model,max_batch=3,prefill_chunk=3,wait_seconds=.02)
    actual=coordinator.run([lambda c,i=i:scenario(BatchedLog(c),i) for i in range(3)])
    assert max(coordinator.batch_sizes)>1
    for (x,r),(y,s) in zip(reference,actual):
        torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)
        assert r['tokens']==s['tokens'] and r['page_of']==s['page_of']
        # Per-feed chunking can add no resident versions; padding is absent.
        assert r['snapshots']==s['snapshots']


def test_batch_worker_exception_does_not_deadlock_other_requests():
    model=tiny_qwen3(40).eval()
    coordinator=BatchCoordinator(model,max_batch=2)
    def fail(c):raise ValueError('intentional failure')
    with pytest.raises(ValueError,match='intentional'):
        coordinator.run([fail,lambda c:scenario(BatchedLog(c),0)])
    assert coordinator.aborted.is_set()


def test_batched_rollout_logprobs_match_masked_training_and_private_rng():
    from self_paging.core.protocol import Tok
    from self_paging.testing.toy_tokenizer import ToyTokenizer
    from self_paging.envs.t1_warehouse import T1Config
    from self_paging.train.grpo import GRPOConfig,batched_rollout_group
    from self_paging.train.batch import pack,token_logprobs
    tok=Tok(ToyTokenizer())
    model=tiny_qwen3(tok.vocab_size).eval()
    cfg=GRPOConfig(group=3,rollout_batch=3,max_turns=2,max_new_tokens=8,
                   t1=T1Config(n_decisions=1,warmup=2,gap=(1,2),events_per_page=(1,2)))
    records=batched_rollout_group(model,tok,5,.6,cfg,torch.Generator().manual_seed(42))
    assert len({r['meta']['member_seed'] for r in records})==3
    for rec in records:
        with torch.no_grad():lp=token_logprobs(model,pack([rec]),attn='sdpa')
        mask=torch.tensor(rec['loss_mask'],dtype=torch.bool)
        torch.testing.assert_close(lp[mask],torch.tensor(rec['logprobs'])[mask],rtol=1e-5,atol=3e-6)
    cfg.rollout_batch=1
    sequential=batched_rollout_group(model,tok,5,.6,cfg,torch.Generator().manual_seed(42))
    assert [r['tokens'] for r in records]==[r['tokens'] for r in sequential]
