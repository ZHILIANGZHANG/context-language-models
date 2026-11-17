"""Matched T1 comparison using compact StreamingLLM/H2O KV and frozen SFT weights.

System arms receive task tools only and no paging instructions/dashboard. Environment
pages, digest contents, decision order and canonical absolute budgets are identical.
Full retains all history; the automatic algorithms never replay dropped tokens.
The model arm uses the original paging runner and pays for its protocol overhead.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from ..agent.episode import EpisodeConfig, ModelPolicy, run_episode
from ..core.protocol import (KV_TOOLS, render_system, render_user, assistant_prefix, parse_tool_calls,
    block_open, block_close, response_header, response_trailer, escape)
from ..envs.t1_warehouse import WarehouseEnv, T1Config
from ..data.build_t1_sft import t1_budget
from ..hf.faithful_cache import FaithfulKV
from ..hf.logstore import PagedKV
from ..train.common import load_model, load_tokenizer


def system_episode(env, log, tok, max_turns=60, max_new=1024, max_log=32768, *, latency=None):
    meta = dict(invalid=0, unoffered_kv_calls=0, kv_in=0, kv_out=0, notes=0,
                forced_evictions=0, truncated_tokens=0, over_budget=0, turns=0, env_calls=[])
    turns, env_pages, generated, decode_residency = [], {}, [], []
    def prefill(pid, ids):
        return log.feed(pid, ids) if latency is None else latency.feed(log, pid, ids)
    def feed(kind, text):
        pid = log.open(kind)
        ids = tok.encode(text) if isinstance(text,str) else text
        if log.log_len+len(ids) > max_log:
            raise ValueError('Absolute token cap exceeded')
        return pid, prefill(pid, ids)
    feed('system', render_system(env.system_for_tools(False), list(env.tools)))
    feed('task', render_user(env.task))
    for turn in range(max_turns):
        meta['turns'] = turn+1
        if log.log_len + max_new + 64 > max_log:
            meta['log_limit_reached'] = 1
            break
        pid, result = feed('assistant', assistant_prefix(False))
        ids=[]
        if latency is not None:
            latency.begin_decode()
        for _ in range(max_new):
            t = int(result[-1].argmax())
            if not ids and latency is not None:
                latency.first_token()
            ids.append(t)
            generated.append(log.log_len)
            result = log.feed(pid, [t])
            decode_residency.append(log.feed_stats[-1][1] if isinstance(log,FaithfulKV) else log.resident_tokens())
            if t == tok.im_end:
                break
        if latency is not None:
            latency.end_decode(len(ids))
        if not ids or ids[-1] != tok.im_end:
            prefill(pid, [tok.im_end])
            meta['generation_cap_hits'] = meta.get('generation_cap_hits',0)+1
        prefill(pid, tok.encode('\n'))
        text = tok.decode(ids)
        calls, _, bad = parse_tool_calls(text)
        meta['invalid'] += bad + int(not calls)
        replies=[]
        for c in calls:
            if c['name'] not in {x['function']['name'] for x in env.tools}:
                meta['invalid'] += 1
                meta['unoffered_kv_calls'] += int(c['name'] in KV_TOOLS)
                replies.append(('tool='+c['name'], 'ERROR: unknown tool. Use only the listed task tools.', None))
                continue
            pending = getattr(env,'pending',None)
            body, info, done = env.step(c['name'], c['arguments'])
            meta['env_calls'].append(dict(turn=turn, name=c['name'], args=c['arguments'],
                                          decision_page=getattr(pending,'page',None)))
            label='tool='+c['name']+(f" | {info['digest']}" if info.get('digest') else '')
            replies.append((label,body,info.get('env_page')))
        turns.append(dict(turn=turn,output=text,calls=calls,resident=log.resident_tokens()))
        if env.done or meta['invalid']>=5:
            break
        if not replies:
            replies=[('error','ERROR: no valid tool call. Call a listed task tool.',None)]
        required = sum(len(tok.encode(body))+256 for _,body,_ in replies)
        if log.log_len+required+64>max_log:
            meta['log_limit_reached']=1
            break
        feed('header',block_open())
        for label,body,env_page in replies:
            # Same digest and wrapper content as the autonomous arm. Page ids are local log ids.
            h, b, tail = tok.encode_wrapped(response_header(log.next_pid+1,label), escape(body) or ' ',response_trailer())
            feed('header',h)
            p,_=feed('tool' if env_page is not None else 'ack',b)
            if env_page is not None: env_pages[str(env_page)]=p
            feed('header',tail)
        feed('header',block_close())
    if latency is not None:
        latency.finish()
    rec=log.record()
    rec.update(meta=meta,turns=turns,env_page=env_pages,reward=float(env.score()),
               budget=log.budget,peak_resident=log.peak_resident,
               generated_positions=generated,decode_residency=decode_residency)
    if log.budget is not None:
        assert log.peak_resident <= log.budget
    return rec


def atomic(path, value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value)+'\n');temp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--tokenizer',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--seeds',nargs=2,type=int,required=True)
    ap.add_argument('--methods',nargs='+',choices=['full','model','streaming','h2o'],required=True)
    ap.add_argument('--ratios',nargs='+',type=float,default=[.25,.35,.4,.5,.6])
    ap.add_argument('--prefill-evict',choices=('token','space'),default='token',
                    help="StreamingLLM/H2O prefill when full: 'token' (one token per forward) or 'space' (evict-for-space per chunk); use a new --out")
    a=ap.parse_args(); torch.manual_seed(0)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    tok=load_tokenizer(a.tokenizer)
    model=load_model(a.model,'sdpa',device='cuda').eval()
    for seed in range(*a.seeds):
        for method in a.methods:
            for ratio in ([1.] if method=='full' else a.ratios):
                name=f'{seed}-{method}-{ratio}'; dst=out/(name+'.json'); raw=out/(name+'.trajectory.json')
                if dst.exists():
                    old=json.loads(dst.read_text());assert hashlib.sha256(raw.read_bytes()).hexdigest()==old['trajectory_sha256']
                    assert method in ('full','model') or old.get('prefill_evict','token')==a.prefill_evict,'Mixed prefill modes in one --out'
                    continue
                assert not raw.exists(),'Uncommitted raw output; inspect before resume'
                env=WarehouseEnv(seed,T1Config(n_decisions=4)); budget=t1_budget(env,tok,ratio)
                started=time.time()
                if method=='model':
                    log=PagedKV(model,budget=budget,strict=False)
                    rec=run_episode(env,log,ModelPolicy(tok,temperature=0),tok,
                        EpisodeConfig(budget=budget,max_turns=60,max_new_tokens=1024,max_log_tokens=32768))
                else:
                    log=PagedKV(model,budget=None,strict=False) if method=='full' else FaithfulKV(model,method,budget,prefill=a.prefill_evict)
                    try: rec=system_episode(env,log,tok)
                    finally:
                        if isinstance(log,FaithfulKV):log.close()
                decisions=[dict(page=d.page,kind=d.kind,gold=d.answer,predicted=env.answers[d.page]) for d in env.ep.decisions]
                rec.update(seed=seed,method=method,ratio=ratio,decisions=decisions,environment_pages=env.ep.pages)
                assert len(decisions)==4 and rec['reward']==sum(d['gold']==d['predicted'] for d in decisions)/4
                assert len(rec['tokens'])<=32768 and not rec['meta']['over_budget']
                atomic(raw,rec)
                row=dict(seed=seed,method=method,ratio=ratio,budget=budget,reward=rec['reward'],
                    peak_resident=rec['peak_resident'],tokens=len(rec['tokens']),meta=rec['meta'],
                    trajectory_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),seconds=time.time()-started)
                if method in ('streaming','h2o') and a.prefill_evict!='token':row['prefill_evict']=a.prefill_evict
                if method!='model':row['mean_resident_decode']=sum(rec['decode_residency'])/max(1,len(rec['decode_residency']))
                else:
                    from ..core.costs import episode_cost
                    row['mean_resident_decode']=episode_cost(rec,batch=1)['mean_resident_decode']
                atomic(dst,row);print(json.dumps(row),flush=True)
                del rec,log
                torch.cuda.empty_cache()

if __name__=='__main__':main()
