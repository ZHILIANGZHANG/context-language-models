"""Greedy T1/T1b system-control evaluation (formal E0, distinct from sampled pilot).

Use identical seeds, ratios and source snapshot for all conditions. QuestPage
is the documented page-key approximation; AttnEvict uses actual attention mass
over the last 16 queries, not a key-similarity surrogate. No latency claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .long_eval import atomic_json,digest,source_digest
from ..agent.episode import EpisodeConfig,ModelPolicy,run_episode
from ..agent.pagers import RecentPager,BeladyPager,QuestPagePager,AttnEvictPager
from ..core.costs import episode_cost
from ..data.build_t1_sft import t1_budget
from ..envs.t1_warehouse import WarehouseEnv,T1Config
from ..envs.t1b_lookup import LookupEnv,LookupConfig
from ..hf.logstore import PagedKV
from ..hf.observed import ObservedPagedKV
from ..train.common import load_model,load_tokenizer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task',choices=['t1','t1b'],required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--tokenizer',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',type=int,nargs=2,default=[100000,100200])
    ap.add_argument('--methods',nargs='+',choices=['full','recent','belady','quest','attn_evict','model'],
                    default=['full','recent','belady','quest','attn_evict'])
    ap.add_argument('--ratios',type=float,nargs='+',default=[.25,.5])
    ap.add_argument('--n-decisions',type=int,default=4)
    ap.add_argument('--overflow',choices=['truncate','lru'],default='truncate')
    ap.add_argument('--irreversible',action='store_true')
    ap.add_argument('--no-notes',action='store_true')
    a=ap.parse_args()
    if a.seeds[0]>=a.seeds[1] or not all(0<r<1 for r in a.ratios):
        ap.error('Invalid seeds or ratios')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    settings=vars(a).copy()
    for key in ['out','seeds']:settings.pop(key)
    settings.update(source_sha256=source_digest(),temperature=0,max_turns=60,max_new_tokens=1024,
                    observation_window=16,log_max=32768,thinking=False)
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=settings:raise ValueError('Incompatible run')
    atomic_json(mp,settings)
    tok=load_tokenizer(a.tokenizer)
    model=load_model(a.model,'sdpa',device='cuda').eval()
    for seed in range(*a.seeds):
        for method in a.methods:
            for ratio in ([1.] if method=='full' else a.ratios):
                name=f'{seed}-{method}-{ratio}'
                rp,tp=out/(name+'.json'),out/(name+'.trajectory.json')
                if rp.exists():
                    row=json.loads(rp.read_text())
                    if not tp.exists() or hashlib.sha256(tp.read_bytes()).hexdigest()!=row['trajectory_sha256']:
                        raise ValueError('Corrupt completed trajectory')
                    continue
                if tp.exists():raise ValueError('Uncommitted trajectory; inspect before resume')
                env=(WarehouseEnv(seed,T1Config(n_decisions=a.n_decisions)) if a.task=='t1' else
                     LookupEnv(seed,LookupConfig(n_decisions=a.n_decisions)))
                budget=None if method=='full' else t1_budget(env,tok,ratio)
                cfg=EpisodeConfig(budget=budget,max_turns=60,max_new_tokens=1024,kv_tools=method=='model',
                                  overflow=a.overflow,notes=not a.no_notes,reversible=not a.irreversible)
                pager={'recent':RecentPager,'belady':BeladyPager,'quest':QuestPagePager,'attn_evict':AttnEvictPager}.get(method)
                log=(ObservedPagedKV(model,budget=budget,strict=False,attention_mass=method=='attn_evict')
                     if method in ('quest','attn_evict') else PagedKV(model,budget=budget,strict=False))
                rec=run_episode(env,log,ModelPolicy(tok,temperature=0),tok,cfg,pager=pager() if pager else None)
                if len(rec['tokens'])>32768 or rec['meta']['over_budget']:
                    atomic_json(out/(name+'.failed-evidence.json'),rec)
                    raise ValueError('Registered token or resident budget exceeded')
                rec['decisions']=[dict(page=d.page,kind=d.kind,gold=d.answer,predicted=env.answers[d.page]) for d in env.ep.decisions]
                rec['environment_pages']=env.ep.pages
                atomic_json(tp,rec)
                row=dict(task=a.task,seed=seed,method=method,ratio=ratio,budget=budget,reward=rec['reward'],
                         tokens=len(rec['tokens']),peak_resident=rec['peak_resident'],meta=rec['meta'],
                         cost_batch1=episode_cost(rec,batch=1),cost_batch32=episode_cost(rec,batch=32),
                         trajectory_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),settings_sha256=digest(settings))
                atomic_json(rp,row)
                print(json.dumps({k:row[k] for k in ['task','seed','method','ratio','reward']}),flush=True)


if __name__=='__main__':main()
