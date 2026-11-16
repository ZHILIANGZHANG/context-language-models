"""Base Qwen3-8B controls paired with the existing sampled SFT evaluation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from self_paging.eval.t1_eval import run_one
from self_paging.agent.episode import EpisodeConfig, ModelPolicy, run_episode
from self_paging.agent.pagers import QuestPagePager, AttnEvictPager
from self_paging.hf.observed import ObservedPagedKV
from self_paging.data.build_t1_sft import t1_budget
from self_paging.envs.t1_warehouse import WarehouseEnv, T1Config
from self_paging.core.costs import episode_cost
from self_paging.train.common import load_model, load_tokenizer, append_jsonl
from audit_protocol import audit

H = Path('/home/zhangcheng/self-paging-20261116')
N = Path('/nas/Public/zhangcheng/self-paging-20261116')
M = '/nas/Public/zhangcheng/models/Qwen3-8B'
R = H / 'base-controls-two'
CONDITIONS = [(m, r) for m in ['recent', 'quest'] for r in [.25, .4, .6]]

def observed_one(method, model, tok, seed, ratio, gen):
    env = WarehouseEnv(seed, T1Config(n_decisions=4))
    budget = t1_budget(env, tok, ratio)
    cfg = EpisodeConfig(budget=budget, max_turns=60, max_new_tokens=1024, kv_tools=False)
    log = ObservedPagedKV(model, budget=budget, strict=False,
                          attention_mass=method == 'attn_evict', window=16)
    pager = QuestPagePager() if method == 'quest' else AttnEvictPager()
    policy = ModelPolicy(tok, temperature=.6, top_p=.95, top_k=20, generator=gen)
    rec = run_episode(env, log, policy, tok, cfg, pager=pager)
    rec.update(method=method, seed=seed, ratio=ratio,
        decisions=[dict(page=d.page, kind=d.kind, gold=d.answer,
                        predicted=env.answers[d.page]) for d in env.ep.decisions],
        environment_pages=env.ep.pages)
    row = dict(method=method, seed=seed, ratio=ratio, budget=budget,
               reward=rec['reward'], peak_resident=rec['peak_resident'],
               tokens=len(rec['tokens']), **episode_cost(rec), **rec['meta'])
    return row, rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    a = ap.parse_args()
    job = R / f'job-{a.start}'
    rawdir = N / 'base-controls-two' / f'job-{a.start}'
    rawdir.mkdir(parents=True, exist_ok=True)
    reused = {(r['seed'],r['method'],r['ratio']):r
              for r in json.loads((R/'reused.json').read_text())}
    tok = load_tokenizer(M)
    model = load_model(M, 'sdpa', device='cuda').eval()
    assert not hasattr(model, 'peft_config'), 'Expected original model without adapters'
    print(json.dumps(dict(event='base_model_loaded', path=M, adapter=False)), flush=True)
    for seed in range(a.start, a.end):
        for method, ratio in CONDITIONS:
            if any(p.exists() for p in [H/'STOP', R/'STOP', H/'quality-review/STOP']):
                raise RuntimeError('User STOP')
            previous = reused.get((seed,method,ratio))
            if previous is not None:
                path = Path(previous['raw_path'])
                assert hashlib.sha256(path.read_bytes()).hexdigest() == previous['raw_sha256']
                assert previous['model_path'] == M and previous['sample_seed'] == 0
                append_jsonl(job/'summary.jsonl', previous)
                print(json.dumps(dict(event='reused',seed=seed,method=method,ratio=ratio)),flush=True)
                continue
            start = time.time()
            gen = torch.Generator(device='cuda').manual_seed(seed)
            with torch.inference_mode():
                if method in ['quest', 'attn_evict']:
                    row, rec = observed_one(method, model, tok, seed, ratio, gen)
                else:
                    captured = []
                    row = run_one(method, model, tok, seed, ratio,
                        T1Config(n_decisions=4), temperature=.6, top_p=.95, top_k=20,
                        max_turns=60, max_new_tokens=1024, gen=gen, record_sink=captured.append)
                    assert len(captured) == 1
                    rec = captured[0]
            name = f'{seed}-{method}-{ratio}.json.gz'
            path = rawdir / name
            assert not path.exists()
            payload = json.dumps(rec).encode()
            tmp = path.with_suffix('.tmp')
            tmp.write_bytes(gzip.compress(payload))
            tmp.replace(path)
            assert len(rec['decisions']) == 4
            assert row['reward'] == sum(d['gold'] == d['predicted'] for d in rec['decisions']) / 4
            assert row['over_budget'] == 0 and len(rec['tokens']) <= 32768
            evidence = audit(rec, tok)
            row.update(sample_seed=0, seconds=time.time()-start,
                       raw_path=str(path), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       protocol_counts=dict(evidence['counts']), model_path=M)
            append_jsonl(job/'summary.jsonl', row)
            print(json.dumps({k:row[k] for k in ['seed','method','ratio','reward','seconds']}), flush=True)
    (job/'accepted.json').write_text(json.dumps(dict(episodes=(a.end-a.start)*6, time=time.time())))

if __name__ == '__main__':
    main()
