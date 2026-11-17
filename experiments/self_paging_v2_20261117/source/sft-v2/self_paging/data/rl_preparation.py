"""CPU RL prompt preparation. No policy training or privileged hints in prompts.

Adapted from upstream 223cc65 build_rl_pool; reference failures are NOT proofs
of infeasibility. Preserves all outcomes and disjoint world-level splits.
"""
import hashlib
import json
import random
from dataclasses import asdict

from ..agent.episode import EpisodeConfig, ScriptedPolicy, run_episode
from ..agent.pagers import BeladyPager, RecentPager, rule_task_policy
from ..core.log import TokenLog
from ..data.build_t1_sft import t1_budget
from ..envs.t1_warehouse import T1Config, WarehouseEnv, rule_read
from ..oracle.t1_oracle import T1Oracle

SPLITS = {'train': (10000, 13000), 'dev': (92000, 92200),
          'test': (110000, 110200), 'ood': (210000, 210100)}
HISTORICAL = [(0, 1000), (2000, 3000), (90000, 90100), (91000, 91100),
              (100000, 100050)]

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def config(seed, ood=False):
    r = random.Random(f'ood-{seed}' if ood else f'cfg-{seed}')
    if ood:
        return T1Config(n_decisions=r.randint(7, 9), warmup=r.randint(30, 50), gap=(20, 60),
                        events_per_page=(8, 16), p_arrival=r.uniform(.2, .4), p_dispatch=r.uniform(.08, .2))
    lo, ep_lo = r.randint(2, 10), r.randint(2, 6)
    return T1Config(n_decisions=r.randint(2, 6), warmup=r.randint(10, 30),
                    gap=(lo, lo+r.randint(5, 30)), events_per_page=(ep_lo, ep_lo+r.randint(2, 8)),
                    p_arrival=r.uniform(.2, .4), p_dispatch=r.uniform(.08, .2))

def decode_config(value):
    return T1Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in value.items()})

def split_audit():
    seen = set()
    for name, (a, b) in SPLITS.items():
        current = set(range(a, b))
        assert not seen & current
        assert not any(a < oldb and olda < b for olda, oldb in HISTORICAL)
        seen |= current
    return {'splits': SPLITS, 'excluded_historical_seed_ranges': HISTORICAL,
            'seed_overlap': 0, 'historical_scope': 'Recorded T1 SFT, heldout and pilot seed manifests'}

def clean_reference(rec):
    bad = ('invalid', 'truncated_tokens', 'forced_evictions', 'kv_errors', 'over_budget')
    return rec['reward'] == 1.0 and not any(rec['meta'].get(k, 0) for k in bad)

def summary(rec):
    return {'reward': rec['reward'], 'tokens': len(rec['tokens']), 'peak_resident': rec['peak_resident'],
            'meta': rec['meta'], 'completed_cleanly': clean_reference(rec)}

def reference(seed, cfg, tok, ratio, method, max_turns):
    env = WarehouseEnv(seed, cfg)
    budget = t1_budget(env, tok, ratio)
    pager = {'belady': BeladyPager, 'recent': RecentPager}.get(method)
    policy = rule_task_policy if method in ('full', 'belady', 'recent') else T1Oracle(method, seed=seed)
    rec = run_episode(env, TokenLog(budget), ScriptedPolicy(tok, policy), tok,
                      EpisodeConfig(budget=budget, max_turns=max_turns,
                                    kv_tools=method not in ('full', 'belady', 'recent')),
                      pager=pager() if pager else None)
    rec['meta'].update(seed=seed, ratio=ratio, reference_method=method)
    return rec

def describe(seed, split, tok):
    cfg = config(seed, split == 'ood')
    env = WarehouseEnv(seed, cfg)
    lengths = [len(tok.encode(p)) for p in env.ep.pages]
    for d in env.ep.decisions:
        assert rule_read(env.ep.pages[:d.page+1]) == d.answer, 'Full-context oracle wrong'
        assert rule_read([env.ep.pages[i] for i in sorted(env.ep.need[d.page])]) == d.answer
    cfgdict = json.loads(json.dumps(asdict(cfg)))
    max_turns = max(60, len(env.ep.pages)*2 + 4*len(env.ep.decisions) + 8)
    return dict(split=split, env='t1', seed=seed, t1_config=cfgdict,
                world_hash=digest(env.ep.pages), config_hash=digest(cfgdict),
                n_pages=len(lengths), n_decisions=len(env.ep.decisions),
                page_tokens=sum(lengths), page_lengths=lengths,
                max_turns=max_turns, max_new_tokens=512, max_log_tokens=32768,
                event_kinds=[d.kind for d in env.ep.decisions]), cfg

def select_training(rows, target=5000):
    """Exact budget quotas where possible, decision-count-balanced within each budget."""
    selected, gaps = [], {}
    for ratio, proportion in [(1., .1), (.6, .3), (.4, .3), (.25, .3)]:
        quota = round(target*proportion)
        buckets = {}
        for r in rows:
            if r['split'] == 'train' and r['ratio'] == ratio and r['train_eligible']:
                buckets.setdefault(r['n_decisions'], []).append(r)
        for values in buckets.values():
            values.sort(key=lambda r: digest(['select-v1', r['prompt_id']]))
        got = []
        while len(got) < quota and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(got) < quota: got.append(buckets[key].pop())
        selected.extend(got)
        gaps[str(ratio)] = quota-len(got)
    return selected, gaps

def select_profile(rows, count=64):
    buckets = {}
    for r in rows:
        key = (r['ratio'], r['n_decisions'], r.get('sensitive', False))
        buckets.setdefault(key, []).append(r)
    for vs in buckets.values(): vs.sort(key=lambda x: digest(['profile-v1', x['prompt_id']]))
    result = []
    while len(result) < count and any(buckets.values()):
        for k in sorted(buckets):
            if buckets[k] and len(result) < count: result.append(buckets[k].pop())
    assert len(result) == count
    return result
