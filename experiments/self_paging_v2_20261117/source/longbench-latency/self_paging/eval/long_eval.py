"""BCP / LongMemEval inference with auditable, deferred local judging.

Pending judge results are null, never interpreted as incorrect predictions.
Every episode is independently seeded and saved atomically. Resume requires
the exact source, model identifier, input digest and scientific settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from ..agent.episode import EpisodeConfig, ModelPolicy, run_episode
from ..agent.pagers import RecentPager
from ..core.costs import episode_cost
from ..envs.longmemeval import LongMemEvalEnv, load_items
from ..envs.t2_bcp import BCPEnv, PyseriniSearcher, load_split
from ..hf.logstore import PagedKV
from ..train.common import load_model, load_tokenizer


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(root.rglob('*.py'))})


class DeferredJudge:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return 0.0  # internal placeholder; replaced by null before any result is saved


def validate_items(items, dataset):
    key = 'id' if dataset == 'bcp' else 'question_id'
    ids = [str(x[key]) for x in items]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs')
    return ids


def make_result(dataset, item, env, rec, judge, cfg):
    pending = bool(judge.calls)
    if len(judge.calls) > 1:
        raise ValueError('An episode requested more than one judgement')
    # A missing or policy-rejected final answer scores zero without an LLM.
    # An answer awaiting semantic judgement is never scored as zero.
    rec['reward'] = None if pending else 0.0
    return {'dataset': dataset, 'question_id': str(item['id'] if dataset == 'bcp' else item['question_id']),
            'answer': env.final, 'reference': item['answer'], 'question': item['question'],
            'question_type': item.get('question_type'), 'question_date': item.get('question_date'),
            'judge_status': 'pending' if pending else 'no_valid_final_answer',
            'reward': rec['reward'], 'budget': cfg.budget, 'log_tokens': len(rec['tokens']),
            'peak_resident': rec['peak_resident'], 'meta': rec['meta'], **episode_cost(rec)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', choices=['bcp', 'longmemeval'], required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--index')
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='+', choices=['full', 'recent', 'model'], default=['full', 'recent', 'model'])
    ap.add_argument('--budgets', nargs='+', type=int, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int)
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--sample-start', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--temperature', type=float, default=.6)
    ap.add_argument('--max-turns', type=int)
    ap.add_argument('--max-new-tokens', type=int)
    ap.add_argument('--question-first', action='store_true')
    ap.add_argument('--no-notes', action='store_true')
    ap.add_argument('--irreversible', action='store_true')
    a = ap.parse_args()
    if a.max_new_tokens is None:
        a.max_new_tokens = 4096 if a.dataset == 'bcp' else 512
    if a.dataset == 'bcp' and (not a.index or a.question_first):
        ap.error('BCP needs --index and does not support --question-first')
    if min(a.budgets) <= 0 or a.samples < 1 or a.sample_start < 0:
        ap.error('Invalid budget or sample count')
    items = load_split(a.data) if a.dataset == 'bcp' else load_items(a.data)
    ids = validate_items(items, a.dataset)
    end = len(items) if a.end is None else a.end
    if not 0 <= a.start < end <= len(items):
        ap.error('Invalid item interval')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = vars(a).copy()
    # Shard boundaries do not change individual episode semantics.
    for field in ['start', 'end', 'out', 'sample_start', 'samples']:
        settings.pop(field)
    settings.update(source_sha256=source_digest(), data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                    yarn_factor=4, absolute_log_limit=120000, thinking=a.dataset == 'bcp', kv_backing='cpu',
                    prefill_chunk=512, last_logits_only=True,
                    top_p=.95, top_k=20, max_turns=a.max_turns or (60 if a.dataset == 'bcp' else 400))
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Refusing to mix incompatible runs')
    atomic_json(manifest, settings)
    tok = load_tokenizer(a.tokenizer)
    searcher = PyseriniSearcher(a.index) if a.dataset == 'bcp' else None
    model = load_model(a.model, 'sdpa', device='cuda', yarn_factor=4).eval()
    for i in range(a.start, end):
        for method in a.methods:
            for budget in ([None] if method == 'full' else a.budgets):
                for sample in range(a.sample_start, a.sample_start + a.samples):
                    key = digest([ids[i], method, budget, sample])[:24]
                    result_path, trace_path = out / f'{key}.json', out / f'{key}.trajectory.json'
                    if result_path.exists():
                        row = json.loads(result_path.read_text())
                        if not trace_path.exists() or row['trajectory_sha256'] != hashlib.sha256(trace_path.read_bytes()).hexdigest():
                            raise ValueError('Existing episode has missing or damaged trajectory')
                        continue
                    if trace_path.exists():
                        raise ValueError('Uncommitted trajectory present; inspect before recovery')
                    judge = DeferredJudge()
                    env = (BCPEnv(items[i], searcher, judge, tok) if a.dataset == 'bcp' else
                           LongMemEvalEnv(items[i], judge, question_first=a.question_first))
                    cfg = EpisodeConfig(budget=budget, max_turns=settings['max_turns'],
                                        max_new_tokens=a.max_new_tokens, thinking=settings['thinking'],
                                        kv_tools=method == 'model', notes=not a.no_notes,
                                        reversible=not a.irreversible, max_log_tokens=120000,
                                        answer_reserve=a.max_new_tokens + 512)
                    sample_seed = int(digest([a.seed, ids[i], sample])[:15], 16)
                    policy = ModelPolicy(tok, temperature=a.temperature, top_p=.95, top_k=20,
                                         generator=torch.Generator(device='cuda').manual_seed(sample_seed))
                    rec = run_episode(env, PagedKV(model, budget=budget, strict=False, store_device='cpu',
                                                  prefill_chunk=512, last_logits_only=True), policy, tok, cfg,
                                      pager=RecentPager() if method == 'recent' else None)
                    if len(rec['tokens']) > 120000 or rec['meta']['over_budget']:
                        raise ValueError('Log or resident budget violated')
                    row = make_result(a.dataset, items[i], env, rec, judge, cfg)
                    row.update(method=method, sample=sample, sample_seed=sample_seed, item_index=i,
                               settings_sha256=digest(settings), source_sha256=settings['source_sha256'])
                    atomic_json(trace_path, rec)
                    row['trajectory_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                    atomic_json(result_path, row)
                    print(json.dumps({k: row[k] for k in ['question_id', 'method', 'sample', 'judge_status', 'log_tokens']}), flush=True)


if __name__ == '__main__':
    main()
