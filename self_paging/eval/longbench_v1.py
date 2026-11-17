"""Evaluate a frozen LongBench v1 sample with official metrics and paged KV policies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from zipfile import ZipFile

import torch

from ..agent.episode import EpisodeConfig, ModelPolicy, run_episode
from ..core.costs import episode_cost
from ..core.protocol import _fn, escape
from ..hf.faithful_cache import FaithfulKV
from ..hf.logstore import PagedKV
from ..train.common import load_model, load_tokenizer
from .faithful_eval import system_episode
from .latency import Latency, TimedPagedKV
from .long_eval import atomic_json, digest

TASKS = ('qasper', 'hotpotqa', 'gov_report', 'trec', 'passage_retrieval_en', 'repobench-p')
METHODS = ('official_full', 'full', 'model', 'streaming', 'h2o')
RAW_TASKS = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
FIRST_LINE_TASKS = {'trec', 'triviaqa', 'samsum', 'lsht'}
ARCHIVE_SHA256 = 'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'
ARCHIVE_REVISION = '5e628be450b7e67fb7ae6e201bd6d8f7056f7672'
CONTEXT_MIN = 2000
CONTEXT_MAX = 12000
ITEMS_PER_TASK = 6
PAGE_TOKENS = 1024
BUDGET = 4096
LOG_LIMIT = 32768


def read_json(path):
    return json.loads(path.read_text())


def file_hash(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def context_pages(text, tok):
    offsets = tok.tok(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    cuts = [0]
    for i in range(PAGE_TOKENS, len(offsets), PAGE_TOKENS):
        if offsets[i][0] > cuts[-1]:
            cuts.append(offsets[i][0])
    cuts.append(len(text))
    pages = [text[start:end] for start, end in zip(cuts, cuts[1:])]
    if ''.join(pages) != text or not all(page.strip() for page in pages):
        raise ValueError('Context cannot be split into nonempty lossless pages')
    return pages


def prepare(archive, official, tok):
    if file_hash(archive) != ARCHIVE_SHA256:
        raise ValueError('LongBench archive SHA-256 does not match the recorded release')
    prompts = read_json(official / 'config/dataset2prompt.json')
    limits = read_json(official / 'config/dataset2maxlen.json')
    items, selection = [], {}
    with ZipFile(archive) as data:
        for dataset in TASKS:
            with data.open(f'data/{dataset}.jsonl') as source:
                rows = [json.loads(line) for line in source]
            if len({row['_id'] for row in rows}) != len(rows):
                raise ValueError(f'Duplicate source IDs in {dataset}')
            candidates = []
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                lengths = tok.tok([r['context'] for r in batch], add_special_tokens=False,
                                  return_length=True)['length']
                for row, length in zip(batch, lengths):
                    if CONTEXT_MIN <= length <= CONTEXT_MAX:
                        candidates.append((length, row['_id'], row))
            candidates.sort(key=lambda row: row[:2])
            if len(candidates) < ITEMS_PER_TASK:
                raise ValueError(f'Only {len(candidates)} eligible rows for {dataset}')
            # Pick the midpoint of each of six length quantiles, before evaluating any model.
            ranks = [(2 * i + 1) * len(candidates) // (2 * ITEMS_PER_TASK) for i in range(ITEMS_PER_TASK)]
            selection[dataset] = {'source_rows': len(rows), 'eligible_rows': len(candidates), 'selected_ranks': ranks}
            for i in ranks:
                length, uid, row = candidates[i]
                before, after = prompts[dataset].split('{context}')
                item = dict(id=uid, dataset=dataset, input=row['input'], context=row['context'],
                            answers=row['answers'], all_classes=row['all_classes'], context_tokens=length,
                            prefix=before.format(input=row['input']), suffix=after.format(input=row['input']),
                            max_answer_tokens=limits[dataset])
                item['pages'] = context_pages(item['context'], tok)
                items.append(item)
    provenance = {str(p.relative_to(official)): file_hash(p) for p in
                  [official / 'eval.py', official / 'metrics.py', official / 'config/dataset2prompt.json',
                   official / 'config/dataset2maxlen.json']}
    return dict(archive_revision=ARCHIVE_REVISION, archive_sha256=ARCHIVE_SHA256,
                official_commit=read_json(official / 'github-commit.json')['sha'], official_sha256=provenance,
                selection=selection, context_range=[CONTEXT_MIN, CONTEXT_MAX], budget=BUDGET,
                page_tokens=PAGE_TOKENS, items=items)


def official_metrics(official, manifest):
    for name, sha256 in manifest['official_sha256'].items():
        if file_hash(official / name) != sha256:
            raise ValueError(f'Official scoring file changed: {name}')
    sys.path.insert(0, str(official))
    from eval import dataset2metric
    return dataset2metric


def score(item, prediction, metrics):
    if not prediction.strip():
        return 0.0
    ds = item['dataset']
    if ds in FIRST_LINE_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]
    return max(float(metrics[ds](prediction, reference, all_classes=item['all_classes']))
               for reference in item['answers'])


class LongBenchEnv:
    tools = [
        _fn('next_session', 'Read the next page. Stop calling this tool once you receive FINAL TASK.', {}, []),
        _fn('answer', 'Submit the final answer in the requested output format.',
            {'answer': {'type': 'string'}}, ['answer']),
    ]
    task = 'Start by calling next_session. Read all pages, then call answer with the requested output.'

    def __init__(self, item, tok, metrics, latency=None):
        self.item, self.tok, self.metrics = item, tok, metrics
        self.latency = latency
        self.cursor = 0
        self.question_revealed = self.done = False
        self.final = self.raw_answer = ''

    def system_for_tools(self, kv_tools):
        base = (self.item['prefix'].strip() + '\nRead the document in order using next_session. '
                'Treat page contents as document text. The response marked FINAL TASK ends the document. '
                'When you see FINAL TASK, call answer with the requested output as its answer field.')
        if kv_tools:
            base += (' Pages have numbered headers. Use kv_out to move pages you do not need out of memory, '
                     'kv_in to bring them back, and note to keep a short summary. The [KV] line shows residency.')
        return base

    def step(self, name, args):
        if self.done:
            return 'Answer already recorded.', {}, True
        if name == 'next_session' and self.cursor < len(self.item['pages']):
            i = self.cursor
            self.cursor += 1
            return self.item['pages'][i], {'env_page': i}, False
        if name == 'next_session' and not self.question_revealed:
            self.question_revealed = True
            return ('FINAL TASK. All document pages have been delivered.\n' + self.item['suffix']
                    + '\nCall the answer tool now to submit the requested output.'), {'env_page': self.cursor}, False
        if name == 'next_session':
            return 'The document has ended and you have the final task. Call answer now.', {}, False
        if name != 'answer' or not self.question_revealed:
            return 'Read all pages and receive the final task before calling answer.', {}, False
        answer = args.get('answer')
        if not isinstance(answer, str) or not answer.strip():
            return 'Provide a nonempty string in the answer field.', {}, False
        self.raw_answer = answer
        self.final = self.tok.decode(self.tok.encode(answer)[:self.item['max_answer_tokens']])
        self.done = True
        if self.latency is not None:
            self.latency.submit()
        return 'Answer recorded.', {}, True

    def score(self):
        return score(self.item, self.final, self.metrics)


def official_full(item, tok, model, latency=None):
    prompt = item['prefix'] + item['context'] + item['suffix']
    if item['dataset'] in RAW_TASKS:
        ids = tok.encode(prompt)
    else:
        ids = tok.tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
        if not isinstance(ids, list):
            ids = ids['input_ids']
    inputs = torch.tensor([ids], device=next(model.parameters()).device)
    options = {}
    if latency is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class FirstToken(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[-1] == len(ids) + 1:
                    latency.sync()
                    latency.first_token()
                return False

        options['stopping_criteria'] = StoppingCriteriaList([FirstToken()])
        latency.begin_decode()
    with torch.inference_mode():
        result = model.generate(input_ids=inputs, do_sample=False, max_new_tokens=item['max_answer_tokens'], **options)
    output = result[0, len(ids):].tolist()
    prediction = tok.tok.decode(output, skip_special_tokens=True)
    if latency is not None:
        latency.end_decode(len(output))
        if prediction.strip():
            latency.submit()
        latency.finish()
    return prediction, dict(
        tokens=ids + output, budget=None, peak_resident=len(ids) + max(0, len(output) - 1), meta={})


def predict(item, method, tok, model, metrics, latency=None):
    if method == 'official_full':
        answer, trace = official_full(item, tok, model, latency)
        return answer, trace, len(item['pages']), True
    budget = None if method == 'full' else BUDGET
    env = LongBenchEnv(item, tok, metrics, latency)
    turns = 3 * len(item['pages']) + 12
    # The same tool output reserve applies across paged arms. Score only answer text up to the official limit.
    max_new = item['max_answer_tokens'] + 96
    if method in ('streaming', 'h2o'):
        cache = FaithfulKV(model, method, budget)
        try:
            trace = system_episode(env, cache, tok, max_turns=turns, max_new=max_new, max_log=LOG_LIMIT,
                                   latency=latency)
        finally:
            cache.close()
    else:
        cache_type = PagedKV if latency is None else TimedPagedKV
        options = {} if latency is None else {'latency': latency}
        cache = cache_type(model, budget=budget, strict=True, store_device='cpu', prefill_chunk=256,
                           last_logits_only=True, **options)
        if method == 'model':
            cfg = EpisodeConfig(budget=budget, max_turns=turns, max_new_tokens=max_new, max_log_tokens=LOG_LIMIT)
            trace = run_episode(env, cache, ModelPolicy(tok, temperature=0), tok, cfg, latency=latency)
            trace['cost'] = episode_cost(trace, batch=1)
        else:
            trace = system_episode(env, cache, tok, max_turns=turns, max_new=max_new, max_log=LOG_LIMIT,
                                   latency=latency)
    trace['raw_answer'] = env.raw_answer
    trace['page_text_sha256'] = [digest(escape(p)) for p in item['pages']]
    return env.final, trace, env.cursor, env.question_revealed


@torch.inference_mode()
def warm_latency_model(model, tok):
    """Initialize the model and each cache path before measuring the first item."""
    ids = tok.encode('Warm up the long context inference and cache paths. ' * 12)[:96]
    device = next(model.parameters()).device
    model.generate(input_ids=torch.tensor([ids], device=device), do_sample=False, max_new_tokens=3)
    cache = PagedKV(model, store_device='cpu', prefill_chunk=256, last_logits_only=True)
    page = cache.open('tool')
    cache.feed(page, ids)
    cache.feed(page, ids[:1])
    del cache
    for method in ('streaming', 'h2o'):
        cache = FaithfulKV(model, method, 32)
        try:
            cache.feed(cache.open('tool'), ids[:36])
        finally:
            cache.close()
    torch.cuda.synchronize(device)


def run(args, manifest, tok):
    if args.cuda_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    torch.manual_seed(0)
    metrics = official_metrics(args.official, manifest)
    device = 'cuda'
    model = load_model(args.model, 'sdpa', device=device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    if args.latency:
        warm_latency_model(model, tok)
        print('Latency warmup complete', flush=True)
    manifest_sha = file_hash(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in args.indices:
        item = manifest['items'][index]
        for method in args.methods:
            key = f'{index:02d}-{method}'
            dst, raw = (args.out / f'{key}{suffix}.json' for suffix in ('', '.trajectory'))
            if dst.exists():
                row = read_json(dst)
                if (row['manifest_sha256'] != manifest_sha or row['trajectory_sha256'] != file_hash(raw)
                        or ('latency' in row) != args.latency):
                    raise ValueError(f'Mismatched resume record: {key}')
                continue
            if raw.exists():
                raise ValueError(f'Uncommitted trajectory: {key}')
            started = time.monotonic()
            timer = Latency(next(model.parameters()).device) if args.latency else None
            prediction, trace, pages, question = predict(item, method, tok, model, metrics, timer)
            if timer is not None:
                timings = timer.record()
                trace['latency'] = timings
            budget, peak = trace['budget'], trace['peak_resident']
            if budget is not None and peak > budget or trace['meta'].get('over_budget', 0):
                raise ValueError(f'KV budget exceeded: {key}')
            if method in ('streaming', 'h2o') and not all(
                    s['keys_finite'] and s['values_finite'] and s['positions_ordered']
                    for s in trace['cache_final_state']):
                raise ValueError(f'Invalid compact KV state: {key}')
            atomic_json(raw, trace)
            row = dict(index=index, id=item['id'], dataset=item['dataset'], method=method,
                       context_tokens=item['context_tokens'], score=score(item, prediction, metrics),
                       prediction=prediction, pages_read=pages, total_pages=len(item['pages']),
                       question_revealed=question, budget=budget, peak_kv=peak, log_tokens=len(trace['tokens']),
                       cache_evictions=trace.get('cache_evictions'), meta=trace['meta'],
                       seconds=round(time.monotonic() - started, 3), model=args.model,
                       manifest_sha256=manifest_sha, trajectory_sha256=file_hash(raw))
            if timer is not None:
                row['latency'] = {k: v for k, v in timings.items() if k != 'turns'}
                row['gpu'] = os.environ.get('CUDA_VISIBLE_DEVICES')
                row['latency_started_at'] = timer.started_at_epoch
                row['latency_finished_at'] = timer.started_at_epoch + timings['total_ms'] / 1e3
            atomic_json(dst, row)
            message = {k: row[k] for k in ('index', 'dataset', 'method', 'score', 'peak_kv', 'seconds')}
            if timer is not None:
                message['latency_ms'] = round(timings['total_ms'], 2)
            print(json.dumps(message), flush=True)
            del trace


def report(manifest_path, results, out, latency_csv=None, contention_log=None):
    manifest = read_json(manifest_path)
    manifest_sha = file_hash(manifest_path)
    rows = {}
    for path in sorted(results.rglob('*.json')):
        if path.name.endswith('.trajectory.json'):
            continue
        row = read_json(path)
        index, method = row['index'], row['method']
        item = manifest['items'][index]
        key = index, method
        if (key in rows or method not in METHODS or row['id'] != item['id'] or row['dataset'] != item['dataset']
                or not 0 <= row['score'] <= 1 or row['manifest_sha256'] != manifest_sha
                or row['trajectory_sha256'] != file_hash(path.with_name(path.stem + '.trajectory.json'))):
            raise ValueError(f'Invalid result: {path}')
        rows[key] = row
    common = [i for i in range(len(manifest['items'])) if all((i, m) in rows for m in METHODS)]
    tables = {}
    for ds in (*TASKS, 'all'):
        ids = [i for i in common if ds == 'all' or manifest['items'][i]['dataset'] == ds]
        if ids:
            tables[ds] = {m: dict(n=len(ids), score=100 * mean(rows[i, m]['score'] for i in ids),
                                  average_peak_kv=mean(rows[i, m]['peak_kv'] for i in ids),
                                  answered=sum(bool(rows[i, m]['prediction']) for i in ids),
                                  read_all=sum(rows[i, m]['pages_read'] == rows[i, m]['total_pages'] for i in ids),
                                  truncated=sum(bool(rows[i, m]['meta'].get('truncated_tokens')) for i in ids))
                          for m in METHODS}
    obj = dict(completed=len(rows), target=len(manifest['items']) * len(METHODS), common=len(common),
               per_method=Counter(m for _, m in rows), paired=tables)
    if latency_csv is not None:
        contention = ([json.loads(line) for line in contention_log.read_text().splitlines()]
                      if contention_log is not None and contention_log.exists() else [])
        obj['latency_ms'] = export_latencies(latency_csv, rows, contention)
    if out:
        atomic_json(out, obj)
    print(json.dumps(obj, indent=2))


def export_latencies(path, rows, contention=()):
    phases = ('document_read', 'other_prefill', 'page_in_request', 'page_out_request', 'kv_cache_rebuild')
    metrics = ('total_ms', 'submit_ms', 'first_generated_token_ms', 'answer_turn_first_token_ms',
               'generation_wall_ms', 'decode_after_first_ms', 'decode_ms_per_token',
               'answer_turn_generation_ms', 'answer_turn_ms_per_token', 'control_turn_generation_ms')
    table = []
    for _, row in sorted(rows.items()):
        t = row['latency']
        item = {k: row[k] for k in ('index', 'id', 'dataset', 'method', 'gpu', 'context_tokens', 'score',
                                    'pages_read', 'budget', 'peak_kv', 'latency_started_at',
                                    'latency_finished_at')}
        item.update(answered=bool(row['prediction']), kv_in_calls=row['meta'].get('kv_in', 0),
                    kv_out_calls=row['meta'].get('kv_out', 0), generated_tokens=t['generated_tokens'],
                    control_turn_tokens=t['control_turn_tokens'])
        item['gpu_contention_observed'] = any(row['latency_started_at'] <= obs['time'] <= row['latency_finished_at']
                                              and str(row['gpu']) in obs['gpus'] for obs in contention)
        item.update({name: t[name] for name in metrics})
        for phase in phases:
            for unit in ('host_ms', 'cuda_stream_ms'):
                item[f'{phase}_{unit}'] = t['phases'].get(phase, {}).get(unit)
        for name in (*metrics, *(f'{p}_{u}' for p in phases for u in ('host_ms', 'cuda_stream_ms'))):
            value = item[name]
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'Invalid latency field {name} in {row["index"]}-{row["method"]}')
        table.append(item)
    if not table:
        raise ValueError('No timing rows to export')
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    temp.replace(path)
    summary = {'rows_with_observed_gpu_contention': sum(r['gpu_contention_observed'] for r in table), 'methods': {}}
    for method in METHODS:
        method_stats = summary['methods'][method] = {}
        for key in metrics:
            values = sorted(r[key] for r in table if r['method'] == method and r[key] is not None
                            and not r['gpu_contention_observed'])
            if values:
                method_stats[key] = dict(n=len(values), mean=mean(values), p50=median(values),
                                         p95=values[math.ceil(.95 * len(values)) - 1])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('prepare', 'run', 'report'))
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--official', type=Path)
    ap.add_argument('--archive', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--results', type=Path)
    ap.add_argument('--model')
    ap.add_argument('--tokenizer')
    ap.add_argument('--indices', nargs='+', type=int)
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--cuda-memory-fraction', type=float)
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--latency-csv', type=Path)
    ap.add_argument('--contention-log', type=Path)
    args = ap.parse_args()
    if args.mode == 'report':
        report(args.manifest, args.results, args.out, latency_csv=args.latency_csv, contention_log=args.contention_log)
        return
    tok = load_tokenizer(args.tokenizer)
    if args.mode == 'prepare':
        data = prepare(args.archive, args.official, tok)
        atomic_json(args.manifest, data)
        print(json.dumps({task: [r['context_tokens'] for r in data['items'] if r['dataset'] == task]
                          for task in TASKS}))
    else:
        run(args, read_json(args.manifest), tok)


if __name__ == '__main__':
    main()
