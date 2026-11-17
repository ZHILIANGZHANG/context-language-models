"""Rebuild snapshot tables and verify captured bytes. Python standard library only."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'tables'
TABLES.mkdir(exist_ok=True)


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def table(name, rows):
    if not rows:
        return
    with (TABLES / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


manifest = json.loads((ROOT / 'manifest.json').read_text())
historical = json.loads((ROOT / 'historical-manifest.json').read_text())
for entry in manifest['files'] + historical['files']:
    path = ROOT / entry['path']
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], path
    if path.suffix == '.gz' and not entry['source'].endswith('.gz'):
        data = gzip.decompress(data)
    assert hashlib.sha256(data).hexdigest() == entry['source_captured_sha256'], path

methods = ['official_full', 'full', 'model', 'streaming', 'h2o']
rows = []
audit = []
verified = 0
for folder in ['longbench-v1-latency', 'longbench-v1', 'longbench-v1-before-final-marker', 'faithful-baselines']:
    seen = set()
    for path in sorted((ROOT / 'raw' / folder).rglob('*.json')):
        if '.trajectory.' in path.name:
            continue
        row = json.loads(path.read_text())
        if 'trajectory_sha256' not in row:
            continue
        raw = path.with_name(path.stem + '.trajectory.json.gz')
        if not raw.exists():
            raw = path.with_name(path.stem + '.trajectory.json')
        data = read(raw)
        assert hashlib.sha256(data).hexdigest() == row['trajectory_sha256'], raw
        verified += 1
        key = (row.get('index', row.get('seed')), row['method'], row.get('ratio'))
        assert key not in seen, (folder, key)
        seen.add(key)
        if folder != 'longbench-v1-latency':
            continue
        rows.append(row)
        if row['method'] == 'model':
            trace = json.loads(data)
            logs = trace['meta'].get('kv_log', [])
            good = [x for x in logs if x['op'] == 'kv_out' and x['ok']]
            bad = [x for x in logs if x['op'] == 'kv_out' and not x['ok']]
            pages = {p['pid']: p for p in trace['pages']}
            kinds = collections.Counter(pages[p]['kind'] for x in good for p in x['pages'])
            assert len(good) == row['meta']['kv_out']
            audit.append(dict(index=row['index'], dataset=row['dataset'], attempts=len(good)+len(bad),
                successful_calls=len(good), failed_calls=len(bad), out_pages=sum(kinds.values()),
                assistant_pages=kinds['assistant'], tool_pages=kinds['tool'], note_pages=kinds['note'],
                kv_in=row['meta']['kv_in'], forced_out=row['meta']['forced_evictions'],
                truncated_tokens=row['meta']['truncated_tokens'], peak_kv=row['peak_kv']))

common = set.intersection(*[{r['index'] for r in rows if r['method'] == m} for m in methods])
scores = []
for dataset in ['ALL'] + sorted({r['dataset'] for r in rows}):
    for method in methods:
        selected = [r for r in rows if r['index'] in common and r['method'] == method
                    and (dataset == 'ALL' or r['dataset'] == dataset)]
        if selected:
            scores.append(dict(dataset=dataset, method=method, paired_items=len(selected),
                               mean_official_score_100=100*sum(r['score'] for r in selected)/len(selected)))
table('longbench_paired_scores.csv', scores)
table('sft_kv_out_audit.csv', audit)
table('longbench_per_item.csv', [dict(index=r['index'], dataset=r['dataset'], method=r['method'],
    context_tokens=r['context_tokens'], score=r['score'], budget=r['budget'], peak_kv=r['peak_kv'],
    total_ms=r.get('latency', {}).get('total_ms'),
    successful_out_calls=r['meta'].get('kv_out', 0), successful_in_calls=r['meta'].get('kv_in', 0),
    forced_out=r['meta'].get('forced_evictions', 0), truncated_tokens=r['meta'].get('truncated_tokens', 0)) for r in rows])
training = [json.loads(line) for line in (ROOT/'training/sft.jsonl').read_text().splitlines() if line.strip()]
validation = [r for r in training if 'eval_loss' in r]
table('sft_validation.csv', [dict(completed_steps=r.get('completed_steps'), eval_loss=r['eval_loss'],
                                records=r.get('eval_records'), seconds=r.get('eval_sec')) for r in validation])
legacy = []
for path in (ROOT/'control/post-sft-eval').rglob('summary.jsonl'):
    legacy.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
report = dict(captured_at=manifest['captured_at'], verified_files=len(manifest['files']),
    verified_historical_files=len(historical['files']), historical_captured_at=historical['captured_at'],
    verified_trajectory_pairs=verified, longbench_completed=len(rows), longbench_target=180,
    method_counts=dict(collections.Counter(r['method'] for r in rows)), paired_items=len(common),
    paired_indices=sorted(common), sft_completed_steps=max(r.get('completed_steps', 0) for r in training),
    validation=validation, legacy_superseded_post_sft_rows=len(legacy),
    kv_totals={k:sum(a[k] for a in audit) for k in ['attempts','successful_calls','failed_calls','out_pages',
        'assistant_pages','tool_pages','note_pages','kv_in','forced_out','truncated_tokens']})
(TABLES/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
