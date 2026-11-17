"""Run the fixed T1 evaluation after the formal SFT checkpoint is saved."""
import collections
import contextlib
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

H = Path('/home/zhangcheng/self-paging-v2-20261117')
OLD = H.with_name('self-paging-20261116')
C = Path('/data/zhangcheng/context-language-models-sft-v2-8fc082e')
CHECKPOINT = Path('/nas/Public/zhangcheng/self-paging-v2-20261117/checkpoints/sft-v2-r64')
BASE = '/nas/Public/zhangcheng/models/Qwen3-8B'
R = H / 'post-sft-eval'
SEEDS = range(100000, 100050)
RATIOS = (0.25, 0.4, 0.6)
PHASES = {'main': ('full', 'model'), 'controls': ('recent', 'belady')}
ALLOWED_GPUS = {0, 1, 2}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pid):
    proc = Path('/proc') / str(pid)
    return dict(pid=pid, uid=proc.stat().st_uid, cwd=os.readlink(proc/'cwd'),
                start_ticks=int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19]))


def alive(expected):
    try:
        return identity(expected['pid']) == expected
    except (FileNotFoundError, ProcessLookupError):
        return False


def check_stop():
    if (H/'STOP').exists() or (R/'STOP').exists():
        raise RuntimeError('User STOP')


def health(phase, **fields):
    write(R/'health.json', dict(phase=phase, time=time.time(), **fields))


def preflight():
    for name, digest in read(H/'source-manifest.json').items():
        assert sha256(C/name) == digest, name
    for name, version in read(OLD/'env-spec.json')['pins'].items():
        assert importlib.metadata.version(name) == version, name
    plan = read(H/'training-plan.json')
    launch = read(H/'jobs/sft-quality/launch.json')
    assert Path(launch['cmd'][launch['cmd'].index('--out') + 1]) == CHECKPOINT
    assert plan['model'] == BASE and launch['identity']['cwd'] == str(C)
    assert set(launch['gpus']) == ALLOWED_GPUS
    return plan, launch


def wait_for_sft(plan, launch):
    exit_path = H/'jobs/sft-quality/exit.json'
    controller = read(H/'identity.json')
    while not exit_path.exists() or alive(launch['identity']):
        check_stop()
        assert alive(controller), 'SFT controller exited before training exit was recorded'
        health('WAIT_SFT', training_pid=launch['identity']['pid'],
               expected_steps=plan['optimizer_steps'])
        time.sleep(15)
    code = read(exit_path)
    assert code['returncode'] == 0 and code['identity'] == launch['identity'], code
    history = list(rows(H/'sft.jsonl'))
    steps = plan['optimizer_steps']
    assert [r['completed_steps'] for r in history if 'step' in r] == list(range(1, steps+1))
    validation = [r for r in history if 'eval_loss' in r][-1]
    assert validation['completed_steps'] == steps
    assert validation['eval_records'] == plan['validation_records']
    assert math.isfinite(validation['eval_loss'])
    final = CHECKPOINT/f'checkpoint-{steps:06d}'
    meta = read(final/'checkpoint.json')
    assert meta['completed_steps'] == steps and meta['eval_loss'] == validation['eval_loss']
    assert read(CHECKPOINT/'adapter_config.json')['base_model_name_or_path'] == BASE
    digests = {}
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        digests[name] = sha256(CHECKPOINT/name)
        assert sha256(final/name) == digests[name], name
    write(R/'checkpoint-accepted.json', dict(path=str(CHECKPOINT), sha256=digests,
          completed_steps=steps, final_validation=validation, training_exit=code))


def acquire_lock(stack, path):
    lock = stack.enter_context(path.open('a'))
    while True:
        check_stop()
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            health('WAIT_CAMPAIGN_LOCK', lock=str(path))
            time.sleep(10)


def free_gpus():
    gpu_args = ['nvidia-smi', '--format=csv,noheader,nounits']
    devices = subprocess.check_output(gpu_args+['--query-gpu=index,uuid,memory.used,utilization.gpu'],
                                      text=True).splitlines()
    apps = subprocess.check_output(gpu_args+['--query-compute-apps=gpu_uuid,pid'], text=True)
    busy = {line.split(',')[0].strip() for line in apps.splitlines()}
    return {int(i) for i, uuid, used, util in (line.split(',') for line in devices)
            if int(i) in ALLOWED_GPUS and int(used) < 500 and int(util) == 0 and uuid.strip() not in busy}


def key(row):
    return row['seed'], row['method'], row['ratio']


def expected_keys(methods, lo, hi):
    return {(s, method, ratio) for s in range(lo, hi) for method in methods
            for ratio in ((1.0,) if method == 'full' else RATIOS)}


def validate_job(directory, expected):
    summaries = list(rows(directory/'summary.jsonl'))
    assert len(summaries) == len(expected) and {key(r) for r in summaries} == expected
    for summary, rec in zip(summaries, rows(directory/'records.jsonl'), strict=True):
        assert key(summary) == key(rec)
        assert len(rec['decisions']) == 4
        assert summary['reward'] == sum(d['gold'] == d['predicted'] for d in rec['decisions']) / 4
        assert summary['reward'] == rec['reward'] and math.isfinite(summary['cost'])
        assert summary['over_budget'] == 0
        if summary['method'] != 'model':
            assert all(summary[k] == 0 for k in ('kv_in', 'kv_out', 'notes'))
    write(directory/'accepted.json', dict(episodes=len(expected), time=time.time()))


def dispatch(phase, ranges, active):
    methods = PHASES[phase]
    pending = list(ranges)
    while pending or active:
        check_stop()
        for gpu, (process, ident, directory, expected, log) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            del active[gpu]
            write(directory/'exit.json', dict(returncode=process.returncode, identity=ident, time=time.time()))
            assert process.returncode == 0, directory
            validate_job(directory, expected)
        if pending:
            available = free_gpus()
            time.sleep(3)
            for gpu in sorted(available & free_gpus() - active.keys()):
                if not pending:
                    break
                check_stop()
                lo, hi = pending.pop(0)
                directory = R/f'{phase}-{lo}'
                directory.mkdir()
                expected = expected_keys(methods, lo, hi)
                args = [sys.executable, '-m', 'self_paging.eval.t1_eval', '--model', str(CHECKPOINT),
                        '--tokenizer', BASE, '--methods', *methods, '--ratios', *map(str, RATIOS),
                        '--seeds', str(lo), str(hi), '--n-decisions', '4', '--attn', 'sdpa',
                        '--temperature', '0.6', '--sample-seed', '0',
                        '--out', str(directory/'summary.jsonl'), '--records-out', str(directory/'records.jsonl')]
                log = (directory/'stdout.log').open('w')
                process = subprocess.Popen(args, cwd=C, stdout=log, stderr=subprocess.STDOUT,
                                           env=os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu)},
                                           start_new_session=True)
                ident = identity(process.pid)
                active[gpu] = process, ident, directory, expected, log
                write(directory/'launch.json', dict(identity=ident, gpu=gpu, args=args,
                      expected=len(expected), time=time.time()))
        health(phase, pending=pending, active={gpu: item[1] for gpu, item in active.items()})
        if pending or active:
            time.sleep(10)


def finalize():
    from self_paging.envs.t1_warehouse import T1Config
    from self_paging.eval.error_analysis import analyse

    groups = collections.defaultdict(collections.Counter)
    budgets, worlds, seen = {}, {}, set()
    report_raw = R/'records.jsonl.gz'
    temporary = report_raw.with_suffix('.tmp')

    def records():
        with gzip.open(temporary, 'wt') as stream:
            for done in sorted(R.glob('*/accepted.json')):
                directory = done.parent
                for row, rec in zip(rows(directory/'summary.jsonl'), rows(directory/'records.jsonl'), strict=True):
                    check_stop()
                    seed, method, ratio = k = key(row)
                    assert k == key(rec) and k not in seen
                    seen.add(k)
                    fingerprint = hashlib.sha256(json.dumps(rec['environment_pages']).encode()).hexdigest()
                    assert worlds.setdefault(seed, fingerprint) == fingerprint
                    if method != 'full':
                        assert budgets.setdefault((seed, ratio), row['budget']) == row['budget']
                    else:
                        assert row['budget'] is None
                    counts = groups[f'{method}@{ratio}']
                    counts.update(episodes=1, perfect_episodes=int(row['reward'] == 1),
                                  cost_sum=row['cost'], peak_tokens_sum=row['peak_resident'])
                    for name in ('kv_in', 'kv_out', 'notes', 'invalid', 'unoffered_kv_calls',
                                 'forced_evictions', 'truncated_tokens', 'kv_errors', 'turns'):
                        counts[name] += row.get(name, 0)
                    for decision in rec['decisions']:
                        kind = decision['kind']
                        counts[f'{kind}_total'] += 1
                        counts[f'{kind}_correct'] += int(decision['gold'] == decision['predicted'])
                    stream.write(json.dumps(rec, separators=(',', ':'))+'\n')
                    yield rec

    errors = analyse(records(), T1Config(n_decisions=4))
    assert seen == expected_keys(PHASES['main'] + PHASES['controls'], SEEDS.start, SEEDS.stop)
    assert all(not counts.get('config_mismatch') for counts in errors.values())
    for name, counts in groups.items():
        assert counts['episodes'] == 50 and errors[name]['decisions'] == 200
        counts['correct'] = errors[name].get('right', 0)
        counts['decisions'] = 200
        counts['accuracy'] = counts['correct'] / 200
        for kind in ('explicit', 'implicit', 'delayed'):
            total = counts[f'{kind}_total']
            counts[f'{kind}_accuracy'] = counts[f'{kind}_correct'] / total if total else None
        counts['mean_cost'] = counts.pop('cost_sum') / counts['episodes']
        counts['mean_peak_resident'] = counts.pop('peak_tokens_sum') / counts['episodes']
    temporary.replace(report_raw)
    write(R/'summary.json', dict(groups=groups, error_analysis=errors))
    write(R/'complete.json', dict(status='T1_EVALUATION_COMPLETE', episodes=len(seen),
          decisions=len(seen)*4, checkpoint=read(R/'checkpoint-accepted.json'),
          records_sha256=sha256(report_raw), summary_sha256=sha256(R/'summary.json'), time=time.time()))
    health('COMPLETE', episodes=len(seen))


def main():
    active = {}
    with contextlib.ExitStack() as stack:
        lock = stack.enter_context((R/'controller.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (R/'identity.json').exists(), 'Controller already launched'
        write(R/'identity.json', identity(os.getpid()))
        try:
            plan, launch = preflight()
            wait_for_sft(plan, launch)
            acquire_lock(stack, H/'campaign.lock')
            acquire_lock(stack, OLD/'campaign.lock')
            dispatch('main', [(100000, 100001)], active)
            dispatch('main', [(100001, 100005)] + [(s, s+5) for s in range(100005, 100050, 5)], active)
            dispatch('controls', [(s, s+5) for s in range(100000, 100050, 5)], active)
            health('VERIFY_RESULTS')
            finalize()
        except BaseException:
            write(R/'error.json', dict(traceback=traceback.format_exc(), time=time.time()))
            for process, ident, directory, _, log in active.values():
                if process.poll() is None and alive(ident):
                    os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                log.close()
                write(directory/'exit.json', dict(returncode=process.returncode, identity=ident, time=time.time()))
            raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        plan, launch = preflight()
        print(json.dumps(dict(steps=plan['optimizer_steps'], training=launch['identity'],
              conditions={phase: len(expected_keys(methods, SEEDS.start, SEEDS.stop))
                          for phase, methods in PHASES.items()})))
    else:
        main()
