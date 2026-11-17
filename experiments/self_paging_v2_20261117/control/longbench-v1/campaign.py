"""Run LongBench on one suitable GPU, admitting only our existing T1 job for sharing."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'src'
T1 = ROOT.parent / 'faithful-baselines'
RESULTS = Path('/nas/Public/zhangcheng/self-paging-v2-20261117/longbench-v1')
BASE = Path('/nas/Public/zhangcheng/models/Qwen3-8B')
ADAPTER = RESULTS.parent / 'checkpoints/sft-v2-r64'
MANIFEST = ROOT / 'manifest.json'
WORKER = 'self_paging.eval.longbench_v1'
ALLOWED_GPUS = {0, 1, 2}
MEMORY_FRACTION = 0.46


def write(name, obj):
    dst = ROOT / name
    tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(dst)


def health(stage, **info):
    write('health.json', {'stage': stage, 'time': time.time(), **info})


def process_identity(pid):
    p = Path('/proc') / str(pid)
    return {'pid': pid, 'uid': p.stat().st_uid, 'cwd': str((p / 'cwd').resolve(strict=True)),
            'start_ticks': int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19])}


def query(kind, fields):
    text = subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                                   text=True, timeout=10)
    return [[x.strip() for x in row] for row in csv.reader(text.splitlines())]


def candidates():
    records = json.loads((T1 / 'health.json').read_text()).get('active', {})
    compute = {}
    for uuid, pid in query('compute-apps', 'gpu_uuid,pid'):
        compute.setdefault(uuid, []).append(int(pid))
    out = {}
    for gpu, uuid, free, total in query('gpu', 'index,uuid,memory.free,memory.total'):
        if int(gpu) not in ALLOWED_GPUS or int(free) < int(total) * MEMORY_FRACTION + 4096:
            continue
        ids = compute.get(uuid, [])
        if ids:
            active = records.get(gpu, {}).get('identity')
            if not active or len(ids) != 1 or ids[0] != active['pid']:
                continue
            try:
                if process_identity(ids[0]) != active:
                    continue
            except FileNotFoundError:
                continue
        out[int(gpu)] = {'shared_with': ids, 'free_mib': int(free)}
    return out


def stopped():
    return (ROOT / 'STOP').exists()


def main():
    with (ROOT / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'identity.json').exists():
            raise RuntimeError('LongBench campaign already started')
        write('identity.json', process_identity(os.getpid()))
        try:
            execute()
        except BaseException:
            write('error.json', {'traceback': traceback.format_exc(), 'time': time.time()})
            raise


def execute():
    manifest = json.loads(MANIFEST.read_text())
    items = manifest['items']
    # Start with a genuine long example before cycling across task categories.
    first = min((i for i in range(6) if items[i]['context_tokens'] > 5000),
                key=lambda i: items[i]['context_tokens'])
    indices = [first] + [i for j in range(6) for i in range(j, len(items), 6) if i != first]
    while not stopped():
        a = candidates()
        time.sleep(3)
        b = candidates()
        stable = {g for g in a.keys() & b.keys() if a[g]['shared_with'] == b[g]['shared_with']}
        if stable and not stopped():
            gpu = min(stable, key=lambda g: (bool(b[g]['shared_with']), -b[g]['free_mib'], g))
            break
        health('WAIT_GPU', candidates=b)
        time.sleep(12)
    else:
        health('STOPPED')
        return
    env = os.environ | {'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                        'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT / 'scorer-deps-pypi')}
    cmd = [sys.executable, '-m', WORKER, 'run', '--manifest', str(MANIFEST),
           '--official', str(ROOT / 'official'), '--model', str(ADAPTER), '--tokenizer', str(BASE),
           '--out', str(RESULTS), '--cuda-memory-fraction', str(MEMORY_FRACTION), '--indices', *map(str, indices)]
    with (ROOT / 'worker.log').open('x') as log:
        worker = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        ident = process_identity(worker.pid)
        write('launch.json', {'worker': ident, 'gpu': gpu, **b[gpu], 'first_index': first, 'time': time.time()})
        def interrupt(*_):
            if worker.poll() is None and process_identity(worker.pid) == ident:
                os.killpg(worker.pid, signal.SIGTERM)
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        while worker.poll() is None:
            if stopped():
                interrupt()
            health('RUNNING', gpu=gpu, worker=ident,
                   completed=sum(not p.name.endswith('.trajectory.json') for p in RESULTS.glob('*.json')),
                   target=len(items) * 5)
            time.sleep(5)
        write('exit.json', {'returncode': worker.returncode, 'worker': ident, 'time': time.time()})
        if stopped():
            health('STOPPED')
            return
        if worker.returncode:
            raise RuntimeError(f'LongBench worker exited {worker.returncode}')
    subprocess.run([sys.executable, '-m', WORKER, 'report', '--manifest', str(MANIFEST),
                    '--results', str(RESULTS), '--out', str(ROOT / 'summary.json')], cwd=SOURCE, env=env, check=True)
    summary = json.loads((ROOT / 'summary.json').read_text())
    if summary['common'] != len(items) or summary['completed'] != summary['target']:
        raise ValueError('LongBench result coverage incomplete')
    health('COMPLETE', completed=summary['completed'])


if __name__ == '__main__':
    main()
