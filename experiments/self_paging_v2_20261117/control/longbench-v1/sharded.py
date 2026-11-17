"""Continue LongBench on separate GPUs after the single-worker launch stops."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

from campaign import (ADAPTER, ALLOWED_GPUS, BASE, MANIFEST, RESULTS, ROOT, SOURCE, WORKER,
                      process_identity, query)

CONTROL = ROOT / 'parallel-v2'


def record(name, **values):
    path = CONTROL / f'{name}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2) + '\n')
    tmp.replace(path)


def free_gpus():
    busy = {uuid for uuid, _ in query('compute-apps', 'gpu_uuid,pid')}
    return {int(i) for i, uuid, mem, util in query('gpu', 'index,uuid,memory.used,utilization.gpu')
            if int(i) in ALLOWED_GPUS and uuid not in busy and int(mem) < 500 and int(util) == 0}


def stop():
    return (CONTROL / 'STOP').exists() or (ROOT.parent / 'STOP').exists()


def partitions(items, gpus):
    chunks = {gpu: [] for gpu in gpus}
    load = dict.fromkeys(gpus, 0)
    for i in sorted(range(len(items)), key=lambda j: -items[j]['context_tokens']):
        gpu = min(gpus, key=lambda g: (load[g], g))
        chunks[gpu].append(i)
        load[gpu] += max(items[i]['context_tokens'] - 4096, 0) + 1500
    for shard in chunks.values():
        shard.sort(key=lambda i: (i != 3, i % 6, i))
    assert sorted(i for shard in chunks.values() for i in shard) == list(range(len(items)))
    return chunks


def main():
    workers = {}
    with ExitStack() as stack:
        for path in (ROOT / 'campaign.lock', CONTROL / 'campaign.lock'):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / 'identity.json').exists():
            raise RuntimeError('Sharded campaign already started')
        record('identity', **process_identity(os.getpid()))
        try:
            execute(stack, workers)
        except BaseException:
            record('error', traceback=traceback.format_exc(), time=time.time())
            terminate(workers)
            raise


def terminate(workers):
    for child, identity in workers.values():
        if child.poll() is None and process_identity(child.pid) == identity:
            os.killpg(child.pid, signal.SIGTERM)
    for child, _ in workers.values():
        child.wait()


def execute(stack, workers):
    items = json.loads(MANIFEST.read_text())['items']
    batch = []
    while not stop() and not batch:
        before = free_gpus()
        time.sleep(3)
        batch = sorted(before & free_gpus())
        if not batch:
            record('health', stage='WAIT_GPU', time=time.time())
            time.sleep(10)
    if stop():
        record('health', stage='STOPPED', time=time.time())
        return
    selected = partitions(items, batch)
    base_env = os.environ | {'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                             'PYTHONPATH': str(ROOT / 'scorer-deps-pypi')}
    for gpu, indices in selected.items():
        if gpu not in free_gpus():
            raise RuntimeError(f'GPU {gpu} became occupied before dispatch')
        log = stack.enter_context((CONTROL / f'gpu{gpu}.log').open('x'))
        cmd = [sys.executable, '-m', WORKER, 'run', '--manifest', str(MANIFEST), '--official', str(ROOT / 'official'),
               '--model', str(ADAPTER), '--tokenizer', str(BASE), '--out', str(RESULTS),
               '--indices', *map(str, indices)]
        env = base_env | {'CUDA_VISIBLE_DEVICES': str(gpu)}
        child = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        workers[gpu] = child, process_identity(child.pid)
    record('launch', workers={gpu: ident for gpu, (_, ident) in workers.items()}, indices=selected, time=time.time())
    signal.signal(signal.SIGTERM, lambda *_: terminate(workers))
    while True:
        if stop():
            terminate(workers)
            record('health', stage='STOPPED', time=time.time())
            return
        codes = {gpu: child.poll() for gpu, (child, _) in workers.items()}
        failures = {g: c for g, c in codes.items() if c not in (None, 0)}
        if failures:
            record('exit', codes=codes, time=time.time())
            raise RuntimeError(f'LongBench worker failed: {failures}')
        completed = sum(not p.name.endswith('.trajectory.json') for p in RESULTS.glob('*.json'))
        record('health', stage='RUNNING', time=time.time(), codes=codes, completed=completed, target=len(items)*5)
        if None not in codes.values():
            break
        time.sleep(5)
    record('exit', codes=codes, time=time.time())
    subprocess.run([sys.executable, '-m', WORKER, 'report', '--manifest', str(MANIFEST), '--results', str(RESULTS),
                    '--out', str(CONTROL / 'summary.json')], cwd=SOURCE, env=base_env, check=True)
    report = json.loads((CONTROL / 'summary.json').read_text())
    if report['common'] != len(items) or report['completed'] != report['target']:
        raise ValueError('LongBench result coverage incomplete')
    record('health', stage='COMPLETE', completed=report['completed'], time=time.time())


if __name__ == '__main__':
    main()
