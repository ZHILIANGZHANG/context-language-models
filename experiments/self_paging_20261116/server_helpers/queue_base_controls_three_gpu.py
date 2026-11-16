"""Exclusive base-controls campaign, then resume the deferred RL profile."""
import collections
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET

import queue_rl_profile as shared

H, C, N, P = shared.H, shared.C, shared.N, shared.P
R = H / 'base-controls-two'
S = H / 'quality-training-val50'
ACTIVE = {}

class AdoptedWorker:
    def __init__(self, identity, directory):
        self.identity = identity
        self.pid = identity['pid']
        self.directory = directory
        self.returncode = None
        self.adopted = True

    def poll(self):
        if alive(self.identity):
            return None
        # Original parent's wait status is unavailable. Require the worker's
        # durable completion marker and validate its rows instead.
        self.returncode = 0 if (self.directory/'accepted.json').exists() else 1
        return self.returncode

    def wait(self, timeout=30):
        deadline = time.time()+timeout
        while self.poll() is None:
            if time.time() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(.2)
        return self.returncode

write, ident, alive, free = shared.write, shared.ident, shared.alive, shared.free
shared.R = R

def stop():
    if any(p.exists() for p in [H/'STOP', R/'STOP', S/'STOP', H/'quality-review/STOP']):
        raise RuntimeError('User STOP')

def verify():
    shared.verify_source()
    manifest = json.loads((H/'base-controls-three-gpu-helpers.json').read_text())
    for f, sha in manifest.items():
        assert hashlib.sha256((H/f).read_bytes()).hexdigest() == sha, f

def dispatch(jobs, phase):
    pending = list(jobs)
    while pending or ACTIVE:
        stop()
        for gpu, (p, i, d, f) in list(ACTIVE.items()):
            if p.poll() is not None:
                f.close()
                write(d/'exit.json', dict(returncode=None if getattr(p,'adopted',False) else p.returncode, identity=i, time=time.time(), adopted=getattr(p,'adopted',False), completion_marker_valid=p.returncode==0))
                del ACTIVE[gpu]
                if p.returncode:
                    raise RuntimeError(f'Worker failed: {d}')
                assert json.loads((d/'accepted.json').read_text())['episodes'] == json.loads((d/'launch.json').read_text())['expected']
        if pending:
            first = free()
            time.sleep(3)
            for gpu in sorted(first & free()):
                if not pending or gpu in ACTIVE:
                    continue
                verify()
                lo, hi = pending.pop(0)
                d = R / f'job-{lo}'
                d.mkdir(exist_ok=False)
                f = (d/'stdout.log').open('w')
                args = [P, str(H/'base_controls_two_worker.py'), '--start', str(lo), '--end', str(hi)]
                proc = subprocess.Popen(args, cwd=C, env=os.environ | {
                    'CUDA_VISIBLE_DEVICES':str(gpu), 'PYTHONPATH':str(C),
                    'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2',
                    'TOKENIZERS_PARALLELISM':'false'}, stdout=f, stderr=subprocess.STDOUT,
                    start_new_session=True)
                i = ident(proc.pid)
                ACTIVE[gpu] = proc, i, d, f
                write(d/'launch.json', dict(identity=i, gpu=gpu, cmd=args, expected=(hi-lo)*6, time=time.time()))
        write(R/'health.json', dict(time=time.time(), phase=phase, pending=pending,
              active={g:v[1] for g,v in ACTIVE.items()}))
        time.sleep(10)

def finalize():
    from base_controls_two_worker import CONDITIONS
    from self_paging.core.costs import episode_cost
    rows = [json.loads(l) for p in R.glob('job-*/summary.jsonl') for l in p.read_text().splitlines()]
    expected = {(s,m,r) for s in range(100000,100050) for m,r in CONDITIONS}
    key = lambda r:(r['seed'],r['method'],r['ratio'])
    assert len(rows) == 300 and {key(r) for r in rows} == expected
    sft = [json.loads(l) for p in S.glob('eval-*/summary.jsonl') for l in p.read_text().splitlines()]
    assert len(sft)==200
    # Compare actual worlds and budget tokens to their SFT counterparts.
    worlds = {}
    for p in sorted(N.glob('sft-quality-val50-eval-*.jsonl')):
        for line in p.open():
            rec = json.loads(line)
            world = rec['environment_pages']
            assert rec['seed'] not in worlds or worlds[rec['seed']] == world
            worlds[rec['seed']] = world
    budgets = {(r['seed'],r['ratio']):r['budget'] for r in sft}
    for row in rows:
        path = Path(row['raw_path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['raw_sha256']
        rec = json.loads(gzip.decompress(path.read_bytes()))
        assert rec['environment_pages'] == worlds[row['seed']]
        assert row['budget'] == budgets[(row['seed'],row['ratio'])]
        assert row['reward'] == sum(d['gold']==d['predicted'] for d in rec['decisions'])/4
        assert abs(episode_cost(rec)['cost']-row['cost']) < max(1e-5, row['cost']*1e-8)
    grouped = collections.defaultdict(list)
    for weight, data in [('Base',rows),('SFT',sft)]:
        for r in data:
            grouped[(weight,r['method'],r['ratio'])].append(r)
    table = []
    for (weight, method, ratio), values in grouped.items():
        assert len(values)==50
        table.append(dict(weights=weight, method=method, ratio=ratio, episodes=50,
            decision_accuracy=sum(r['reward'] for r in values)/50,
            perfect_episodes=sum(r['reward']==1 for r in values),
            mean_peak_resident=sum(r['peak_resident'] for r in values)/50,
            mean_cost=sum(r['cost'] for r in values)/50,
            forced_episodes=sum(r['forced_evictions']>0 for r in values),
            kv_errors=sum(r['kv_errors'] for r in values)))
    write(R/'comparison.json',table)
    lines = ['# Base and SFT paired T1 evaluation', '',
             '50 identical worlds, four decisions each. Temperature 0.6, top-p 0.95, top-k 20, paired RNG seed. Costs are analytical.',
             'QuestPage is a page-key approximation. Baselines run on original Qwen3-8B; model-controlled paging runs on SFT weights.',
             '', '| Weights | Method | Budget ratio | Correct decisions | Perfect episodes | Mean peak tokens | Forced episodes | KV errors |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['weights']} | {r['method']} | {r['ratio']} | {round(r['decision_accuracy']*200)}/200 | {r['perfect_episodes']}/50 | {r['mean_peak_resident']:.1f} | {r['forced_episodes']} | {r['kv_errors']} |")
    (R/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(R/'complete.json',dict(status='BASE_TWO_300_AND_SFT_200_PAIRED_VALIDATED',time=time.time(),episodes=300))

def main():
    R.mkdir(exist_ok=True)
    lock = open(R/'controller.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (R/'identity.json').exists()
    write(R/'identity.json',ident(os.getpid()))
    while not (S/'complete.json').exists() or alive(json.loads((S/'identity.json').read_text())):
        stop()
        if (S/'error.json').exists():
            raise RuntimeError('SFT dependency failed')
        write(R/'health.json',dict(time=time.time(),phase='WAIT_SFT_EVALUATION'))
        time.sleep(10)
    verify()
    suites = ET.parse(H/'base-controls-cpu.xml').getroot().iter('testsuite')
    totals = {name:0 for name in ['tests','failures','errors','skipped']}
    for suite in suites:
        for name in totals:
            totals[name] += int(suite.get(name,0))
    assert totals == dict(tests=8,failures=0,errors=0,skipped=0), totals
    assert json.loads((S/'complete.json').read_text())['episodes']==200
    campaign = open(H/'campaign.lock','a')
    while True:
        stop()
        try:
            fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(10)
    assert (R/'sft-evaluation-integrity.json').exists()
    launch = json.loads((R/'job-100000/launch.json').read_text())
    directory = R/'job-100000'
    proc = AdoptedWorker(launch['identity'],directory)
    if proc.poll() is None:
        ACTIVE[launch['gpu']] = (proc,launch['identity'],directory,(directory/'adoption.log').open('a'))
    else:
        assert proc.returncode == 0
        assert json.loads((directory/'accepted.json').read_text())['episodes'] == 6
    
    dispatch([(100001,100005)]+[(s,s+5) for s in range(100005,100050,5)],'BASE_CONTROLS')
    finalize()
    fcntl.flock(campaign,fcntl.LOCK_UN)
    # Original profile controller is deferred until the validated controls finish.
    stop()
    profile = H/'rl-profile64'
    assert not (profile/'identity.json').exists()
    out = (profile/'controller.log').open('a')
    proc = subprocess.Popen([P,str(H/'queue_rl_profile.py')],cwd=C,
        env=os.environ | {'PYTHONPATH':str(C),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
        stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    write(R/'rl-profile-resumed.json',dict(identity=ident(proc.pid),time=time.time()))

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        for p,i,d,f in ACTIVE.values():
            if p.poll() is None and alive(i):
                os.killpg(p.pid,signal.SIGTERM)
        for p,i,d,f in ACTIVE.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
