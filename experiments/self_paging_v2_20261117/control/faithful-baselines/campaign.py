"""Unique corrected baseline campaign; stops superseded queue, never resumes it."""
import sys,os,json,time,hashlib,fcntl,subprocess,contextlib,traceback,signal,statistics,random,math
from pathlib import Path
H=Path('/home/zhangcheng/self-paging-v2-20261117');R=H/'faithful-baselines'
C=Path('/data/zhangcheng/context-language-models-faithful-20261117')
N=Path('/nas/Public/zhangcheng/self-paging-v2-20261117/faithful-baselines')
sys.path.insert(0,str(H/'post-sft-eval'))
from post_sft_eval import identity,alive,free_gpus,read,write,sha256,OLD,CHECKPOINT,BASE
RATIOS=(.25,.35,.4,.5,.6); METHODS=('full','model','streaming','h2o')


def stop():
    if (H/'STOP').exists() or (R/'STOP').exists():raise RuntimeError('User STOP')


def health(phase,**kw):write(R/'health.json',dict(phase=phase,time=time.time(),**kw))


def preflight():
    import importlib.metadata
    for package,version in read(OLD/'env-spec.json')['pins'].items():assert importlib.metadata.version(package)==version,package
    for name,expected in read(R/'source-manifest.json').items():assert sha256(C/name)==expected,name
    for method,ref in read(R/'references/manifest.json').items():
        for name,expected in ref['files'].items():assert sha256(R/'references'/method/name)==expected
    cpu=read(R/'cpu-accepted.json');assert cpu['passed']>=11 and not cpu['failed'] and not cpu['skipped']
    assert sha256(R/'cpu-tests.log')==cpu['log_sha256']
    for old in ('post-sft-eval','token-baselines-9c7dbd5'):
        assert not alive(read(H/old/'identity.json'))
        assert (H/old/'STOP').exists()
        for path in (H/old).glob('*/launch.json'):
            d=read(path);assert not alive(d['identity']),str(path)
    accepted=read(H/'post-sft-eval/checkpoint-accepted.json')
    assert accepted['completed_steps']==516
    for name,value in accepted['sha256'].items():assert sha256(CHECKPOINT/name)==value
    write(R/'checkpoint-accepted.json',accepted)
    write(R/'plan.json',dict(methods=METHODS,ratios=RATIOS,seeds=[100000,100050],
        temperature=0,episodes=800,decisions=3200,checkpoint=str(CHECKPOINT),source=str(C),
        allowed_gpus=[0,1,2],max_turns=60,max_new_tokens=1024,max_log_tokens=32768,
        gating='author-code equivalence, CUDA native parity and real first-world canary; NO score ordering gate'))


def cmd(module,*args):
    return [sys.executable,'-m','self_paging.eval.'+module,'--model',str(CHECKPOINT),'--tokenizer',BASE,*map(str,args)]


def evaluate_job(name,lo,hi,method,ratios):
    args=cmd('faithful_eval','--out',N/name,'--seeds',lo,hi,'--methods',method,'--ratios',*ratios)
    return dict(name=name,args=args,lo=lo,hi=hi,method=method,ratios=list(ratios))


def validate(job):
    if job['name']=='cuda-probe':
        assert read(R/'cuda-probe.json')['passed'];return
    directory=N/job['name'];count=0
    for seed in range(job['lo'],job['hi']):
        for ratio in ([1.] if job['method']=='full' else job['ratios']):
            name=f"{seed}-{job['method']}-{ratio}";row=read(directory/(name+'.json'));rawpath=directory/(name+'.trajectory.json');raw=read(rawpath)
            assert sha256(rawpath)==row['trajectory_sha256']
            assert row['reward']==sum(d['gold']==d['predicted'] for d in raw['decisions'])/4
            assert len(raw['decisions'])==4 and row['tokens']<=32768 and not raw['meta']['over_budget']
            assert math.isfinite(row['mean_resident_decode'])
            if row['budget'] is not None:assert row['peak_resident']<=row['budget']
            if job['method']!='model':
                assert all(raw['meta'].get(k,0)==0 for k in ['kv_in','kv_out','notes','forced_evictions','truncated_tokens'])
            if job['method'] in ('streaming','h2o'):
                assert raw['cache_capacity']==row['budget']-1
                assert all(x['slots']<=raw['cache_capacity'] and x['keys_finite'] and x['values_finite'] and x['positions_ordered'] for x in raw['cache_final_state'])
            count+=1
    write(R/job['name']/'accepted.json',dict(episodes=count,time=time.time()))


def run_jobs(phase,jobs,active):
    pending=list(jobs)
    while pending or active:
        stop()
        for gpu,(p,ident,j,f) in list(active.items()):
            if p.poll() is None:continue
            f.close();del active[gpu]
            write(R/j['name']/'exit.json',dict(returncode=p.returncode,identity=ident,time=time.time()))
            if p.returncode:raise RuntimeError('Worker failed '+j['name'])
            validate(j)
        if pending:
            first=free_gpus();time.sleep(3)
            for gpu in sorted(first&free_gpus()-active.keys()):
                if not pending:break
                stop();j=pending.pop(0);d=R/j['name'];d.mkdir(exist_ok=False)
                f=(d/'stdout.log').open('w')
                p=subprocess.Popen(j['args'],cwd=C,stdout=f,stderr=subprocess.STDOUT,
                   env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'},start_new_session=True)
                ident=identity(p.pid);assert ident['uid']==os.getuid() and ident['cwd']==str(C)
                active[gpu]=(p,ident,j,f);write(d/'launch.json',dict(identity=ident,gpu=gpu,job=j,time=time.time()))
        health(phase,pending=[j['name'] for j in pending],active={str(g):dict(identity=x[1],job=x[2]['name']) for g,x in active.items()})
        if pending or active:time.sleep(10)


def finalize():
    groups={};seen=set();worlds={};budgets={};scores={}
    for accepted in R.glob('*/accepted.json'):
        directory=N/accepted.parent.name
        for p in directory.glob('*.json'):
            if p.name.endswith('.trajectory.json'):continue
            row=read(p);raw=read(p.with_name(p.stem+'.trajectory.json'));key=(row['seed'],row['method'],row['ratio']);assert key not in seen;seen.add(key)
            assert worlds.setdefault(row['seed'],raw['environment_pages'])==raw['environment_pages']
            if row['method']!='full':assert budgets.setdefault((row['seed'],row['ratio']),row['budget'])==row['budget']
            group=f"{row['method']}@{row['ratio']}";groups.setdefault(group,[]).append((row,raw));scores.setdefault(group,{})[row['seed']]=row['reward']
    expected={(s,m,r) for s in range(100000,100050) for m in METHODS for r in ([1.] if m=='full' else RATIOS)}
    assert seen==expected
    report={}
    for g,items in groups.items():
        types={};missing=0
        for row,raw in items:
            for d in raw['decisions']:
                cell=types.setdefault(d['kind'],[0,0]);cell[0]+=int(d['predicted']==d['gold']);cell[1]+=1;missing+=int(d['predicted'] is None)
        report[g]=dict(correct=sum(int(r['reward']*4) for r,_ in items),decisions=len(items)*4,
            accuracy=statistics.mean(r['reward'] for r,_ in items),by_type=types,unsubmitted=missing,
            mean_peak=statistics.mean(r['peak_resident'] for r,_ in items),max_peak=max(r['peak_resident'] for r,_ in items),
            mean_decode_kv=statistics.mean(r['mean_resident_decode'] for r,_ in items),
            unoffered_kv_calls=sum(r['meta'].get('unoffered_kv_calls',0) for r,_ in items))
    paired={}
    for ratio in RATIOS:
        for m in ['streaming','h2o']:
            delta=[scores[f'model@{ratio}'][s]-scores[f'{m}@{ratio}'][s] for s in sorted(worlds)]
            rng=random.Random(0);draws=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(4000))
            paired[f'model-minus-{m}@{ratio}']=dict(delta=statistics.mean(delta),ci95=[draws[100],draws[3899]],unit='world')
    write(R/'summary.json',dict(groups=report,paired=paired))
    write(R/'complete.json',dict(status='VALIDATED',episodes=len(seen),summary_sha256=sha256(R/'summary.json'),time=time.time()))
    health('COMPLETE',episodes=len(seen))


def main():
    active={}
    with contextlib.ExitStack() as stack:
        for path in [R/'controller.lock',H/'campaign.lock',OLD/'campaign.lock']:
            f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (R/'identity.json').exists()
        write(R/'identity.json',identity(os.getpid()))
        try:
            stop();preflight();N.mkdir(exist_ok=False)
            probe=dict(name='cuda-probe',args=cmd('faithful_probe','--out',R/'cuda-probe.json'))
            run_jobs('CUDA_PROBE',[probe],active)
            canary=[evaluate_job('canary-'+m,100000,100001,m,[.25]) for m in METHODS]
            run_jobs('CANARY',canary,active)
            write(R/'canary-accepted.json',dict(jobs=[j['name'] for j in canary],time=time.time()))
            jobs=[evaluate_job('first-rest-'+m,100000,100001,m,RATIOS[1:]) for m in METHODS if m!='full']
            for lo in range(100001,100050,5):
                for m in METHODS:jobs.append(evaluate_job(f't1-{lo}-{m}',lo,min(lo+5,100050),m,RATIOS))
            run_jobs('T1_COMPARISON',jobs,active);finalize()
        except BaseException:
            write(R/'error.json',dict(traceback=traceback.format_exc(),time=time.time()))
            for p,ident,j,f in active.values():
                if p.poll() is None and alive(ident):os.killpg(p.pid,signal.SIGTERM)
                p.wait();f.close();write(R/j['name']/'exit.json',dict(returncode=p.returncode,identity=ident,time=time.time()))
            raise

if __name__=='__main__':main()
