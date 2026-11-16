"""Wait for CPU pool and validated SFT evaluation; acquire existing GPU lease.
Runs only the authorized first64 x4 profile, never RL training or auto retries.
"""
import collections,fcntl,hashlib,json,math,os,pathlib,signal,subprocess,time,traceback
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'rl-profile64';D=H/'rl-preparation'
S=H/'quality-training-val50';C=pathlib.Path('/data/zhangcheng/context-language-models-223cc65-rlprep')
N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116');P='/data/zhangcheng/self-paging-20261116/venv/bin/python'
ACTIVE={}
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def ident(pid):
    p=pathlib.Path('/proc')/str(pid);return dict(pid=pid,uid=p.stat().st_uid,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve()))
def alive(i):
    try:return ident(i['pid'])==i
    except FileNotFoundError:return False
def stop():
    if any(p.exists() for p in (H/'STOP',R/'STOP',D/'STOP',H/'quality-review/STOP',S/'STOP')):raise RuntimeError('User STOP')
def free():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(i) for i,u,m,v in (r.split(',') for r in rows) if int(m)<500 and int(v)==0 and u.strip() not in apps}
def verify_source():
    m=json.loads((H/'rl-prep-source-manifest.json').read_text())
    assert all(hashlib.sha256((C/f).read_bytes()).hexdigest()==sha for f,sha in m.items())
    helpers=json.loads((H/'rl-prep-helpers-manifest.json').read_text())
    assert all(hashlib.sha256((H/f).read_bytes()).hexdigest()==sha for f,sha in helpers.items())
def sft_audit():
    from audit_protocol import audit
    from self_paging.train.common import load_tokenizer
    from self_paging.core.costs import episode_cost
    tok=load_tokenizer('/nas/Public/zhangcheng/models/Qwen3-8B')
    m=json.loads((H/'val50-source-manifest.json').read_text())
    assert all(hashlib.sha256((pathlib.Path('/data/zhangcheng/context-language-models-797680f-val50')/f).read_bytes()).hexdigest()==sha for f,sha in m.items())
    expected={(s,m,r) for s in range(100000,100050) for m,r in [('full',1.),('model',.25),('model',.4),('model',.6)]}
    key=lambda x:(x['seed'],x['method'],x['ratio'])
    summaries=[json.loads(l) for p in S.glob('eval-*/summary.jsonl') for l in p.read_text().splitlines()]
    assert len(summaries)==200 and {key(r) for r in summaries}==expected
    lookup={key(r):r for r in summaries};seen=set();counts=collections.defaultdict(collections.Counter);sha={}
    for p in sorted(N.glob('sft-quality-val50-eval-*.jsonl')):
        stop();sha[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for l in p.open():
            rec=json.loads(l);k=key(rec);assert k in expected and k not in seen;seen.add(k)
            assert len(rec['decisions'])==4
            assert sum(d['gold']==d['predicted'] for d in rec['decisions'])/4==rec['reward']==lookup[k]['reward']
            assert rec['meta']['over_budget']==0 and len(rec['tokens'])==lookup[k]['tokens']<=32768
            cost=episode_cost(rec)['cost'];assert abs(cost-lookup[k]['cost'])<max(1e-5,cost*1e-8)
            a=audit(rec,tok);counts[str(k[1:])].update(a['counts'])
    assert seen==expected
    # Record actual protocol quality; profiling is a bounded diagnostic, not RL admission.
    errors={k:v['environment_errors']/max(1,v['environment_calls']) for k,v in counts.items()}
    result=dict(time=time.time(),episodes=200,raw_sha256=sha,environment_error_rates=errors,
                protocol_under_five_percent=all(x<=.05 for k,x in errors.items() if 'model' in k),
                scope='SFT results integrity accepted; profile permitted for diagnosis, NOT permission to train RL')
    write(R/'sft-evaluation-integrity.json',result)
    checkpoint=N/'checkpoints/sft-quality-val50-r64'
    write(R/'checkpoint.json',dict(path=str(checkpoint),sha256={f:hashlib.sha256((checkpoint/f).read_bytes()).hexdigest() for f in ['adapter_model.safetensors','adapter_config.json']}))
def main():
    R.mkdir(exist_ok=True);lock=open(R/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (R/'identity.json').exists();write(R/'identity.json',ident(os.getpid()))
    while True:
        stop()
        if (D/'error.json').exists() or (S/'error.json').exists():raise RuntimeError('Dependency failed; preserve and diagnose')
        if (D/'complete.json').exists() and (S/'complete.json').exists():break
        write(R/'health.json',dict(time=time.time(),stage='WAIT_CPU_POOL_AND_SFT_EVALUATION',cpu_complete=(D/'complete.json').exists(),sft_complete=(S/'complete.json').exists()));time.sleep(20)
    cpu=json.loads((D/'complete.json').read_text());assert cpu['status']=='CPU_POOL_ACCEPTED_GPU_PROFILE_PENDING'
    assert json.loads((S/'complete.json').read_text())['episodes']==200
    old=json.loads((S/'identity.json').read_text())
    while alive(old):stop();time.sleep(5)
    verify_source();sft_audit()
    pool=N/'data/rl-preparation/profile64.jsonl'
    assert hashlib.sha256(pool.read_bytes()).hexdigest()==cpu['sha256']['profile64']
    prompts=[json.loads(l) for l in pool.read_text().splitlines()];assert len(prompts)==len({r['prompt_id'] for r in prompts})==64
    assert all(r['split']=='train' and r['train_eligible'] for r in prompts)
    for shard in range(8):write(R/f'shard-{shard}.json',prompts[shard::8])
    campaign=open(H/'campaign.lock','a')
    while True:
        stop()
        try:fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(10)
    pending=list(range(8));failed=[]
    while pending or ACTIVE:
        stop()
        for gpu,(p,identity,d,f) in list(ACTIVE.items()):
            if p.poll() is not None:
                f.close();write(d/'exit.json',dict(returncode=p.returncode,time=time.time(),identity=identity))
                if p.returncode:failed.append(str(d))
                del ACTIVE[gpu]
        if failed:raise RuntimeError('Profile failure; no blind retry: '+str(failed))
        if pending:
            a=free();time.sleep(3)
            for gpu in sorted(a&free()):
                if not pending or gpu in ACTIVE:continue
                verify_source();shard=pending.pop(0);d=R/f'worker-{shard}';assert not d.exists();d.mkdir()
                f=(d/'stdout.log').open('w');args=[P,str(H/'rl_profile_worker.py'),'--shard',str(shard)]
                p=subprocess.Popen(args,cwd=C,env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(C),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                i=ident(p.pid);ACTIVE[gpu]=(p,i,d,f);write(d/'launch.json',dict(identity=i,gpu=gpu,cmd=args,time=time.time()))
        write(R/'health.json',dict(time=time.time(),stage='PROFILE64',pending=pending,active={k:v[1] for k,v in ACTIVE.items()}));time.sleep(10)
    rows=[json.loads(l) for p in R.glob('worker-*/summary.jsonl') for l in p.read_text().splitlines()]
    assert len(rows)==len({(r['prompt_id'],r['member']) for r in rows})==256
    profiles=[]
    for p in prompts:
        rs=[r for r in rows if r['prompt_id']==p['prompt_id']];assert len(rs)==4
        rewards=[r['reward'] for r in rs];success=[r for r in rs if r['reward']==1.]
        profiles.append(dict(prompt_id=p['prompt_id'],ratio=p['ratio'],n_decisions=p['n_decisions'],rewards=rewards,mean_reward=sum(rewards)/4,
                             success_rate=len(success)/4,task_signal=max(rewards)>min(rewards),
                             potential_efficiency_signal=len(success)>=2 and len({r['cost'] for r in success})>1,
                             mean_seconds=sum(r['seconds'] for r in rs)/4))
    write(R/'profiles.json',profiles)
    write(R/'complete.json',dict(time=time.time(),status='PROFILE64_COMPLETE_REVIEW_BEFORE_EXPANSION_OR_RL',prompts=64,episodes=256,
          task_signal_groups=sum(p['task_signal'] for p in profiles),potential_efficiency_groups=sum(p['potential_efficiency_signal'] for p in profiles)))
if __name__=='__main__':
    try:main()
    except BaseException:
        for p,i,d,f in ACTIVE.values():
            if p.poll() is None and alive(i):os.killpg(p.pid,signal.SIGTERM)
        for p,i,d,f in ACTIVE.values():
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
