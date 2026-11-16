"""Gated automatic continuation; no retries, no claims from exit code alone."""
import fcntl,hashlib,json,os,pathlib,signal,subprocess,time,traceback,xml.etree.ElementTree as ET
H=pathlib.Path('/home/zhangcheng/self-paging-20261116')
C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-val50')
N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116')
P='/data/zhangcheng/self-paging-20261116/venv/bin/python'
M='/nas/Public/zhangcheng/models/Qwen3-8B'
R=H/'quality-training-val50';R.mkdir(exist_ok=True)
CHILDREN={}
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def ident(pid):
 p=pathlib.Path('/proc')/str(pid)
 return dict(pid=pid,uid=p.stat().st_uid,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve()))
def stopped():
 if (H/'STOP').exists() or (R/'STOP').exists() or (H/'quality-review/STOP').exists():raise RuntimeError('User STOP')
def free():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 return {int(i) for i,u,m,v in (r.split(',') for r in rows) if int(m)<500 and int(v)==0 and u.strip() not in apps}
def check_source():
 m=json.loads((H/'val50-source-manifest.json').read_text())
 assert all(hashlib.sha256((C/k).read_bytes()).hexdigest()==v for k,v in m.items())
def idle_cards(minimum):
 while True:
  stopped();a=free();time.sleep(3);b=free();g=sorted(a&b)
  if len(g)>=minimum:return g
  write(R/'health.json',dict(time=time.time(),stage='waiting_free_cards',minimum=minimum,free=g));time.sleep(20)
def run(name,args,gpus):
 stopped();check_source();d=R/name;assert not d.exists(),d;d.mkdir()
 env=os.environ|{'PYTHONPATH':str(C),'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
 with (d/'stdout.log').open('w') as f:
  p=subprocess.Popen(args,cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  identity=ident(p.pid);write(d/'launch.json',dict(identity=identity,gpus=gpus,cmd=args,time=time.time()))
  CHILDREN[p.pid]=(p,identity)
  try:
   while p.poll() is None:
    stopped()
    if name=='sft-quality':
     if (R/'swanlab/error.json').exists():raise RuntimeError('SwanLab relay failed')
     rows=[json.loads(x) for x in (R/'sft.jsonl').read_text().splitlines() if x.endswith('}')] if (R/'sft.jsonl').exists() else []
     cloud=json.loads((R/'swanlab/cloud-verified.json').read_text()) if (R/'swanlab/cloud-verified.json').exists() else {}
     vals=[x['eval_step'] for x in rows if 'eval_loss' in x]
     if sum('step' in x for x in rows)>=3 and (not cloud or set(vals)-set(cloud.get('validation_matching_steps',[]))):
      verify_cloud()
    write(R/'health.json',dict(time=time.time(),stage=name,identity=identity,gpus=gpus));time.sleep(10)
  except BaseException:
   if p.poll() is None and ident(p.pid)==identity:os.killpg(p.pid,signal.SIGTERM)
   p.wait();raise
  write(d/'exit.json',dict(returncode=p.returncode,time=time.time(),identity=identity))
  CHILDREN.pop(p.pid,None)
  if p.returncode:raise RuntimeError(f'{name} failed; evidence retained, no blind retry')
def ddp(gpus,script,args):
 return [P,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(len(gpus)),script]+args
def verify_cloud():
 for attempt in range(6):
  stopped()
  with (R/f'cloud-verify-{attempt}.log').open('a') as f:rc=subprocess.call([P,str(H/'swanlab_val50_verify.py')],cwd=C,env=os.environ|{'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT)
  if rc==0:return
  time.sleep(20)
 raise RuntimeError('Cloud metrics could not be verified; stop owned training and preserve evidence')
def wait_specific(gpus):
 while True:
  stopped();a=free();time.sleep(3)
  if set(gpus)<=a&free():return
  write(R/'health.json',dict(stage='waiting_smoke_verified_cards',gpus=gpus,time=time.time()));time.sleep(15)
def main():
 lock=open(R/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'identity.json').exists(),'Existing controller evidence; no duplicate'
 write(R/'identity.json',ident(os.getpid()))
 D=N/'data/quality-accepted'
 while not (D/'data-accepted.json').exists():
  stopped()
  prior=H/'teacher-final-admission/complete.json'
  detail=json.loads(prior.read_text())['status'] if prior.exists() else 'generation_or_final_admission_pending'
  write(R/'health.json',dict(stage='WAITING_ACCEPTED_DATA',detail=detail,time=time.time()));time.sleep(15)
 check_source();accepted=json.loads((D/'data-accepted.json').read_text())
 assert accepted['status']=='DATA_ACCEPTED_TRAINING_SMOKE_STILL_REQUIRED' and accepted['zero_fallback'] is True
 assert accepted['counts']['train']['selected']==2400 and accepted['counts']['heldout']['selected']==240
 for split in ['train','heldout']:assert hashlib.sha256((D/f'{split}.jsonl').read_bytes()).hexdigest()==accepted['sha256'][split]
 suites=ET.parse(H/'quality-cpu.xml').getroot().findall('testsuite')
 assert sum(int(s.get('tests')) for s in suites)==73 and all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
 assert json.loads((H/'mask-gpu-accepted.json').read_text())['status']=='PASS'
 campaign=open(H/'campaign.lock','a')
 while True:
  stopped()
  try:fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:time.sleep(5)
 g=idle_cards(3);world=len(g)
 suites=ET.parse(H/'val50-cpu.xml').getroot().findall('testsuite')
 assert sum(int(x.get('tests')) for x in suites)>=5 and all(int(x.get(k,0))==0 for x in suites for k in ['failures','errors','skipped'])
 import random,math,gc
 from self_paging.train.sft import micro_batches
 records=[json.loads(x) for x in (D/'train.jsonl').read_text().splitlines()];assert len(records)==2400
 order=sorted(records,key=lambda r:len(r['tokens']))
 for name,rs in [('short',order[:world]),('long',order[-world:])]:
  (R/f'{name}-witness.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rs))
 steps=math.ceil(len(micro_batches(records,32768,random.Random(0)))/world)
 plan=dict(model='Qwen3-8B',upstream='797680fc691514587fb8cc298236c624ac3bbac8',source_manifest_sha256=hashlib.sha256((H/'val50-source-manifest.json').read_bytes()).hexdigest(),train_sha256=accepted['sha256']['train'],heldout_sha256=accepted['sha256']['heldout'],train_records=2400,heldout_records=240,epochs=1,lora_rank=64,lora_alpha=128,lr=1e-4,optimizer_steps=steps,seed=0,max_packed_tokens=32768,world_size=world,gpus=g,corpus='zero-fallback GPU rewrites of audited oracle plans; not autonomous teacher reasoning',plain_next_loss_retention=.25,checkpoint_initialization='original base; smoke updates discarded',eval_every=50,save_every=50,validation_records=240,checkpoint_kind='evaluable LoRA adapters; no optimizer resume state')
 write(R/'training-plan.json',plan)
 (R/'periodic-witness.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in order[:world]+order[-world:]))
 (R/'periodic-eval-witness.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in order[:world]))
 del records,order,rs;gc.collect()
 wait_specific(g)
 run('smoke-periodic',ddp(g,str(H/'val50_training_witness.py'),['--data',str(R/'periodic-witness.jsonl'),'--eval-data',str(R/'periodic-eval-witness.jsonl'),'--checkpoint-root',str(N/'checkpoints/val50-witness-discarded'),'--out',str(R/'periodic-accepted.json')]),g)
 result=json.loads((R/'periodic-accepted.json').read_text());assert result['world']==world and result['changed_tensors']>0 and result['checkpoint_discarded']
 # The online relay must authenticate privately before formal launch.
 f=(R/'swanlab-relay.log').open('w');relay=subprocess.Popen([P,str(H/'swanlab_val50_relay.py')],cwd=C,env=os.environ|{'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,start_new_session=True);ri=ident(relay.pid);CHILDREN[relay.pid]=(relay,ri);write(R/'relay-launch.json',dict(identity=ri,time=time.time()))
 deadline=time.time()+90
 while not (R/'swanlab/auth-ready.json').exists():
  stopped();assert relay.poll() is None and time.time()<deadline,'SwanLab authentication failed or timed out';time.sleep(2)
 wait_specific(g);out=N/'checkpoints/sft-quality-val50-r64';assert not out.exists()
 args=[P,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(world),'-m','self_paging.train.sft','--model',M,'--tokenizer',M,'--data',str(D/'train.jsonl'),'--eval-data',str(D/'heldout.jsonl'),'--out',str(out),'--log',str(R/'sft.jsonl'),'--attn','flex_attention','--grad-ckpt','--lora-r','64','--lr','1e-4','--epochs','1','--max-tokens','32768','--seed','0','--eval-every','50','--save-every','50']
 run('sft-quality',args,g)
 logs=[json.loads(l) for l in (R/'sft.jsonl').read_text().splitlines()];assert sum('step' in r for r in logs)==steps and [r['completed_steps'] for r in logs if 'eval_loss' in r]==sorted(set(list(range(50,steps+1,50))+[steps]))
 for number in sorted(set(list(range(50,steps+1,50))+[steps])):
  ck=out/f'checkpoint-{number:06d}';assert (ck/'adapter_model.safetensors').exists();assert json.loads((ck/'checkpoint.json').read_text())['completed_steps']==number
 assert (out/'adapter_model.safetensors').exists() and (out/'adapter_config.json').exists()
 relay.wait(timeout=120);assert relay.returncode==0;CHILDREN.pop(relay.pid,None);f.close();verify_cloud()
 write(R/'sft-complete.json',dict(checkpoint=str(out),time=time.time(),steps=steps,gpus=g,cloud_verified=True))
 # Execute paired held-out protocol evaluation automatically, distributing
 # independent seeds to genuinely free cards without sharing a GPU.
 pending=list(range(100000,100050,5));active={};failed=[]
 while pending or active:
  stopped()
  for gpu,(proc,d,identity,f) in list(active.items()):
   if proc.poll() is not None:
    write(d/'exit.json',dict(returncode=proc.returncode,time=time.time(),identity=identity));f.close();del active[gpu]
    CHILDREN.pop(proc.pid,None)
    if proc.returncode:failed.append(str(d))
  if failed and not active:raise RuntimeError('Evaluation failures: '+str(failed))
  if pending and not failed:
   first=free();time.sleep(3)
   for gpu in sorted(first&free()):
    if not pending or gpu in active:continue
    s=pending.pop(0);d=R/f'eval-{s}';assert not d.exists();d.mkdir();f=(d/'stdout.log').open('w')
    cmd=[P,'-m','self_paging.eval.t1_eval','--model',str(out),'--tokenizer',M,'--methods','full','model','--seeds',str(s),str(s+5),'--out',str(d/'summary.jsonl'),'--records-out',str(N/f'sft-quality-val50-eval-{s}.jsonl')]
    proc=subprocess.Popen(cmd,cwd=C,env=os.environ|{'PYTHONPATH':str(C),'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    identity=ident(proc.pid);active[gpu]=(proc,d,identity,f);write(d/'launch.json',dict(identity=identity,gpu=gpu,cmd=cmd))
    CHILDREN[proc.pid]=(proc,identity)
  write(R/'health.json',dict(time=time.time(),stage='sft-evaluation',pending=pending,active={g:v[2] for g,v in active.items()}));time.sleep(10)
 ev=[json.loads(l) for p in R.glob('eval-*/summary.jsonl') for l in p.read_text().splitlines()]
 assert len(ev)==200 and all(r['over_budget']==0 for r in ev)
 write(R/'complete.json',dict(time=time.time(),status='SFT_AND_PAIRED_EVAL_COMPLETE_PENDING_SCIENTIFIC_REVIEW',episodes=200,
       next='Assess invalid actions, successful paging, reward and timing before GRPO; E2/BCP/LongMemEval remain separate gates'))
if __name__=='__main__':
 try:main()
 except BaseException:
  for proc,identity in CHILDREN.values():
   if proc.poll() is None and ident(proc.pid)==identity:os.killpg(proc.pid,signal.SIGTERM)
  for proc,identity in CHILDREN.values():
   try:proc.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
  write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
