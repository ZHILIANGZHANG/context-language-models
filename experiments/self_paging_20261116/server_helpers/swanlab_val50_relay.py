"""CPU-only authenticated online relay of the repaired formal SFT log."""
import contextlib,fcntl,hashlib,json,math,os,pathlib,stat,time
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'quality-training-val50';A=R/'swanlab'
OWNER='zhangcheng525';PROJECT='context-language-models'
def write(name,obj):
 p=A/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def ident():
 p=pathlib.Path('/proc')/str(os.getpid())
 return dict(pid=os.getpid(),uid=p.stat().st_uid,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve()))
def main():
 os.umask(0o077)
 lock=open(R/'swanlab.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not A.exists(),'Previous relay evidence; do not duplicate run'
 A.mkdir();write('identity.json',ident())
 for k in list(os.environ):
  if k.startswith('SWANLAB_'):os.environ.pop(k)
 import swanlab
 secret=H/'.swanlab-key';st=secret.stat();assert st.st_uid==os.getuid() and stat.S_IMODE(st.st_mode)==0o600
 with open(os.devnull,'w') as null,contextlib.redirect_stdout(null),contextlib.redirect_stderr(null):
  api=swanlab.Api(api_key=secret.read_text().strip())
  assert api.user().username==OWNER
  assert api.project(f'{OWNER}/{PROJECT}').visibility=='PRIVATE'
  assert swanlab.login(api_key=secret.read_text().strip(),relogin=True,save=False,timeout=12)
 write('auth-ready.json',dict(private=True,owner=OWNER,project=PROJECT,time=time.time()))
 while not (R/'sft-quality/launch.json').exists():
  assert not any(p.exists() for p in [H/'STOP',R/'STOP',R/'error.json'])
  write('health.json',dict(time=time.time(),phase='waiting_formal_launch'));time.sleep(3)
 launch=json.loads((R/'sft-quality/launch.json').read_text())
 config=json.loads((R/'training-plan.json').read_text())
 config.update(world_size=len(launch['gpus']),gpus=launch['gpus'])
 run=swanlab.init(workspace=OWNER,project=PROJECT,public=False,mode='online',
  name='Qwen3-8B-quality-SFT-val50-restart',group='self-paging-20261116',config=config,log_dir=str(A/'swanlog'))
 write('run.json',dict(id=run.id,url=run.url,owner=OWNER,project=PROJECT,config=config,time=time.time()))
 source=R/'sft.jsonl';offset=count=0;step=0
 try:
  while True:
   if source.exists():
    assert source.stat().st_size>=offset,'Training log replaced'
    with source.open('rb') as f:
     f.seek(offset)
     for line in f:
      if not line.endswith(b'\n'):break
      obj=json.loads(line);step=obj.get('step',obj.get('eval_step',step))
      values={}
      for key,dest in [('loss','train/loss'),('lr','train/lr'),('epoch','train/epoch'),('sec','train/step_seconds'),('tokens','train/supervised_tokens'),('eval_loss','validation/loss'),('eval_sec','validation/seconds'),('completed_steps','progress/completed_steps')]:
       if key in obj:values[dest]=float(obj[key])
      if 'step' in obj:values['train/optimizer_step']=step
      assert all(math.isfinite(v) for v in values.values())
      if values:run.log(values,step=step);count+=1
      offset+=len(line)
    write('health.json',dict(time=time.time(),phase='streaming',records=count,last_step=step,offset=offset))
   exit_file=R/'sft-quality/exit.json'
   if exit_file.exists() and (not source.exists() or source.stat().st_size==offset):
    code=json.loads(exit_file.read_text())['returncode'];run.finish(state='success' if code==0 else 'crashed',error=None if code==0 else 'Training failed; inspect preserved local evidence')
    write('exit.json',dict(time=time.time(),train_returncode=code,records=count,last_step=step));return
   if (H/'STOP').exists() or (R/'STOP').exists() or (R/'error.json').exists():
    run.finish(state='crashed',error='User STOP');write('exit.json',dict(time=time.time(),status='USER_STOP'));return
   time.sleep(5)
 except BaseException as exc:
  write('error.json',dict(time=time.time(),type=type(exc).__name__))
  run.finish(state='crashed',error='Metrics relay failure');raise
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if A.exists():write('error.json',dict(time=time.time(),type=type(exc).__name__))
  print('RELAY_FAILED',type(exc).__name__,flush=True);raise SystemExit(1) from None
