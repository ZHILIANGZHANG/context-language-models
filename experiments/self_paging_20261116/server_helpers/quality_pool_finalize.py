"""Unique CPU continuation: select balanced candidates and audit; NEVER trains."""
import fcntl,json,os,pathlib,time,subprocess,traceback,hashlib
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-quality');P='/data/zhangcheng/self-paging-20261116/venv/bin/python';R=H/'quality-pool-finalize'
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 lock=open(H/'quality-pool-finalize.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not R.exists();R.mkdir();p=pathlib.Path('/proc')/str(os.getpid());write(R/'identity.json',dict(pid=os.getpid(),uid=os.getuid(),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve())))
 write(R/'helpers.json',{n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['quality_pool_finalize.py','select_quality_cases.py','diversity_audit.py']})
 while not (H/'quality-candidates/complete.json').exists():
  if any(p.exists() for p in [H/'STOP',R/'STOP',H/'quality-candidates/STOP',H/'quality-candidates/error.json']):raise RuntimeError('STOP or candidate failure')
  write(R/'health.json',dict(stage='WAITING_CANDIDATE_POOL',time=time.time()));time.sleep(20)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(C)+':'+str(H),'OMP_NUM_THREADS':'2'}
 for name,args in [('selection',['select_quality_cases.py','--output',str(H/'quality-balanced-selection')]),('audit',['diversity_audit.py','--limit','0','--manifest',str(H/'quality-balanced-selection/teacher-cases.jsonl'),'--output',str(H/'quality-balanced-audit')])]:
  assert not (H/'STOP').exists() and not (R/'STOP').exists()
  write(R/'health.json',dict(stage=name,time=time.time()))
  with (R/f'{name}.log').open('w') as f:rc=subprocess.call([P,str(H/args[0])]+args[1:],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
  write(R/f'{name}.exit.json',dict(returncode=rc,time=time.time()));assert rc==0
 d=json.loads((H/'quality-balanced-audit/summary.json').read_text());assert not d['failures'];assert d['records']==2640
 write(R/'complete.json',dict(status='BALANCED_CANDIDATES_AUDITED_NOT_TRAINING_ACCEPTED',time=time.time(),records=d['records']))
if __name__=='__main__':
 try:main()
 except BaseException:
  if R.exists():write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
  raise
