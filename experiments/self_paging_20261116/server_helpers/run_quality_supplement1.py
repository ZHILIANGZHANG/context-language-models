"""One bounded multi-GPU quality batch, no automatic training."""
import fcntl,json,os,pathlib,subprocess,time,signal,traceback,hashlib
from select_quality_cases import select
from run_quality_teacher_v3 import ident,free,write
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-quality');P='/data/zhangcheng/self-paging-20261116/venv/bin/python';R=H/'quality-teacher-supplement1';N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/quality-teacher-supplement1')
def main():
 waitlock=open(H/'quality-supplement1-wait.lock','a');fcntl.flock(waitlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(H/'quality-supplement1-wait-identity.json',ident(os.getpid()))
 while not ((H/'quality-pool-finalize/complete.json').exists() and (H/'quality-teacher-v4/exit.json').exists()):
  if any(p.exists() for p in [H/'STOP',H/'quality-review/STOP',H/'quality-pool-finalize/error.json',H/'quality-teacher-v4/error.json']):raise RuntimeError('Upstream quality stage failed or STOP')
  write(H/'quality-supplement1-wait-health.json',dict(stage='WAITING_BALANCED_CANDIDATES_AND_V4_PROBE',time=time.time()));time.sleep(15)
 lock=open(H/'campaign.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not any(p.exists() for p in [H/'STOP',H/'quality-review/STOP']);assert not R.exists() and not N.exists()
 d=json.loads((H/'quality-teacher-v4/exit.json').read_text());assert d['returncode']==0 and not (pathlib.Path('/proc')/str(d['identity']['pid'])).exists()
 d=json.loads((H/'quality-review/teacher-v4-summary.json').read_text());assert len(d['results'])==8 and all(x['strict_execution'] and not x['audit']['flags'] for x in d['results']); assert sum(x['fallback_share']==0 for x in d['results'])>=7, 'V4 prose gate not met'
 for k,v in json.loads((H/'quality-source-manifest.json').read_text()).items():assert hashlib.sha256((C/k).read_bytes()).hexdigest()==v
 cases=[json.loads(x) for x in (H/'quality-supplement1-selection/cases.jsonl').read_text().splitlines()];summary=json.loads((H/'quality-supplement1-selection/summary.json').read_text());assert len(cases)==49
 g=[]
 while not g:
  assert not (H/'STOP').exists() and not (H/'quality-review/STOP').exists()
  a=free();time.sleep(3);g=sorted(a&free())
  if not g:time.sleep(20)
 R.mkdir();N.mkdir();write(R/'identity.json',ident(os.getpid()));write(R/'selection.json',summary);(R/'cases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in cases));write(R/'helper-sha.json',{n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['quality_writer_v4.py','quality_semantics.py','quality_supplement_worker.py','run_quality_supplement1.py','quality_checks.py','diversity_audit.py']})
 children=[]
 try:
  for rank,gpu in enumerate(g):
   f=(R/f'rank{rank}.log').open('w');env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(C)+':'+str(H),'OMP_NUM_THREADS':'4'}
   p=subprocess.Popen([P,str(H/'quality_supplement_worker.py'),'--rank',str(rank),'--world',str(len(g))],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);i=ident(p.pid);children.append((p,i,f));write(R/f'rank{rank}.launch.json',dict(identity=i,gpu=gpu,time=time.time()))
  while any(p.poll() is None for p,_,_ in children):
   if any(x.exists() for x in [H/'STOP',R/'STOP',H/'quality-review/STOP']):raise RuntimeError('STOP')
   if any(p.poll() not in [None,0] for p,_,_ in children):raise RuntimeError('Worker failed, no retry')
   counts=[sum(1 for _ in p.open()) for p in R.glob('rank*.jsonl')];write(R/'health.json',dict(time=time.time(),stage='GPU_TEACHER_PRODUCTION_CANDIDATES',completed=sum(counts),planned=len(cases),gpus=g));time.sleep(5)
  allrows=[]
  for rank,(p,i,f) in enumerate(children):write(R/f'rank{rank}.exit.json',dict(returncode=p.returncode,identity=i,time=time.time()));f.close();allrows.extend(json.loads(x) for x in (R/f'rank{rank}.jsonl').read_text().splitlines())
  assert len(allrows)==len(cases) and len({x['index'] for x in allrows})==len(cases)
  write(R/'complete.json',dict(status='TEACHER_GENERATED_FINAL_BALANCE_AND_TRAINING_GATE_PENDING',time=time.time(),generated=len(cases),accepted_zero_fallback=sum(r['accepted'] for r in allrows),training_started=False))
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  admission=H/'teacher-final-admission-attempt2';admission.mkdir()
  out=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/quality-accepted');assert not out.exists()
  command=[P,str(H/'assemble_quality_data.py'),'--results',str(N),str(N.parent/'quality-teacher-production'),str(N.parent/'quality-teacher-scale'),'--out',str(out)]
  assert not any(p.exists() for p in [H/'STOP',H/'quality-review/STOP',R/'STOP'])
  write(admission/'identity.json',ident(os.getpid()))
  with (admission/'assembly.log').open('w') as log:
   result=subprocess.run(command,cwd=C,env=os.environ|{'PYTHONPATH':str(C)+':'+str(H)},stdout=log,stderr=subprocess.STDOUT)
  write(admission/'assembly.exit.json',dict(returncode=result.returncode,time=time.time()))
  write(admission/'complete.json',dict(status='DATA_ADMITTED' if result.returncode==0 else 'COVERAGE_OR_QUALITY_GAPS_BLOCK_TRAINING',time=time.time()))
 except BaseException:
  for p,i,f in children:
   if p.poll() is None and ident(p.pid)==i:os.killpg(p.pid,signal.SIGTERM)
  for p,i,f in children:p.wait();f.close()
  raise
if __name__=='__main__':
 try:main()
 except BaseException:
  write((R/'error.json') if R.exists() else H/'quality-supplement1-wait-error.json',dict(time=time.time(),traceback=traceback.format_exc()))
  raise
