"""CPU final admission after GPU generation; never starts training or bypasses gaps."""
import fcntl,json,os,pathlib,subprocess,time,traceback
from run_quality_teacher_v3 import ident,write
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-quality');P='/data/zhangcheng/self-paging-20261116/venv/bin/python';N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data');R=H/'teacher-final-admission'
def main():
 lock=open(H/'teacher-final-admission.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not R.exists();R.mkdir();write(R/'identity.json',ident(os.getpid()))
 while not (H/'quality-teacher-production/complete.json').exists():
  if any(p.exists() for p in [H/'STOP',R/'STOP',H/'quality-review/STOP',H/'quality-production-wait-error.json',H/'quality-teacher-production/error.json']):raise RuntimeError('STOP or upstream failure')
  write(R/'health.json',dict(stage='WAITING_GPU_TEACHER_DATA',time=time.time()));time.sleep(30)
 env=os.environ|{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(C)+':'+str(H),'OMP_NUM_THREADS':'2'}
 with (R/'assembly.log').open('w') as f:rc=subprocess.call([P,str(H/'assemble_quality_data.py'),'--results',str(N/'quality-teacher-production'),str(N/'quality-teacher-scale'),'--out',str(N/'quality-accepted')],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
 write(R/'assembly.exit.json',dict(returncode=rc,time=time.time()))
 write(R/'complete.json',dict(status='DATA_ACCEPTED_PENDING_REAL_TRAINING_SMOKE' if rc==0 else 'COVERAGE_OR_QUALITY_GAPS_BLOCK_TRAINING',time=time.time(),training_started=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  if R.exists():write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
  raise
