"""Single checkpointed CPU candidate builder. No GPU allocation or training."""
import collections,fcntl,hashlib,json,os,pathlib,time,traceback
from quality_checks import inspect
from self_paging.agent.episode import EpisodeConfig,ScriptedPolicy,run_episode
from self_paging.core.log import TokenLog
from self_paging.data.build_t1_sft import t1_budget,passes
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config
from self_paging.oracle.t1_oracle import T1Oracle
from self_paging.train.common import load_tokenizer
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'quality-candidates';N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/quality-candidates');C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-quality')
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def main():
 lock=open(H/'quality-candidates.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not R.exists();R.mkdir();assert not N.exists();N.mkdir(parents=True)
 proc=pathlib.Path('/proc')/str(os.getpid());write(R/'identity.json',dict(pid=os.getpid(),uid=os.getuid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str(proc.joinpath('cwd').resolve())))
 manifest=json.loads((H/'quality-source-manifest.json').read_text());assert all(hashlib.sha256((C/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
 tok=load_tokenizer('/nas/Public/zhangcheng/models/Qwen3-8B');stats=collections.Counter();count=0;started=time.time()
 for split,seeds in [('train',range(2000,3000)),('heldout',range(91000,91100))]:
  out=N/split;out.mkdir()
  for seed in seeds:
   for ratio in [1.,.6,.45,.35]:
    for mode in (['paging'] if ratio==1 else ['paging','notes','eager']):
     if any(p.exists() for p in [H/'STOP',R/'STOP',H/'quality-review/STOP']):raise RuntimeError('User STOP')
     env=WarehouseEnv(seed,T1Config());b=t1_budget(env,tok,ratio);rec=run_episode(env,TokenLog(b),ScriptedPolicy(tok,T1Oracle(mode,seed)),tok,EpisodeConfig(budget=b,max_turns=80));rec['meta'].update(seed=seed,ratio=ratio,mode=mode,source='quality_checked_template_candidate',split=split)
     audit=inspect(rec,tok,seed);upstream=passes(rec,1.);accept=upstream and not audit['flags'];name=f'{seed}-{ratio:g}-{mode}.json';path=out/name;path.write_text(json.dumps(rec));audit.pop('turns')
     row=dict(split=split,seed=seed,ratio=ratio,mode=mode,upstream_pass=upstream,accepted=accept,audit=audit,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
     with (R/'manifest.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     stats[f'{split}/{ratio}/{mode}/candidates']+=1;stats[f'{split}/{ratio}/{mode}/accepted']+=int(accept);count+=1
   write(R/'health.json',dict(time=time.time(),stage='building_and_fact_checking',split=split,seed=seed,candidates=count,planned=11000,seconds=time.time()-started,stats=dict(stats)))
   print(json.dumps(dict(seed=seed,split=split,candidates=count,seconds=time.time()-started)),flush=True)
 write(R/'complete.json',dict(time=time.time(),status='CANDIDATES_COMPLETE_PENDING_BALANCING_AND_GPU_TEACHER',candidates=count,stats=dict(stats)))
if __name__=='__main__':
 try:main()
 except BaseException:
  if R.exists():write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
  raise
