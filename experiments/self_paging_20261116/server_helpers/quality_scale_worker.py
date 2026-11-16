"""Bounded production-quality sample, immutable attempts and zero-fallback acceptance."""
import collections,hashlib,json,os,pathlib,time,traceback,torch,argparse
from quality_writer_v3 import QualityWriterV3,RejectCandidate
from quality_checks import inspect
from diversity_audit import audit
from self_paging.train.common import load_model,load_tokenizer
from self_paging.agent.episode import EpisodeConfig,ScriptedPolicy,run_episode
from self_paging.core.log import TokenLog
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config
from self_paging.oracle.t1_oracle import T1Oracle
from self_paging.data.build_t1_sft import t1_budget,passes
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'quality-teacher-scale';N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/quality-teacher-scale');M='/nas/Public/zhangcheng/models/Qwen3-8B'
p=argparse.ArgumentParser();p.add_argument('--rank',type=int);p.add_argument('--world',type=int);a=p.parse_args();torch.set_num_threads(4);tok=load_tokenizer(M);guard=torch.empty(1,device='cuda');model=load_model(M,'sdpa',device='cuda').eval();cases=[json.loads(l) for l in (R/'cases.jsonl').read_text().splitlines()];started=time.time()
for index,case in enumerate(cases):
 if index%a.world!=a.rank:continue
 assert not any(x.exists() for x in [H/'STOP',R/'STOP',H/'quality-review/STOP'])
 seed,ratio,mode=case['seed'],case['ratio'],case['mode'];torch.manual_seed(seed);key=f'{seed}-{ratio:g}-{mode}';out=N/key;out.mkdir();result=dict(index=index,seed=seed,ratio=ratio,mode=mode,source_candidate=case['path'],source_candidate_sha256=case['sha256'],accepted=False)
 assert hashlib.sha256(pathlib.Path(case['path']).read_bytes()).hexdigest()==case['sha256']
 try:
  env=WarehouseEnv(seed,T1Config());b=t1_budget(env,tok,ratio);writer=QualityWriterV3(model,tok,T1Oracle(mode,seed),audit_path=out/'prose-audit-only.jsonl');rec=run_episode(env,TokenLog(b),ScriptedPolicy(tok,writer),tok,EpisodeConfig(budget=b,max_turns=80));rec['meta'].update(seed=seed,ratio=ratio,mode=mode,split=case['split'],source='gpu_fact_card_rewrite_v3')
  path=out/'trajectory.json';path.write_text(json.dumps(rec));check=inspect(rec,tok,seed);memory=audit(rec,tok,seed);check.pop('turns');result.update(accepted=passes(rec,1.) and not check['flags'] and not memory['flags'] and rec['meta'].get('teacher_fallbacks',0)==0,meta=rec['meta'],audit=check,memory=memory,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.time()-started)
 except RejectCandidate as e:result.update(rejection=str(e),seconds=time.time()-started)
 (out/'result.json').write_text(json.dumps(result,indent=2))
 with (R/f'rank{a.rank}.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print(json.dumps(dict(index=index,seed=seed,accepted=result['accepted'],seconds=result['seconds'])),flush=True)
