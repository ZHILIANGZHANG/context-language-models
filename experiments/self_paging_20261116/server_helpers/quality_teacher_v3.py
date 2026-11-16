"""Bounded GPU teacher trial across budgets and behavior modes; never starts training."""
import collections,hashlib,json,pathlib,torch,time
from self_paging.train.common import load_model,load_tokenizer
from self_paging.core.log import TokenLog
from self_paging.core.protocol import parse_tool_calls
from self_paging.agent.episode import EpisodeConfig,ScriptedPolicy,run_episode
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config,read_state
from self_paging.oracle.t1_oracle import T1Oracle
from quality_writer_v3 import QualityWriterV3 as QualityWriter,RejectCandidate
from quality_checks import inspect
from self_paging.data.build_t1_sft import t1_budget,passes
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'quality-review';N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/quality-teacher-v3');M='/nas/Public/zhangcheng/models/Qwen3-8B'
assert not N.exists();N.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(0);guard=torch.empty(1,device='cuda')
tok=load_tokenizer(M);teacher=load_model(M,'sdpa',device='cuda').eval();original=teacher.generate
# Keep the actual teacher prompts separate from token-exact student training records.
def capture(ids,*a,**kw):
 out=original(ids,*a,**kw)
 with (N/'private-prompts-audit-only.jsonl').open('a') as f:f.write(json.dumps(dict(prompt_ids=ids[0].tolist(),reply_ids=out[0,ids.shape[1]:].tolist(),training_use=False))+'\n')
 return out
teacher.generate=capture
cases=json.loads((R/'teacher-probe-cases.json').read_text());results=[];started=time.time()
for case in cases:
 assert not (H/'STOP').exists() and not (R/'STOP').exists()
 seed,ratio,mode=case['seed'],case['ratio'],case['mode'];env=WarehouseEnv(seed,T1Config());b=t1_budget(env,tok,ratio);oracle=T1Oracle(mode,seed);writer=QualityWriter(teacher,tok,oracle,audit_path=N/'prose-audit.jsonl');turns=[]
 def audited(st):
  text=writer(st);calls,_,bad=parse_tool_calls(text);turns.append(dict(turn=st.turn,text=text,calls=calls,expected_state=[sorted(x) for x in read_state(env.ep.pages[:env.cursor])]))
  return text
 rec=run_episode(env,TokenLog(b),ScriptedPolicy(tok,audited),tok,EpisodeConfig(budget=b,max_turns=80));rec['meta'].update(seed=seed,ratio=ratio,mode=mode,source='quality_teacher_trial')
 p=N/f'{seed}-{ratio:g}-{mode}.json';p.write_text(json.dumps(rec));(N/f'{seed}-{ratio:g}-{mode}.audit.json').write_text(json.dumps(turns,indent=2))
 row=dict(**case,tokens=len(rec['tokens']),reward=rec['reward'],meta=rec['meta'],audit=inspect(rec,tok,seed),strict_execution=passes(rec,1.) and len(rec['tokens'])<=32768,fallback_share=rec['meta'].get('teacher_fallbacks',0)/max(1,rec['meta'].get('teacher_generated_turns',0)),raw=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=time.time()-started)
 results.append(row);(R/'teacher-v3-progress.json').write_text(json.dumps(dict(completed=len(results),planned=len(cases),results=results),indent=2));print(json.dumps(row),flush=True)
(R/'teacher-v3-summary.json').write_text(json.dumps(dict(status='GENERATED_PENDING_PROSE_AND_DIVERSITY_REVIEW',results=results),indent=2))
