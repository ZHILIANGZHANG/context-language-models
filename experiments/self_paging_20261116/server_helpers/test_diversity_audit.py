from diversity_audit import audit
from self_paging.agent.episode import EpisodeConfig,ScriptedPolicy,run_episode
from self_paging.core.log import TokenLog
from self_paging.data.build_t1_sft import t1_budget,passes
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config
from self_paging.oracle.t1_oracle import T1Oracle
from self_paging.train.common import load_tokenizer

def test_exact_transitions_and_tamper():
 tok=load_tokenizer('toy');witness=None
 for seed in range(4):
  env=WarehouseEnv(seed,T1Config());b=t1_budget(env,tok,.45)
  rec=run_episode(env,TokenLog(b),ScriptedPolicy(tok,T1Oracle('eager',seed)),tok,EpisodeConfig(budget=b,max_turns=80))
  if passes(rec,1.):
   a=audit(rec,tok,seed);assert not a['flags'],a
   if rec['meta']['kv_in'] and len(rec['snapshots'])>1:witness=(rec,seed);break
 assert witness,'No nontrivial valid test case'
 rec,seed=witness
 from self_paging.core.protocol import parse_tool_calls
 for p in rec['pages']:
  if p['kind']!='assistant':continue
  calls,_,_=parse_tool_calls(tok.decode(rec['tokens'][p['start']:p['start']+p['length']]))
  if any(c['name']=='kv_out' for c in calls):
   v=rec['seg_of'][p['start']+p['length']-1]+1;rec['snapshots'][v]=[];break
 assert 'snapshot_action_mismatch' in audit(rec,tok,seed)['flags']
