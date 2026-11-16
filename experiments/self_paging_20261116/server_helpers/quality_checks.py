"""Read-only per-trajectory factual/execution checks, independent from oracle acceptance."""
import collections,re
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config,read_state
from self_paging.core.protocol import parse_tool_calls

def inspect(rec,tok,seed):
 env=WarehouseEnv(seed,T1Config());flags=collections.Counter();calls_count=collections.Counter();turns=[];claims=0
 for p in rec['pages']:
  if p['kind']!='assistant':continue
  ids=rec['tokens'][p['start']:p['start']+p['length']];text=tok.decode(ids);calls,_,bad=parse_tool_calls(text);flags['malformed_calls']+=bad
  full=read_state(env.ep.pages[:env.cursor]);prose=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S)
  for shelf,status in re.findall(r'S(\d+) (occupied|quarantined|empty)',prose):
   s=int(shelf);correct=(s in full[0]) if status=='occupied' else (s in full[1]) if status=='quarantined' else s not in full[0] and s not in full[1];claims+=1
   if not correct:flags['false_shelf_claim']+=1
  names=[c['name'] for c in calls];calls_count.update(names)
  turns.append(dict(start=p['start'],length=p['length'],plain_next=names==['next_events'],names=names))
  for c in calls:
   name=c['name'];args=c['arguments']
   if name=='note' and read_state([args['text']])!=full:flags['incomplete_state_note']+=1
   if name in ['next_events','store']:
    body,_,_=env.step(name,args)
    if body.startswith('ERROR'):flags['environment_error']+=1
   elif name not in ['note','kv_in','kv_out','kv_drop']:flags['unknown_call']+=1
 if env.score()!=rec['reward']:flags['reward_replay_mismatch']+=1
 if not env.done:flags['unfinished_episode']+=1
 n=len(rec['tokens'])
 if any(len(rec[k])!=n for k in ['page_of','seg_of','loss_mask','ctrl_mask','logprobs']):flags['array_length_mismatch']+=1
 if n>32768:flags['longer_than32k']+=1
 if any(x not in [0,1] for x in rec['loss_mask']):flags['invalid_loss_mask']+=1
 return dict(flags={k:v for k,v in flags.items() if v},calls=dict(calls_count),claims=claims,turns=turns,kinds=dict(collections.Counter(d.kind for d in env.ep.decisions)),tokens=n,targets=sum(rec['loss_mask']))
