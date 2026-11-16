"""Check explicit lowest-shelf and storage assertions against the validated action/state."""
import re

def semantic_errors(text,calls,state):
 occ,qua=state;lowest=next(i for i in range(1,1000) if i not in occ and i not in qua);errors=[]
 # Limit to a local clause without another shelf reference, avoiding list/range ambiguity.
 claims=[]
 for m in re.finditer(r'\bS(\d+)\s+(?:(?:is|as|being|was|which|the|it|,|currently|now)\s+){0,6}lowest\b',text,re.I):claims.append(int(m.group(1)))
 for m in re.finditer(r'\blowest(?:[- ]numbered)?\s+(?:(?:free|empty|eligible|available|non-quarantined)\s+){0,4}shelf\s*(?:is|:)?\s*S(\d+)\b',text,re.I):claims.append(int(m.group(1)))
 for n in claims:
  if n!=lowest:errors.append('false_lowest_shelf_claim')
 expected=[int(c['arguments']['shelf']) for c in calls if c['name']=='store']
 for m in re.finditer(r'\b(?:store|storing|stored|place|placing)\b(?:(?![.\n]).){0,70}?\b(?:on|in|at|onto)\s+(?:shelf\s+)?S(\d+)\b',text,re.I):
  if not expected or int(m.group(1)) not in expected:errors.append('storage_prose_action_mismatch')
 return errors

def record_errors(rec,tok):
 from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config,read_state
 from self_paging.core.protocol import parse_tool_calls
 env=WarehouseEnv(rec['meta']['seed'],T1Config());errors=[]
 for p in rec['pages']:
  if p['kind']!='assistant':continue
  text=tok.decode(rec['tokens'][p['start']:p['start']+p['length']]);calls,_,_=parse_tool_calls(text);prose=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S)
  errors.extend(semantic_errors(prose,calls,read_state(env.ep.pages[:env.cursor])))
  for c in calls:
   if c['name'] in ['next_events','store']:env.step(c['name'],c['arguments'])
 return errors
