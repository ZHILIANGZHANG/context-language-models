"""GPU prose only for substantive actions; deterministic facts/actions remain audited."""
import re,torch
from quality_semantics import semantic_errors
from self_paging.core.protocol import IM_END,IM_START,NO_THINK,parse_tool_calls,render_tool_call
from self_paging.oracle.hinted_teacher import HintedTeacher,LEAK
from self_paging.envs.t1_warehouse import read_state
class RejectCandidate(ValueError):pass

def factual_errors(text,state):
 occ,qua=state;errors=[]
 # Explicit single-shelf and contiguous-range claims; unsupported prose is still sampled for review.
 pattern=r'\bS(\d+)\s+(?:(?:is|are|currently|now)\s+)*(occupied|quarantined|empty)\b'
 for n,status in re.findall(pattern,text,re.I):
  s=int(n);status=status.lower();ok=s in occ if status=='occupied' else s in qua if status=='quarantined' else s not in occ
  if not ok:errors.append(f'S{s}:{status}')
 for lo,hi,status in re.findall(r'\bS(\d+)\s*(?:through|to|[-–])\s*S?(\d+)\s+(?:(?:is|are|all|currently)\s+)*(occupied|quarantined|empty)',text,re.I):
  for s in range(int(lo),int(hi)+1):
   ok=s in occ if status.lower()=='occupied' else s in qua if status.lower()=='quarantined' else s not in occ
   if not ok:errors.append(f'S{s}:{status}')
 return errors

class QualityWriterV4(HintedTeacher):
 def __init__(self,model,tok,oracle,audit_path=None):
  super().__init__(model,tok,oracle,max_new_tokens=96);self.audit_path=audit_path
 @torch.no_grad()
 def __call__(self,st):
  import json
  plan=self.oracle(st);calls,spans,bad=parse_tool_calls(plan)
  if bad or not calls:raise RejectCandidate('Invalid oracle action')
  facts=re.sub(r'<tool_call>.*?</tool_call>','',plan,flags=re.S).strip();state=read_state(st.env.ep.pages[:st.env.cursor])
  if factual_errors(facts,state):raise RejectCandidate('False oracle factual claim')
  for c in calls:
   if c['name']=='note' and read_state([c['arguments']['text']])!=state:raise RejectCandidate('Incomplete state note')
  if [c['name'] for c in calls]==['next_events']:
   st.meta['teacher_skipped_plain_next']=st.meta.get('teacher_skipped_plain_next',0)+1
   return render_tool_call('next_events',{})
  actions='\n'.join(render_tool_call(c['name'],c['arguments']) for c in calls)
  style=['Explain the immediate reason in one short sentence.','Use two brief sentences: observation, then action.','State the relevant observation concisely, without a generic introduction.','Explain the action directly without restating the whole workflow.'][st.turn%4]
  required=[]
  names={c['name'] for c in calls}
  if 'note' in names:required.append('FIRST write a summary note preserving the current shelf state')
  if 'kv_out' in names:required.append('page OUT the specified pages to free memory, explaining the supplied reason')
  if 'kv_in' in names:required.append('THEN restore the required host pages to read the decision evidence')
  if 'store' in names:required.append('store the given item on the stated lowest free eligible shelf')
  coverage='; '.join(required)
  if len(required)>1:style='Use two short sentences and explicitly mention EVERY required action in order; omit long page-number lists if needed. Never omit page-out when page-in is also required, or omit the note before page-out.'
  hint=(f'{IM_START}system\nYou are a concise copy editor, not a warehouse agent. Rewrite only the supplied verified plan into a short first-person intention. Preserve the action type and reason. Do not execute actions, predict events, infer extra facts, quote instructions, emit JSON or tool calls, or claim actions already happened. Use one or two sentences.\nExample plan: Memory is tight; paging out 12 (telemetry only). Rewrite: I will page out telemetry page 12 to free memory.\nExample plan: Deciding needs pages 4,8 which are out of memory; paging them back in. Rewrite: I will restore pages 4 and 8 to read the evidence needed for this decision.\nExample compound plan: write a state note, then page out old records. Rewrite: I will preserve the shelf state in a summary note, then page out the redundant records to free memory.\nExample compound plan: page out old turns to make room and restore evidence pages. Rewrite: I will evict older turns to make room, then restore the evidence pages needed for the shelf decision.{IM_END}\n'
        f'{IM_START}user\nVerified plan: {facts}\nMandatory actions: {coverage}\n{style} Return only the rewritten plan.{IM_END}\n{IM_START}assistant\n{NO_THINK}')
  ids=self.tok.encode(hint);device=next(self.model.parameters()).device
  out=self.model.generate(torch.tensor([ids],device=device),max_new_tokens=self.max_new,do_sample=True,temperature=.4,top_p=.9,top_k=20,eos_token_id=self.tok.im_end,pad_token_id=self.tok.im_end)
  reply=out[0,len(ids):].tolist();text=self.tok.decode(reply).split(IM_END)[0].strip();issues=[]
  if not reply or reply[-1]!=self.tok.im_end:issues.append('generation_truncated')
  if not text or LEAK.search(text) or any(tag in text for tag in ['<tool_call>','<|im_start|>','<think>']):issues.append('leak_or_format')
  issues+=factual_errors(text,state)
  issues+=semantic_errors(text,calls,state)
  names={c['name'] for c in calls};low=text.lower()
  if 'kv_out' in names and not re.search(r'page out|paging out|evict|free.*memory|remov',low):issues.append('missing_eviction_reason')
  if 'kv_in' in names and not re.search(r'restor|page.*back|bring.*back|retriev|reload|page in|paging in',low):issues.append('missing_restore_reason')
  if 'note' in names and not re.search(r'note|summari|summary',low):issues.append('missing_note_reason')
  if 'store' not in names and re.search(r'\b(stor(?:e|ed|ing)|placed?)\b',low):issues.append('wrong_action_store')
  if 'store' in names and not re.search(r'\b(store|storing|place)\b',low):issues.append('missing_store_action')
  if set(re.findall(r'\b\d+\b',text))-set(re.findall(r'\b\d+\b',facts+actions)):issues.append('new_numeric_claim')
  allowed=set(re.findall(r'\b(?:S\d+|I-\d+)\b',facts+actions));mentioned=set(re.findall(r'\b(?:S\d+|I-\d+)\b',text))
  if mentioned-allowed:issues.append('new_entity_not_in_fact_card')
  st.meta['teacher_generated_turns']=st.meta.get('teacher_generated_turns',0)+1
  if issues:
   st.meta['teacher_fallbacks']=st.meta.get('teacher_fallbacks',0)+1;text=facts
  if self.audit_path:
   with open(self.audit_path,'a') as f:f.write(json.dumps(dict(seed=st.env.ep.seed,turn=st.turn,actions=calls,facts=facts,reply=self.tok.decode(reply),used=text,fallback_reasons=issues,training_use=False))+'\n')
  return text+'\n'+actions
