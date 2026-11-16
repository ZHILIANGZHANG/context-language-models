"""Final release gate: balanced actual GPU trajectories, NOT template candidate acceptance."""
import argparse,collections,copy,hashlib,json,pathlib
from select_quality_cases import select
from quality_checks import inspect
from quality_semantics import record_errors
from diversity_audit import audit
from self_paging.data.build_t1_sft import passes
from self_paging.train.common import load_tokenizer

def thin_plain_next(rec,turns):
 out=copy.deepcopy(rec);removed=0;eligible=0;kept=0
 for t in turns:
  if not t['plain_next']:continue
  eligible+=1
  key=f"next-loss-v1/{rec['meta']['seed']}/{t['start']}";retain=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%4==0
  if retain:kept+=1;continue
  for i in range(t['start'],t['start']+t['length']):removed+=out['loss_mask'][i];out['loss_mask'][i]=0
 out['meta']['plain_next_loss_sampling']=dict(version=1,retention_probability=.25,eligible_turns=eligible,retained_turns=kept,removed_loss_tokens=removed,context_unchanged=True)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();dest=pathlib.Path(a.out);assert not dest.exists()
 rows=[];seen={};rejected=collections.Counter()
 for root in a.results:
  for f in pathlib.Path(root).glob('*/result.json'):
   r=json.loads(f.read_text())
   if not r['accepted']:rejected['producer_rejected']+=1;continue
   raw=pathlib.Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];rec=json.loads(raw)
   assert rec['meta']['source'] in ['gpu_fact_card_rewrite_v3','gpu_fact_card_rewrite_v4'] and rec['meta'].get('teacher_fallbacks',0)==0
   key=(r['seed'],r['ratio'],r['mode'])
   if key in seen:
    assert seen[key]==r['sha256'],'Conflicting duplicate trajectory';continue
   seen[key]=r['sha256'];r['split']=rec['meta']['split'];r['accepted']=True;rows.append(r)
 selected=[];summaries={}
 for split,total,nseeds in [('train',2400,600),('heldout',240,80)]:
  rs,s=select([r for r in rows if r['split']==split],total,nseeds);summaries[split]=s;selected+=rs
 assert all(not s['deficits'] for s in summaries.values()),json.dumps(summaries)
 trainseeds={r['seed'] for r in selected if r['split']=='train'};valseeds={r['seed'] for r in selected if r['split']=='heldout'}
 assert not trainseeds&valseeds and all(2000<=s<3000 for s in trainseeds) and all(91000<=s<91100 for s in valseeds)
 budget_pairs=collections.defaultdict(set)
 for r in selected:
  if r['split']=='train':budget_pairs[r['seed']].add(r['ratio'])
 assert sum(len(x)>=2 for x in budget_pairs.values())>=300
 assert sum(1. in x and len(x)>=2 for x in budget_pairs.values())>=100
 tok=load_tokenizer('/nas/Public/zhangcheng/models/Qwen3-8B');staged=[];coverage=collections.Counter();event_kinds=collections.Counter();source_rows=[];targets=collections.Counter()
 for r in selected:
  rec=json.loads(pathlib.Path(r['path']).read_text());check=inspect(rec,tok,r['seed']);mem=audit(rec,tok,r['seed']);assert passes(rec,1.) and not check['flags'] and not mem['flags'] and not record_errors(rec,tok);assert rec['meta'].get('teacher_fallbacks',0)==0
  if r['split']=='train':
   for name in ['in_only','out_only','both']:coverage[(r['ratio'],name)]+=int(mem['counts'].get('turn/'+name,0)>0)
   for k,n in check['kinds'].items():event_kinds[(r['ratio'],k)]+=n
  transformed=thin_plain_next(rec,check['turns']) if r['split']=='train' else rec
  # All inputs, states, positions, tool labels remain identical; only repetitive-next target weight changes.
  for k in rec:
   if k not in ['loss_mask','meta']:assert transformed[k]==rec[k]
  assert sum(transformed['loss_mask'])>0;staged.append((r['split'],transformed));source_rows.append({k:r[k] for k in ['seed','ratio','mode','path','sha256','split']});targets[r['split']+'/before']+=sum(rec['loss_mask']);targets[r['split']+'/after']+=sum(transformed['loss_mask'])
 for ratio in [.6,.45,.35]:
  assert coverage[(ratio,'in_only')]>=30 and coverage[(ratio,'both')]>=30,dict(coverage)
  for kind in ['explicit','implicit','delayed']:assert event_kinds[(ratio,kind)]>0
 dest.mkdir();hashes={}
 for split in ['train','heldout']:
  f=dest/f'{split}.jsonl'
  with f.open('w') as out:
   for group,rec in staged:
    if group==split:out.write(json.dumps(rec)+'\n')
  hashes[split]=hashlib.sha256(f.read_bytes()).hexdigest()
 (dest/'provenance.json').write_text(json.dumps(source_rows,indent=2))
 (dest/'data-accepted.json').write_text(json.dumps(dict(status='DATA_ACCEPTED_TRAINING_SMOKE_STILL_REQUIRED',counts=summaries,sha256=hashes,coverage={str(k):v for k,v in coverage.items()},event_kinds={str(k):v for k,v in event_kinds.items()},loss_targets=targets,zero_fallback=True,multi_budget_seeds=sum(len(x)>=2 for x in budget_pairs.values()),full_vs_budget_pairs=sum(1. in x and len(x)>=2 for x in budget_pairs.values()),rejections=rejected),indent=2))
if __name__=='__main__':main()
