"""Balanced CPU selection; immutable manifest, no GPU or training side effects."""
import collections,hashlib,json,pathlib,argparse

def select(rows,total,min_seeds):
 assert total%15==0
 # 20% unbudgeted; 40% each out-only and restored, evenly across finite budgets.
 quotas={(1.,'none'):total//5}
 quotas.update({(b,c):total*2//15 for b in [.6,.45,.35] for c in ['out','both']})
 bins=collections.defaultdict(list)
 for r in rows:
  if not r['accepted']:continue
  cs=r['audit']['calls'];kind='both' if cs.get('kv_in') else 'out' if cs.get('kv_out') else 'none'
  k=(r['ratio'],kind)
  if k in quotas:bins[k].append(r)
 selected=[];seed_counts=collections.Counter();mode_counts=collections.Counter();deficits={}
 for k in sorted(quotas,key=lambda k:len(bins[k])/quotas[k]):
  candidates=bins[k][:]
  for _ in range(quotas[k]):
   viable=[r for r in candidates if seed_counts[r['seed']]<5]
   if not viable:break
   # Prefer new seeds and spread modes where nature provides them; never duplicate a trajectory.
   r=min(viable,key=lambda r:(seed_counts[r['seed']],mode_counts[(k,r['mode'])],hashlib.sha256(f"{r['seed']}/{r['ratio']}/{r['mode']}".encode()).hexdigest()))
   candidates.remove(r);selected.append(r);seed_counts[r['seed']]+=1;mode_counts[(k,r['mode'])]+=1
  actual=sum((r['ratio'], 'both' if r['audit']['calls'].get('kv_in') else 'out' if r['audit']['calls'].get('kv_out') else 'none')==k for r in selected)
  if actual!=quotas[k]:deficits[str(k)]=quotas[k]-actual
 if len(seed_counts)<min_seeds:deficits['distinct_seeds']=min_seeds-len(seed_counts)
 return selected,dict(quotas={str(k):v for k,v in quotas.items()},selected=len(selected),distinct_seeds=len(seed_counts),max_variants=max(seed_counts.values(),default=0),modes={str(k):v for k,v in mode_counts.items()},deficits=deficits)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/home/zhangcheng/self-paging-20261116');p.add_argument('--output',required=True);a=p.parse_args();h=pathlib.Path(a.root);out=pathlib.Path(a.output);assert not out.exists()
 assert (h/'quality-candidates/complete.json').exists(),'Candidate pool not complete; no selection released'
 rows=[json.loads(x) for x in (h/'quality-candidates/manifest.jsonl').read_text().splitlines()];allsel=[];summaries={}
 for split,total,nseeds in [('train',2400,600),('heldout',240,80)]:
  chosen,s=select([r for r in rows if r['split']==split],total,nseeds);summaries[split]=s;allsel+=chosen
 assert not ({r['seed'] for r in allsel if r['split']=='train'}&{r['seed'] for r in allsel if r['split']=='heldout'})
 out.mkdir();(out/'summary.json').write_text(json.dumps(summaries,indent=2))
 if any(s['deficits'] for s in summaries.values()):raise RuntimeError('Explicit coverage deficit; do not start training or duplicate examples')
 for r in allsel:assert hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()==r['sha256']
 (out/'teacher-cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in allsel))
 (out/'READY_FOR_GPU_GENERATION_ONLY.json').write_text(json.dumps(dict(status='candidate_selection_only',count=len(allsel),post_teacher_reaudit_required=True)))
if __name__=='__main__':main()
