"""Offline action coverage plus exact resident transitions; never changes source examples."""
import collections,json,pathlib,hashlib,argparse
from self_paging.core.protocol import parse_tool_calls,parse_pages
from self_paging.envs.t1_warehouse import WarehouseEnv,T1Config,rule_read
from self_paging.train.common import load_tokenizer
H=pathlib.Path('/home/zhangcheng/self-paging-20261116')
def audit(rec,tok,seed):
 env=WarehouseEnv(seed,T1Config());pages={p['pid']:p for p in rec['pages']};snaps=[set(s) for s in rec['snapshots']];counts=collections.Counter();flags=[];details=[];outs={};notes={};envtexts={}
 def visible(res):return [envtexts.get(p,notes.get(p,'')) for p in sorted(res,key=lambda p:pages[p]['start']) if p in envtexts or p in notes]
 ep_by_pid={int(pid):int(e) for e,pid in rec['env_page'].items()}
 for p in rec['pages']:
  if p['pid'] in ep_by_pid:envtexts[p['pid']]=env.ep.pages[ep_by_pid[p['pid']]]
  if p['kind']=='note':notes[p['pid']]=tok.decode(rec['tokens'][p['start']:p['start']+p['length']])
  if p['kind']!='assistant':continue
  calls,_,bad=parse_tool_calls(tok.decode(rec['tokens'][p['start']:p['start']+p['length']]))
  kv=[c for c in calls if c['name'] in ['kv_out','kv_in','kv_drop']];names={c['name'] for c in kv};kind='both' if {'kv_in','kv_out'}<=names else 'in_only' if 'kv_in' in names else 'out_only' if 'kv_out' in names else 'none';counts['turn/'+kind]+=1
  end=p['start']+p['length'];v=rec['seg_of'][end-1];existing={k for k,q in pages.items() if q['start']<end};res=snaps[v]&existing
  for c in kv:
   name=c['name'];ids=parse_pages(c['arguments']['pages']);counts['action/'+name]+=1;before=set(res);reading_before=rule_read(visible(res))
   if not ids or len(ids)!=len(set(ids)):flags.append('empty_or_duplicate_pages')
   if any(k not in existing for k in ids):flags.append('future_or_unknown_page');continue
   if name=='kv_in':
    if any(k in res or k not in outs for k in ids):flags.append('page_in_not_previous_host')
    for k in ids:counts['restored_page_kind/'+pages[k]['kind']]+=1
    res|=set(ids)
   else:
    if any(k not in res or pages[k]['pinned'] or k==p['pid'] for k in ids):flags.append('illegal_page_out')
    for k in ids:outs[k]=end;counts['evicted_page_kind/'+pages[k]['kind']]+=1
    res-=set(ids)
   v+=1
   if v>=len(snaps) or snaps[v]&existing!=res:flags.append('snapshot_action_mismatch')
   if rec['budget'] is not None and sum(pages[k]['length'] for k in res)>rec['budget']:flags.append('action_budget_exceeded')
   if name=='kv_in':
    after=rule_read(visible(res));truth=rule_read(env.ep.pages[:env.cursor]);counts['restore/changes_reference_read']+=int(reading_before!=after);counts['restore/repairs_current_reference_answer']+=int(reading_before!=truth and after==truth)
    details.append(dict(turn=p['pid'],pages=ids,before=reading_before,after=after,truth=truth,pending=env.pending is not None))
  for c in calls:
   if c['name'] in ['next_events','store']:env.step(c['name'],c['arguments'])
 allin=counts['action/kv_in'];allout=counts['action/kv_out'];counts['trajectory/'+('both' if allin and allout else 'in_only' if allin else 'out_only' if allout else 'none')]+=1
 return dict(counts=dict(counts),flags=flags,restores=details)
def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=400);p.add_argument('--output',required=True);p.add_argument('--manifest');args=p.parse_args();dest=pathlib.Path(args.output);assert not dest.exists();dest.mkdir()
 lines=(pathlib.Path(args.manifest) if args.manifest else H/'quality-candidates/manifest.jsonl').read_text().splitlines();rows=[json.loads(l) for l in lines if l.strip()];rows=[r for r in rows if r['accepted']];groups=collections.defaultdict(list)
 for r in rows:groups[(r['split'],r['ratio'],r['mode'],bool(r['audit']['calls'].get('kv_in'))) ].append(r)
 for g in groups.values():g.sort(key=lambda r:hashlib.sha256(str(r['seed']).encode()).hexdigest())
 selected=[]
 while groups and (not args.limit or len(selected)<args.limit):
  for k in list(groups):
   if args.limit and len(selected)>=args.limit:break
   selected.append(groups[k].pop(0))
   if not groups[k]:del groups[k]
 tok=load_tokenizer('/nas/Public/zhangcheng/models/Qwen3-8B');stats=collections.Counter();bad=[];cells=collections.defaultdict(collections.Counter)
 with (dest/'rows.jsonl').open('w') as f:
  for r in selected:
   raw=pathlib.Path(r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];a=audit(json.loads(raw),tok,r['seed']);stats.update(a['counts']);cells[f"{r['split']}/{r['ratio']}/{r['mode']}"].update(a['counts']);f.write(json.dumps(dict(seed=r['seed'],ratio=r['ratio'],mode=r['mode'],path=r['path'],audit=a))+'\n')
   if a['flags']:bad.append(dict(path=r['path'],flags=a['flags']))
 (dest/'summary.json').write_text(json.dumps(dict(records=len(selected),available_accepted=len(rows),scope='deterministic_stratified_sample' if args.limit else 'all_accepted_at_snapshot',counts=stats,cells=cells,failures=bad,interpretation='Reference-reader necessity, not a claim about model causal use. in_only is a turn-level class; host starts empty.'),indent=2))
if __name__=='__main__':main()
