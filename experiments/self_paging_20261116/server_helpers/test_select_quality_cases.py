from select_quality_cases import select

def test_balanced_and_insufficient():
 rows=[]
 for seed in range(100):
  for ratio in [1.,.6,.45,.35]:
   for mode,kind in ([('paging','none')] if ratio==1 else [('paging','out'),('notes','out'),('eager','both')]):
    calls={} if kind=='none' else {'kv_out':3}|({'kv_in':1} if kind=='both' else {})
    rows.append(dict(seed=seed,ratio=ratio,mode=mode,accepted=True,audit={'calls':calls}))
 a,s=select(rows,60,40);b,t=select(rows[::-1],60,40)
 assert a==b and s==t and not s['deficits'] and len(a)==60 and s['distinct_seeds']>=40
 assert sum(r['ratio']==1 for r in a)==12
 for ratio in [.6,.45,.35]:assert sum(r['ratio']==ratio for r in a)==16
 _,s=select([r for r in rows if r['ratio']!=.35],60,40)
 assert s['deficits']
