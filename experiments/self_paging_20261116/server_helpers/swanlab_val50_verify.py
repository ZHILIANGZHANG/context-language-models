import contextlib,json,math,os,pathlib,time
H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'quality-training-val50';A=R/'swanlab'
import swanlab
with open(os.devnull,'w') as null,contextlib.redirect_stdout(null),contextlib.redirect_stderr(null):
 api=swanlab.Api(api_key=(H/'.swanlab-key').read_text().strip())
 assert api.user().username=='zhangcheng525'
 assert api.project('zhangcheng525/context-language-models').visibility=='PRIVATE'
identity=json.loads((A/'run.json').read_text())
expected={r['step']:r for r in map(json.loads,(R/'sft.jsonl').read_text().splitlines()) if 'step' in r}
run=api.run(f"zhangcheng525/context-language-models/{identity['id']}")
matched={}
validation={r['eval_step']:r['eval_loss'] for r in map(json.loads,(R/'sft.jsonl').read_text().splitlines()) if 'eval_loss' in r}
validation_matched=[]
for series in run.series():
 if series.key=='validation/loss' and validation:
  points=series.metric(sample=100,ignore_timestamp=True)['metrics'];actual={r['index']:r['data'] for r in points}
  assert validation.keys()<=actual.keys(),'Validation points not uploaded yet'
  assert all(math.isclose(float(v),float(actual[i]),rel_tol=1e-6,abs_tol=1e-8) for i,v in validation.items())
  validation_matched=sorted(validation)
 if series.key not in ['train/loss','train/lr','train/optimizer_step']:continue
 key={'train/loss':'loss','train/lr':'lr','train/optimizer_step':'step'}[series.key]
 points=series.metric(sample=100,ignore_timestamp=True)['metrics']
 actual={r['index']:r['data'] for r in points};common=expected.keys() & actual.keys()
 assert common,series.key
 assert all(math.isclose(float(expected[i][key]),float(actual[i]),rel_tol=1e-6,abs_tol=1e-8) for i in common),series.key
 matched[series.key]=sorted(common)
assert len(matched)==3,matched
common=set.intersection(*(set(x) for x in matched.values()));assert common
assert not validation or validation_matched==sorted(validation),'Missing cloud validation metrics'
out=dict(validation_matching_steps=validation_matched,time=time.time(),url=run.url,id=identity['id'],matching_steps=sorted(common),local_latest=max(expected),private=True)
(A/'cloud-verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
