import argparse,hashlib,json,pathlib,torch
from self_paging.train.common import dist_info,load_model,load_tokenizer
from self_paging.train.sft import train_sft
from self_paging.data.build_t1_sft import build
from self_paging.envs.t1_warehouse import T1Config
ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--eval-data',required=True);ap.add_argument('--checkpoint-root',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
rank,world,local=dist_info();assert world>=2
torch.manual_seed(0);torch.set_num_threads(4)
M='/nas/Public/zhangcheng/models/Qwen3-8B';tok=load_tokenizer(M)
recs=[json.loads(l) for l in pathlib.Path(a.data).read_text().splitlines()]
eval_recs=[json.loads(l) for l in pathlib.Path(a.eval_data).read_text().splitlines()]
assert len(recs)==2*world and len(eval_recs)==world
model=load_model(M,'flex_attention',device=f'cuda:{local}',lora_r=64,grad_ckpt=True)
params=[p for p in model.parameters() if p.requires_grad];before=[p.detach().clone() for p in params]
history=train_sft(model,recs,epochs=1,lr=1e-4,max_tokens=1,attn='flex_attention',rank=rank,world=world,eval_records=eval_recs,eval_every=1,save_every=1,out=a.checkpoint_root)
assert len(history)==4 and all(torch.isfinite(torch.tensor(r.get('loss',r.get('eval_loss')))) for r in history)
assert [r['completed_steps'] for r in history if 'eval_loss' in r]==[1,2]
changed=sum(int(not torch.equal(p,b)) for p,b in zip(params,before));assert changed>0
h=hashlib.sha256()
for p in params:
 assert torch.isfinite(p).all();h.update(p.detach().view(torch.uint8).cpu().numpy().tobytes())
digests=[None]*world;torch.distributed.all_gather_object(digests,h.hexdigest());assert len(set(digests))==1
if rank==0:
 from safetensors.torch import load_file
 from peft import get_peft_model_state_dict
 final=load_file(str(pathlib.Path(a.checkpoint_root)/'checkpoint-000002/adapter_model.safetensors'))
 live=get_peft_model_state_dict(model)
 assert set(final)==set(live) and all(torch.equal(final[k],live[k].cpu()) for k in final)
 first=load_file(str(pathlib.Path(a.checkpoint_root)/'checkpoint-000001/adapter_model.safetensors'))
 assert any(not torch.equal(first[k],final[k]) for k in final)
 for step in [1,2]:
  meta=json.loads((pathlib.Path(a.checkpoint_root)/f'checkpoint-{step:06d}'/'checkpoint.json').read_text())
  assert meta['completed_steps']==step and meta['eval_loss']==[r['eval_loss'] for r in history if r.get('completed_steps')==step and 'eval_loss' in r][0]
 result=dict(world=world,history=history,lora_r=64,changed_tensors=changed,parameter_sha256=digests[0],tokens=[len(r['tokens']) for r in recs],checkpoint_discarded=True,peak_memory_bytes=torch.cuda.max_memory_allocated())
 pathlib.Path(a.out).write_text(json.dumps(result,indent=2));print('WITNESS_PASS',json.dumps(result),flush=True)
torch.distributed.barrier()
torch.distributed.destroy_process_group()
