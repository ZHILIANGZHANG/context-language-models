"""Read-only SFT-policy difficulty profile; no oracle actions or training."""
import argparse,gzip,hashlib,json,math,os,pathlib,time,traceback
import torch
from audit_protocol import audit
from self_paging.agent.episode import EpisodeConfig,ModelPolicy,run_episode
from self_paging.core.costs import episode_cost
from self_paging.data.rl_preparation import decode_config
from self_paging.data.build_t1_sft import t1_budget
from self_paging.envs.t1_warehouse import WarehouseEnv
from self_paging.hf.logstore import PagedKV
from self_paging.train.common import load_model,load_tokenizer

H=pathlib.Path('/home/zhangcheng/self-paging-20261116');R=H/'rl-profile64'
N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116')
M='/nas/Public/zhangcheng/models/Qwen3-8B';CK=N/'checkpoints/sft-quality-val50-r64'
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def stop():
    if any(p.exists() for p in [H/'STOP',R/'STOP',H/'rl-preparation/STOP',H/'quality-review/STOP']):raise RuntimeError('User STOP')
def main(a):
    torch.set_num_threads(2);stop()
    jobs=json.loads((R/f'shard-{a.shard}.json').read_text());out=N/'data/rl-profile64'/str(a.shard)
    assert not out.exists();out.mkdir(parents=True)
    tok=load_tokenizer(M);model=load_model(str(CK),'sdpa',device='cuda').eval()
    for prompt in jobs:
        for member in range(4):
            stop();env=WarehouseEnv(prompt['seed'],decode_config(prompt['t1_config']))
            budget=t1_budget(env,tok,prompt['ratio']);assert budget==prompt['budget']
            seed=int(hashlib.sha256(f"profile-v1:{prompt['prompt_id']}:{member}".encode()).hexdigest()[:15],16)
            gen=torch.Generator(device='cuda').manual_seed(seed)
            cfg=EpisodeConfig(budget=budget,max_turns=prompt['max_turns'],max_new_tokens=512,
                              max_log_tokens=32768,answer_reserve=1536,thinking=False)
            start=time.time()
            with torch.inference_mode():
                rec=run_episode(env,PagedKV(model,budget=budget,strict=False),
                                ModelPolicy(tok,temperature=1.,top_p=1.,top_k=0,generator=gen),tok,cfg)
            rec['decisions']=[dict(page=d.page,kind=d.kind,gold=d.answer,predicted=env.answers[d.page]) for d in env.ep.decisions]
            rec['environment_pages']=env.ep.pages
            assert rec['reward']==sum(d['gold']==d['predicted'] for d in rec['decisions'])/len(rec['decisions'])
            assert rec['meta']['over_budget']==0 and len(rec['tokens'])<=32768
            assert all(math.isfinite(p) for p,m in zip(rec['logprobs'],rec['loss_mask']) if m)
            counts=audit(rec,tok)
            row=dict(prompt_id=prompt['prompt_id'],member=member,sample_seed=seed,seed=prompt['seed'],ratio=prompt['ratio'],
                     reward=rec['reward'],cost=episode_cost(rec)['cost'],tokens=len(rec['tokens']),
                     seconds=time.time()-start,meta=rec['meta'],protocol=counts,temperature=1.,top_p=1.,top_k=0,
                     checkpoint=str(CK),purpose='Difficulty profile only; not an RL update')
            path=out/f"{prompt['prompt_id']}-{member}.json.gz"
            with gzip.open(path,'wt') as f:json.dump(dict(profile=row,trajectory=rec),f)
            row['raw_path']=str(path);row['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with (R/f'worker-{a.shard}'/'summary.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            write(R/f'worker-{a.shard}'/'progress.json',dict(time=time.time(),prompt_id=prompt['prompt_id'],member=member,seconds=row['seconds']))
            print(json.dumps({k:row[k] for k in ('prompt_id','member','reward','seconds')}),flush=True)
    write(R/f'worker-{a.shard}'/'accepted.json',dict(rows=len(jobs)*4,time=time.time()))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
    try:main(a)
    except BaseException:
        write(R/f'worker-{a.shard}'/'error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
