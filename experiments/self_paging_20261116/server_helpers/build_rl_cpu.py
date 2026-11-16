"""Unique single-CPU-worker preparation, immutable per-world checkpoints."""
import collections, fcntl, gzip, hashlib, json, os, pathlib, time, traceback, xml.etree.ElementTree as ET
from self_paging.data.rl_preparation import (SPLITS, describe, digest, reference, summary,
                                            split_audit, clean_reference, select_training, select_profile)
from self_paging.train.common import load_tokenizer

H=pathlib.Path('/home/zhangcheng/self-paging-20261116')
C=pathlib.Path('/data/zhangcheng/context-language-models-223cc65-rlprep')
R=H/'rl-preparation'
N=pathlib.Path('/nas/Public/zhangcheng/self-paging-20261116/data/rl-preparation')
M='/nas/Public/zhangcheng/models/Qwen3-8B'

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def identity():
    p=pathlib.Path('/proc/self');return dict(pid=os.getpid(),uid=os.getuid(),start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve()))
def stop():
    if any(p.exists() for p in (H/'STOP', R/'STOP', H/'quality-review/STOP')):raise RuntimeError('User STOP')
def main():
    R.mkdir(exist_ok=True);lock=open(R/'cpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (R/'identity.json').exists(),'Do not duplicate or silently resume'
    stop()
    suites=ET.parse(H/'rl-prep-cpu.xml').getroot().findall('testsuite')
    assert sum(int(s.get('tests')) for s in suites)==11
    assert all(int(s.get(k,0))==0 for s in suites for k in ['failures','errors','skipped'])
    write(R/'identity.json',identity());assert not N.exists();N.mkdir(parents=True)
    manifest=json.loads((H/'rl-prep-source-manifest.json').read_text())
    assert all(hashlib.sha256((C/f).read_bytes()).hexdigest()==sha for f,sha in manifest.items())
    write(R/'source-verified.json',dict(time=time.time(),files=len(manifest)))
    write(N/'split-audit.json',split_audit())
    tok=load_tokenizer(M);rows=[];hashes={};stats=collections.Counter();started=time.time()
    for split,(lo,hi) in SPLITS.items():
        for seed in range(lo,hi):
            stop();base,cfg=describe(seed,split,tok)
            old=hashes.get(base['world_hash']);assert old is None or old==split,'Cross-split world collision'
            hashes[base['world_hash']]=split
            records={};ratios=[1.,.6,.4,.25] if split!='ood' else [1.,.25,.15]
            if base['page_tokens']>28000:
                full=None;fullsum={'skipped':'body >28K; unsupported 32K rollout scope'}
            else:
                full=reference(seed,cfg,tok,1.,'full',base['max_turns']);records['full']=full;fullsum=summary(full)
            for ratio in ratios:
                stop();r=base|dict(ratio=ratio,prompt_id=f't1-{split}-{seed}-{ratio:g}',references={'full':fullsum})
                from self_paging.envs.t1_warehouse import WarehouseEnv
                from self_paging.data.build_t1_sft import t1_budget
                r['budget']=t1_budget(WarehouseEnv(seed,cfg),tok,ratio)
                supported=full is not None and clean_reference(full) and len(full['tokens'])<=32768
                r.update(train_eligible=False,structural_eligible=supported,feasibility='unverified',weight=1.)
                if supported and ratio==1.:
                    r.update(train_eligible=True,feasibility='full_reference_witness',sensitive=False)
                elif supported:
                    for method in ['recent','belady']:
                        stop();rec=reference(seed,cfg,tok,ratio,method,base['max_turns'])
                        records[f'{ratio:g}-{method}']=rec;r['references'][method]=summary(rec)
                    successes=[k for k in ('belady',) if r['references'][k]['completed_cleanly'] and r['references'][k]['tokens']<=32768]
                    if not successes:
                        for method in ['paging','notes']:
                            stop();rec=reference(seed,cfg,tok,ratio,method,base['max_turns'])
                            records[f'{ratio:g}-{method}']=rec;r['references'][method]=summary(rec)
                            if clean_reference(rec) and len(rec['tokens'])<=32768:successes.append(method);break
                    r['train_eligible']=bool(successes)
                    r['feasibility']='reference_witness' if successes else 'not_established_by_references'
                    r['witness_methods']=successes
                    r['sensitivity']=r['references']['belady']['reward']-r['references']['recent']['reward']
                    r['sensitive']=r['sensitivity']>=.25;r['weight']=2. if r['sensitive'] else 1.
                if not supported:r['feasibility']='outside_supported_length_or_reference_failed'
                rows.append(r);stats[f'{split}/total']+=1;stats[f'{split}/eligible']+=int(r['train_eligible'])
                with (N/'cpu-profile.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            archive=N/'references'/split/f'{seed}.json.gz';archive.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(archive,'wt') as f:json.dump(records,f)
            write(R/'worlds'/split/f'{seed}.json',dict(seed=seed,split=split,archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
            write(R/'health.json',dict(time=time.time(),stage='CPU_REFERENCES',split=split,seed=seed,worlds=len(hashes),planned_worlds=3500,conditions=len(rows),seconds=time.time()-started,stats=dict(stats)))
            print(json.dumps(dict(seed=seed,split=split,conditions=len(rows),seconds=time.time()-started)),flush=True)
    selected,gaps=select_training(rows)
    def jsonl(name,rs):
        p=N/name;p.write_text(''.join(json.dumps(r)+'\n' for r in rs));return hashlib.sha256(p.read_bytes()).hexdigest()
    sha={'train':jsonl('rl_train_pool.jsonl',selected)}
    # Evaluation is not selected by policy success or pager sensitivity. Retain flags.
    for split in ['dev','test','ood']:
        sha[split]=jsonl(f'{split}_all_conditions.jsonl',[r for r in rows if r['split']==split])
    profile=select_profile(selected);sha['profile64']=jsonl('profile64.jsonl',profile)
    audit=split_audit()|dict(unique_worlds=len(hashes),cross_split_world_hash_overlap=0)
    write(N/'split-audit.json',audit)
    coverage={str(ratio):dict(collections.Counter(r['n_decisions'] for r in selected if r['ratio']==ratio)) for ratio in [1.,.6,.4,.25]}
    status='CPU_POOL_ACCEPTED_GPU_PROFILE_PENDING' if not any(gaps.values()) else 'COVERAGE_GAPS_GPU_BLOCKED'
    result=dict(time=time.time(),status=status,conditions=len(rows),selected=len(selected),gaps=gaps,coverage=coverage,sha256=sha,stats=dict(stats),source_manifest_sha256=hashlib.sha256((H/'rl-prep-source-manifest.json').read_bytes()).hexdigest(),scope='32K-supported T1; OOD outside this scope retained with flags, not silently truncated')
    write(N/'cpu-accepted.json',result);write(R/'complete.json',result)

if __name__=='__main__':
    try:main()
    except BaseException:
        write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
