"""Real-checkpoint forward parity and bounded-cache witness before task evaluation."""
import argparse,json,time
from pathlib import Path
import torch
from ..hf.logstore import PagedKV
from ..hf.faithful_cache import FaithfulKV
from ..train.common import load_model,load_tokenizer
from .faithful_eval import atomic


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--tokenizer',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();torch.manual_seed(0)
    tok=load_tokenizer(a.tokenizer);model=load_model(a.model,'sdpa',device='cuda').eval()
    text='Read this warehouse log carefully. Shelf S7 is empty. Shelf S3 is quarantined. Store the next item on the lowest available non-quarantined shelf. '
    ids=tok.encode(text*8)[:192]
    chunks=[ids[:64],ids[64:128],ids[128:]]
    native=PagedKV(model);p=native.open('tool');reference=[native.feed(p,x).detach().cpu() for x in chunks]
    del native;torch.cuda.empty_cache()
    result=dict(device=torch.cuda.get_device_name(),dtype=str(next(model.parameters()).dtype),checks={})
    for method in ['streaming','h2o']:
        log=FaithfulKV(model,method,512);p=log.open('tool');diff=[];match=[]
        try:
            for x,y in zip(chunks,reference):
                actual=log.feed(p,x).detach().cpu();diff.append(float((actual.float()-y.float()).abs().max()));match.append(bool(torch.equal(actual.argmax(-1),y.argmax(-1))))
            result['checks'][method+'_no_eviction']=dict(max_abs=max(diff),all_top1_equal=all(match),evictions=log.evictions)
            assert max(diff)<1e-4 and all(match) and log.evictions==0,result
        finally:log.close()
        del log;torch.cuda.empty_cache()
        log=FaithfulKV(model,method,64);p=log.open('tool');started=time.time()
        try:
            logits=log.feed(p,ids)
            assert torch.isfinite(logits).all() and log.peak_resident==64 and log.evictions==len(ids)-63
            for s in log.state:
                assert s[0].shape[-2]==63 and s[1].shape[-2]==63
                assert (s[3][:,1:]>s[3][:,:-1]).all()
            result['checks'][method+'_bounded']=dict(tokens=len(ids),seconds=time.time()-started,
                peak=log.peak_resident,evictions=log.evictions,position_example=log.state[0][3][0].cpu().tolist())
        finally:log.close()
        del log;torch.cuda.empty_cache()
    result['passed']=True;atomic(Path(a.out),result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
