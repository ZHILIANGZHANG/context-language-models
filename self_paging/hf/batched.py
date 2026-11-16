"""Batched HF reference feeds for independent, privately paged sessions.

Only one thread calls the shared model. Episode threads submit feed requests;
the coordinator left-pads each private past and right-pads new input, using
explicit per-request causal masks and original global RoPE positions. Padding
is never recorded as real history. This is not an SGLang physical-paging engine.
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError
import queue
import threading
import time

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from ..core.log import TokenLog


class BatchedLog(TokenLog):
    last_logits_only = True

    def __init__(self, coordinator, budget=None):
        super().__init__(budget=budget,strict=False)
        self.coordinator=coordinator
        self.kv={}

    def open(self,kind,pinned=False):
        pid=super().open(kind,pinned=pinned)
        self.kv[pid]=[]
        return pid

    def page_out(self,pids,physical=False):
        pids=super().page_out(pids)
        for p in pids:
            self.kv[p]=[(k.cpu(),v.cpu()) for k,v in self.kv[p]]
        return pids

    def page_in(self,pids):
        pids=super().page_in(pids)
        for p in pids:
            self.kv[p]=[(k.to(self.coordinator.device),v.to(self.coordinator.device)) for k,v in self.kv[p]]
        return pids

    def drop(self,pids):
        pids=super().drop(pids)
        for p in pids:self.kv[p]=[]
        return pids

    def feed(self,pid,ids):
        if not ids:raise ValueError('Empty feed')
        result=None
        for start in range(0,len(ids),self.coordinator.prefill_chunk):
            part=list(ids[start:start+self.coordinator.prefill_chunk])
            if self.coordinator.aborted.is_set():raise RuntimeError('Batched feed cancelled')
            future=Future()
            self.coordinator.requests.put((self,pid,part,future))
            while True:
                try:
                    result=future.result(timeout=.1)
                    break
                except TimeoutError:
                    if self.coordinator.aborted.is_set():raise RuntimeError('Batched feed cancelled')
        return result


class BatchCoordinator:
    def __init__(self,model,max_batch=4,prefill_chunk=512,wait_seconds=.003):
        if max_batch<1 or prefill_chunk<1:raise ValueError('Invalid batch size')
        self.model=model
        self.base=model.get_base_model() if hasattr(model,'get_base_model') else model
        self.device=next(model.parameters()).device
        self.dtype=next(model.parameters()).dtype
        self.max_batch,self.prefill_chunk,self.wait_seconds=max_batch,prefill_chunk,wait_seconds
        self.requests=queue.Queue()
        self.aborted=threading.Event()
        self.batch_sizes=[]

    @torch.no_grad()
    def _feed(self,requests):
        if len({id(log) for log,_,_,_ in requests})!=len(requests):
            raise ValueError('Concurrent requests from the same session')
        pages=[]; lengths=[]
        for log,pid,ids,_ in requests:
            if log.table.current!=pid:raise ValueError('Feed targets a noncurrent page')
            ps=[p for p in log.table.in_order(log.table.resident()) if log.kv[p]]
            pages.append(ps);lengths.append(sum(log.table.pages[p].length for p in ps))
        past=max(lengths,default=0);new=max(len(x[2]) for x in requests);batch=len(requests)
        cache=DynamicCache(config=self.model.config)
        if past:
            for layer in range(self.model.config.num_hidden_layers):
                keys=[];values=[]
                for (log,_,_,_),ps,length in zip(requests,pages,lengths):
                    if ps:
                        k=torch.cat([log.kv[p][layer][0].to(self.device) for p in ps],-2)
                        v=torch.cat([log.kv[p][layer][1].to(self.device) for p in ps],-2)
                    else:
                        heads=self.model.config.num_key_value_heads
                        dim=self.model.config.head_dim
                        k=torch.empty(1,heads,0,dim,dtype=self.dtype,device=self.device);v=k
                    keys.append(F.pad(k,(0,0,past-length,0)))
                    values.append(F.pad(v,(0,0,past-length,0)))
                cache.update(torch.cat(keys,0),torch.cat(values,0),layer)
        ids=torch.zeros(batch,new,dtype=torch.long,device=self.device)
        positions=torch.zeros_like(ids)
        mask=torch.full((batch,1,new,past+new),torch.finfo(self.dtype).min,dtype=self.dtype,device=self.device)
        for i,((log,_,tokens,_),length) in enumerate(zip(requests,lengths)):
            n=len(tokens);ids[i,:n]=torch.tensor(tokens,device=self.device)
            positions[i,:n]=torch.arange(log.log_len,log.log_len+n,device=self.device)
            mask[i,0,:n,past-length:past]=0
            block=torch.triu(torch.full((n,n),torch.finfo(self.dtype).min,dtype=self.dtype,device=self.device),1)
            mask[i,0,:n,past:past+n]=block
            # Discarded padded queries still need one finite attention entry.
            for j in range(n,new):mask[i,0,j,past+j]=0
        out=self.base.model(input_ids=ids,position_ids=positions,attention_mask=mask,
                            past_key_values=cache,use_cache=True)
        last=torch.stack([out.last_hidden_state[i,len(tokens)-1] for i,(_,_,tokens,_) in enumerate(requests)])
        logits=self.base.lm_head(last)
        for i,(log,pid,tokens,future) in enumerate(requests):
            n=len(tokens)
            for layer in range(self.model.config.num_hidden_layers):
                stored=cache.layers[layer]
                k=stored.keys[i:i+1,:,past:past+n].clone()
                v=stored.values[i:i+1,:,past:past+n].clone()
                if len(log.kv[pid])<=layer:log.kv[pid].append((k,v))
                else:
                    oldk,oldv=log.kv[pid][layer]
                    log.kv[pid][layer]=(torch.cat([oldk,k],-2),torch.cat([oldv,v],-2))
            log._record(pid,tokens)
            future.set_result(logits[i:i+1].clone())
        self.batch_sizes.append(batch)

    def run(self,functions):
        """Execute callables fn(coordinator) in bounded waves, preserving result order."""
        if self.aborted.is_set():raise RuntimeError('Cannot reuse a failed coordinator')
        result=[]
        for start in range(0,len(functions),self.max_batch):
            fs=functions[start:start+self.max_batch]
            values=[None]*len(fs);errors=[]
            def worker(i,fn):
                try:values[i]=fn(self)
                except BaseException as error:errors.append(error)
            threads=[threading.Thread(target=worker,args=(i,fn),daemon=True) for i,fn in enumerate(fs)]
            for thread in threads:thread.start()
            try:
                while any(t.is_alive() for t in threads):
                    if errors:raise errors[0]
                    try:first=self.requests.get(timeout=.05)
                    except queue.Empty:continue
                    requests=[first];deadline=time.monotonic()+self.wait_seconds
                    while len(requests)<self.max_batch:
                        remaining=deadline-time.monotonic()
                        if remaining<=0:break
                        try:requests.append(self.requests.get(timeout=remaining))
                        except queue.Empty:break
                    self._feed(requests)
                if errors:raise errors[0]
            except BaseException:
                self.aborted.set()
                raise
            finally:
                for thread in threads:thread.join(timeout=5)
            result.extend(values)
        return result
