"""Periodic validation/checkpoints: exact boundaries and weighted distributed evaluation."""
import json
from pathlib import Path
import pytest
import torch
import torch.multiprocessing as mp
from self_paging.train import sft

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(0.5))
    def save_pretrained(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(out)/"weights.pt")

def loss(model, records, attn, chunk):
    count = sum(sum(r["loss_mask"]) for r in records)
    value = sum(((model.weight-r["target"])**2)*sum(r["loss_mask"]) for r in records)/count
    return value, count

def records(n):
    return [dict(tokens=[1]*(i+1), loss_mask=[1]*(i+1), target=float(i%3)) for i in range(n)]

@pytest.mark.parametrize("n,expected", [(5,[2,4,5]), (4,[2,4])])
def test_periodic_boundaries_and_reload(tmp_path, monkeypatch, n, expected):
    monkeypatch.setattr(sft,"sft_loss",loss)
    model=Toy(); log=tmp_path/"log.jsonl"; out=tmp_path/"model"
    rows=sft.train_sft(model,records(n),max_tokens=1,lr=.1,warmup_frac=0,
        eval_records=records(3),out=str(out),log=str(log),eval_every=2,save_every=2)
    assert [r["completed_steps"] for r in rows if "eval_loss" in r]==expected
    assert sum("step" in r for r in rows)==n
    assert model.training
    for step in expected:
        ck=out/f"checkpoint-{step:06d}"
        meta=json.loads((ck/"checkpoint.json").read_text())
        restored=Toy(); restored.load_state_dict(torch.load(ck/"weights.pt",weights_only=True))
        value=sft.evaluate(restored,records(3),"sdpa",1)
        assert value==pytest.approx(meta["eval_loss"],abs=1e-7)
        assert meta["completed_steps"]==step
    assert torch.equal(torch.load(out/"weights.pt",weights_only=True)["weight"],model.weight.detach())
    assert [json.loads(l) for l in log.read_text().splitlines()]==rows

def _worker(rank, init, result):
    torch.distributed.init_process_group("gloo",init_method=init,rank=rank,world_size=2)
    sft.sft_loss=loss
    model=Toy(); value=sft.evaluate(model,records(3),"sdpa",1,rank=rank,world=2)
    values=[None]*2;torch.distributed.all_gather_object(values,value)
    if rank==0:Path(result).write_text(json.dumps(values))
    torch.distributed.destroy_process_group()

def test_distributed_validation_weighting(tmp_path,monkeypatch):
    monkeypatch.setattr(sft,"sft_loss",loss)
    expected=sft.evaluate(Toy(),records(3),"sdpa",1)
    mp.spawn(_worker,args=("file://"+str(tmp_path/"init"),str(tmp_path/"result")),nprocs=2,join=True)
    assert json.loads((tmp_path/"result").read_text())==pytest.approx([expected,expected])

def test_missing_validation_configuration():
    with pytest.raises(ValueError,match="eval_records"):
        sft.train_sft(Toy(),records(1),eval_every=50)
    with pytest.raises(ValueError,match="require out"):
        sft.train_sft(Toy(),records(1),save_every=50)
