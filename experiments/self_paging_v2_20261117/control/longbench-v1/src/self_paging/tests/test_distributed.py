"""Real Gloo regression: uneven packing and an empty GRPO rank cannot hang."""
import os
import tempfile

import torch
import torch.multiprocessing as mp


def _worker(rank, init_file):
    from self_paging.train.common import tiny_qwen3
    from self_paging.train.sft import train_sft, _allreduce_grads
    from self_paging.train.grpo import train_step, GRPOConfig
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", rank=rank, world_size=2,
                                         init_method="file://" + init_file)
    model = tiny_qwen3(32)
    def record(n):
        return {"tokens": [1] + [2] * (n - 1), "page_of": [0] * n,
                "seg_of": [0] * n, "snapshots": [[0]],
                "pages": [{"pid": 0}], "loss_mask": [0] + [1] * (n - 1),
                "ctrl_mask": [0] * n,
                "logprobs": [0.] * n, "reward": 1., "cost": 1.}
    records = [record(n) for n in (5, 8, 13, 7, 19)]
    rows = train_sft(model, records, max_tokens=14, grad_accum=2,
                     epochs=1, rank=rank, world=2, attn="sdpa")
    assert sum(r["tokens"] for r in rows) == sum(sum(r["loss_mask"]) for r in records)
    good, bad = record(7), record(8)
    bad["reward"] = 0.
    import copy
    reference = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    cfg = GRPOConfig(attn="sdpa", max_tokens=8)
    row = train_step(model, opt, [[good, bad]] if rank == 0 else [], cfg, world=2)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-4)
    ref_row = train_step(reference, ref_opt, [[good, bad]], cfg)
    assert row["train_tokens"] == ref_row["train_tokens"] == 13
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    before = [p.detach().clone() for p in model.parameters()]
    states = [v["step"].clone() for v in opt.state.values()]
    empty = train_step(model, opt, [[good, good]], cfg, world=2)
    assert not empty["optimizer_step"] and empty["train_tokens"] == 0
    for p, q in zip(model.parameters(), before):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert all(torch.equal(a, b["step"]) for a, b in zip(states, opt.state.values()))
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    other = flat.clone()
    torch.distributed.broadcast(other, src=0)
    torch.testing.assert_close(flat, other, rtol=0, atol=0)
    torch.distributed.destroy_process_group()


def test_uneven_sft_and_empty_grpo_rank():
    with tempfile.TemporaryDirectory() as d:
        ctx = mp.spawn(_worker, args=(os.path.join(d, "rendezvous"),), nprocs=2, join=False)
        import time
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if ctx.join(timeout=1):
                return
        for p in ctx.processes:
            p.terminate()
        raise AssertionError("Distributed synchronization timed out")
