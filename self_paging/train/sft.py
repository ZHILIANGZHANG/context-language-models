"""SFT on episode records with the exact self-paging mask (docs/experiment_plan.md §5).

One packed sequence per micro-batch; loss on assistant-generated tokens only. Single process works
on CPU (tests, smoke runs) and on one GPU. Under torchrun: full fine-tuning is sharded with FSDP2;
LoRA keeps a full replica per GPU and all-reduces the adapter gradients. The multi-GPU paths are
written for the server and are not exercised by the CPU tests.

    torchrun --nproc_per_node 8 -m self_paging.train.sft --model /path/Qwen3-8B --tokenizer /path/Qwen3-8B \\
        --data data/t1_sft.jsonl --out ckpt/sft --attn flex_attention --grad-ckpt --lr 5e-6 --epochs 2
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from .batch import pack, token_logprobs
from .common import (append_jsonl, cosine_lr, dist_info, load_model, load_tokenizer, read_jsonl,
                     shard_fsdp2, tiny_qwen3)


def micro_batches(records: list[dict], max_tokens: int, rng: random.Random) -> list[list[dict]]:
    order = records[:]
    rng.shuffle(order)
    out, cur, n = [], [], 0
    for r in order:
        if cur and n + len(r["tokens"]) > max_tokens:
            out.append(cur)
            cur, n = [], 0
        cur.append(r)
        n += len(r["tokens"])
    if cur:
        out.append(cur)
    return out


def sft_loss(model, recs: list[dict], attn: str, chunk: int = 2048):
    p = pack(recs)
    lp = token_logprobs(model, p, attn=attn, chunk=chunk)
    m = p.loss_mask.to(lp.device).float()
    return -(lp * m).sum() / m.sum().clamp(min=1), int(m.sum())


def _allreduce_grads(model, world: int) -> None:
    if world == 1:
        return
    for prm in model.parameters():
        if prm.requires_grad and prm.grad is not None:
            torch.distributed.all_reduce(prm.grad)
            prm.grad /= world


@torch.no_grad()
def evaluate(model, records, attn, max_tokens, chunk=2048) -> float:
    model.eval()
    tot, n = 0.0, 0
    for mb in micro_batches(records, max_tokens, random.Random(0)):
        loss, k = sft_loss(model, mb, attn, chunk)
        tot += float(loss) * k
        n += k
    model.train()
    return tot / max(1, n)


def train_sft(model, records, *, epochs=1, lr=1e-5, max_tokens=32768, attn="sdpa", grad_accum=1,
              warmup_frac=0.03, weight_decay=0.0, clip=1.0, eval_records=None, out=None, log=None,
              seed=0, chunk=2048, rank=0, world=1, sharded=False) -> list[dict]:
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    mine = records[rank::world]
    steps_per_epoch = max(1, len(micro_batches(mine, max_tokens, random.Random(0))) // grad_accum)
    total = steps_per_epoch * epochs
    history, step = [], 0
    model.train()
    for ep in range(epochs):
        mbs = micro_batches(mine, max_tokens, rng)
        for i, mb in enumerate(mbs):
            t0 = time.time()
            loss, ntok = sft_loss(model, mb, attn, chunk)
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum and i + 1 != len(mbs):
                continue
            if not sharded:
                _allreduce_grads(model, world)
            torch.nn.utils.clip_grad_norm_(params, clip)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr, int(warmup_frac * total))
            opt.step()
            opt.zero_grad(set_to_none=True)
            row = {"step": step, "epoch": ep, "loss": loss.item(), "tokens": ntok,
                   "lr": opt.param_groups[0]["lr"], "sec": round(time.time() - t0, 3)}
            history.append(row)
            if log and rank == 0:
                append_jsonl(log, row)
            step += 1
        if eval_records:
            ev = {"epoch": ep, "eval_loss": evaluate(model, eval_records, attn, max_tokens, chunk)}
            history.append(ev)
            if log and rank == 0:
                append_jsonl(log, ev)
    if out:
        save(model, out, rank, sharded)
    return history


def save(model, out, rank: int, sharded: bool) -> None:
    if sharded:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if rank == 0:
            model.save_pretrained(out, state_dict=sd)
    elif rank == 0:
        model.save_pretrained(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path, or 'tiny' for a random CPU model")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data")
    ap.add_argument("--out")
    ap.add_argument("--log")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--yarn", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rank, world, local = dist_info()
    tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if a.model == "tiny":
        model = load_model(tiny_qwen3(tok.vocab_size), a.attn, device=device, lora_r=a.lora_r)
    else:
        model = load_model(a.model, a.attn, device=device, lora_r=a.lora_r, grad_ckpt=a.grad_ckpt,
                           yarn_factor=a.yarn)
    sharded = world > 1 and not a.lora_r
    if sharded:
        shard_fsdp2(model)
    recs = read_jsonl(a.data)
    ev = read_jsonl(a.eval_data) if a.eval_data else None
    train_sft(model, recs, epochs=a.epochs, lr=a.lr, max_tokens=a.max_tokens, attn=a.attn,
              grad_accum=a.grad_accum, eval_records=ev, out=a.out, log=a.log, seed=a.seed,
              rank=rank, world=world, sharded=sharded)


if __name__ == "__main__":
    main()
