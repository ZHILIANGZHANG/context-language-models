"""GRPO for self-paging (docs/data_and_training.md §3).

Rollouts run through ``PagedKV`` (the HF reference engine), so they are exact but slow: use it for
T1 and small runs. Under torchrun every rank keeps a full replica (use LoRA for 8B), rolls out its
share of prompts, and the trainable gradients are all-reduced. A faster rollout backend (SGLang
overlay, docs/experiment_plan.md §7.2) plugs in through ``rollout_fn``.

Per step: sample prompts (seed, budget ratio) → G rollouts each → reward and cost → dual-channel
advantages → drop groups with equal rewards (dynamic sampling) → token-level clipped policy loss
with truncated importance weights against the behaviour log-probs.

    python -m self_paging.train.grpo --model ckpt/sft --tokenizer /path/Qwen3-8B --out ckpt/rl \\
        --steps 200 --prompts 16 --group 8 --lora-r 64 --lr 1e-5
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass, field

import torch

from ..agent.episode import EpisodeConfig, ModelPolicy, run_episode
from ..core.costs import episode_cost
from ..data.build_t1_sft import t1_budget
from ..envs.t1_warehouse import T1Config, WarehouseEnv
from ..hf.logstore import PagedKV
from .advantage import group_advantages, informative, token_advantages
from .batch import pack, token_logprobs
from .common import append_jsonl, dist_info, load_model, load_tokenizer, tiny_qwen3


@dataclass
class GRPOConfig:
    steps: int = 200
    prompts: int = 16               # per step, across all ranks
    group: int = 8
    lr: float = 1e-6
    clip_low: float = 0.2
    clip_high: float = 0.28
    w_eff: float = 0.25
    tis_cap: float = 2.0
    kappa_batch: int = 32
    success_at: float = 1.0
    ratios: tuple[float, ...] = (0.25, 0.4, 0.6)
    seed_range: tuple[int, int] = (0, 100000)
    temperature: float = 1.0
    max_turns: int = 30
    max_new_tokens: int = 512
    max_tokens: int = 32768         # per packed micro-batch
    attn: str = "flex_attention"
    reversible: bool = True
    notes: bool = True
    recompute_old: bool = True
    seed: int = 0
    t1: T1Config = field(default_factory=T1Config)


def rollout_group(model, tok, seed, ratio, cfg: GRPOConfig, gen) -> list[dict]:
    out = []
    for _ in range(cfg.group):
        env = WarehouseEnv(seed, cfg.t1)
        budget = t1_budget(env, tok, ratio)
        ecfg = EpisodeConfig(budget=budget, max_turns=cfg.max_turns, max_new_tokens=cfg.max_new_tokens,
                             reversible=cfg.reversible, notes=cfg.notes)
        rec = run_episode(env, PagedKV(model, budget=budget, strict=False),
                          ModelPolicy(tok, temperature=cfg.temperature, generator=gen), tok, ecfg)
        rec["meta"].update(seed=seed, ratio=ratio)
        rec["cost"] = episode_cost(rec, batch=cfg.kappa_batch)["cost"]
        out.append(rec)
    return out


def policy_loss(model, recs, advs, cfg: GRPOConfig):
    p = pack(recs, advs)
    device = next(model.parameters()).device
    lp = token_logprobs(model, p, attn=cfg.attn)
    mask = p.loss_mask.to(device).float()
    beh = p.old_logprobs.to(device)
    if cfg.recompute_old:
        with torch.no_grad():
            old = token_logprobs(model, p, attn=cfg.attn)
    else:
        old = beh
    ratio = torch.exp(lp - old)
    tis = torch.exp(old - beh).clamp(max=cfg.tis_cap).detach()
    adv = p.adv.to(device)
    pg = -torch.minimum(ratio * adv, ratio.clamp(1 - cfg.clip_low, 1 + cfg.clip_high) * adv) * tis
    loss = (pg * mask).sum() / mask.sum().clamp(min=1)
    with torch.no_grad():
        mismatch = ((old - beh).abs() * mask).sum() / mask.sum().clamp(min=1)
    return loss, {"logprob_mismatch": float(mismatch), "train_tokens": int(mask.sum())}


def train_step(model, opt, groups: list[list[dict]], cfg: GRPOConfig, world: int = 1) -> dict:
    recs, advs, kept = [], [], 0
    for g in groups:
        rewards = [r["reward"] for r in g]
        if not informative(rewards):
            continue
        kept += 1
        a_task, a_eff = group_advantages(rewards, [r["cost"] for r in g], cfg.success_at)
        for r, at, ae in zip(g, a_task, a_eff):
            recs.append(r)
            advs.append(token_advantages(r, at, ae, cfg.w_eff))
    stats = {"groups": len(groups), "groups_kept": kept}
    model.train()
    opt.zero_grad(set_to_none=True)
    if recs:
        # micro-batches by token budget; loss averaged over micro-batches
        mbs, cur, n = [], [], 0
        for r, a in zip(recs, advs):
            if cur and n + len(r["tokens"]) > cfg.max_tokens:
                mbs.append(cur)
                cur, n = [], 0
            cur.append((r, a))
            n += len(r["tokens"])
        mbs.append(cur)
        for mb in mbs:
            loss, s = policy_loss(model, [r for r, _ in mb], [a for _, a in mb], cfg)
            (loss / len(mbs)).backward()
            stats.update(s)
            stats["loss"] = stats.get("loss", 0.0) + loss.item() / len(mbs)
    if world > 1:
        for prm in model.parameters():
            if prm.requires_grad and prm.grad is not None:
                torch.distributed.all_reduce(prm.grad)
                prm.grad /= world
    if recs or world > 1:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return stats


def summarize(groups) -> dict:
    flat = [r for g in groups for r in g]
    if not flat:
        return {}
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    m = lambda k: mean([r["meta"].get(k, 0) for r in flat])  # noqa: E731
    return {"reward": mean([r["reward"] for r in flat]),
            "success": mean([float(r["reward"] >= 1.0) for r in flat]),
            "cost": mean([r["cost"] for r in flat]), "peak_resident": mean([r["peak_resident"] for r in flat]),
            "tokens": mean([len(r["tokens"]) for r in flat]), "turns": m("turns"), "invalid": m("invalid"),
            "kv_out": m("kv_out"), "kv_in": m("kv_in"), "notes": m("notes"), "truncated": m("truncated_tokens"),
            "forced_evictions": m("forced_evictions")}


def train_grpo(model, tok, cfg: GRPOConfig, log=None, out=None, rank=0, world=1, rollout_fn=None) -> list[dict]:
    rng = random.Random(cfg.seed + rank)
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(cfg.seed + rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, betas=(0.9, 0.95))
    rollout_fn = rollout_fn or rollout_group
    history = []
    for step in range(cfg.steps):
        t0 = time.time()
        model.eval()
        groups = []
        for _ in range(max(1, cfg.prompts // world)):
            seed, ratio = rng.randrange(*cfg.seed_range), rng.choice(cfg.ratios)
            groups.append(rollout_fn(model, tok, seed, ratio, cfg, gen))
        row = {"step": step, **summarize(groups), **train_step(model, opt, groups, cfg, world),
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        if log and rank == 0:
            append_jsonl(log, row)
    if out and rank == 0:
        model.save_pretrained(out)
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF path / SFT checkpoint, or 'tiny'")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out")
    ap.add_argument("--log")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--w-eff", type=float, default=0.25)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--irreversible", action="store_true", help="K2 ablation: kv_drop instead of kv_out/kv_in")
    ap.add_argument("--no-notes", action="store_true")
    ap.add_argument("--n-decisions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rank, world, local = dist_info()
    tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    base = tiny_qwen3(tok.vocab_size) if a.model == "tiny" else a.model
    model = load_model(base, a.attn, device=device, lora_r=a.lora_r)
    cfg = GRPOConfig(steps=a.steps, prompts=a.prompts, group=a.group, lr=a.lr, w_eff=a.w_eff, attn=a.attn,
                     max_new_tokens=a.max_new_tokens, max_turns=a.max_turns, reversible=not a.irreversible,
                     notes=not a.no_notes, seed=a.seed, t1=T1Config(n_decisions=a.n_decisions))
    if rank == 0 and a.log:
        append_jsonl(a.log, {"config": {k: str(v) for k, v in asdict(cfg).items()}})
    train_grpo(model, tok, cfg, log=a.log, out=a.out, rank=rank, world=world)


if __name__ == "__main__":
    main()
