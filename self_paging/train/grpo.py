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
    excluded_seed_ranges: tuple[tuple[int, int], ...] = ((90000, 90100),)
    max_sampling_rounds: int = 8  # bounded replacement, never silently train an empty batch
    temperature: float = 1.0
    max_turns: int = 30
    max_new_tokens: int = 512
    max_tokens: int = 32768         # per packed micro-batch
    rollout_batch: int = 1         # HF reference batching; requires independent-session gate
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


def batched_rollout_group(model, tok, seed, ratio, cfg: GRPOConfig, gen) -> list[dict]:
    """Same episode protocol, independent RNG and KV state for each group member.

    Seeds are allocated before threads start, making results independent of
    thread scheduling. Batching is a reference implementation, not a latency claim.
    """
    from ..hf.batched import BatchCoordinator, BatchedLog
    device = next(model.parameters()).device
    seeds = torch.randint(0, 2**62, (cfg.group,), device=device, generator=gen).cpu().tolist()
    coordinator = BatchCoordinator(model, max_batch=cfg.rollout_batch)

    def episode(index):
        def run(c):
            env = WarehouseEnv(seed, cfg.t1)
            budget = t1_budget(env, tok, ratio)
            ecfg = EpisodeConfig(budget=budget, max_turns=cfg.max_turns,
                                 max_new_tokens=cfg.max_new_tokens,
                                 reversible=cfg.reversible, notes=cfg.notes)
            member_gen = torch.Generator(device=device).manual_seed(seeds[index])
            rec = run_episode(env, BatchedLog(c, budget=budget),
                              ModelPolicy(tok, temperature=cfg.temperature, generator=member_gen), tok, ecfg)
            rec['meta'].update(seed=seed, ratio=ratio, member_seed=seeds[index],
                               rollout_backend='hf_private_batch')
            rec['cost'] = episode_cost(rec, batch=cfg.kappa_batch)['cost']
            return rec
        return run

    return coordinator.run([episode(i) for i in range(cfg.group)])


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
    device = next(model.parameters()).device
    counts = torch.tensor([len(groups), kept, sum(sum(r["loss_mask"]) for r in recs)],
                          device=device, dtype=torch.float64)
    if world > 1:
        torch.distributed.all_reduce(counts)
    global_tokens = int(counts[2])
    stats = {"groups": int(counts[0]), "groups_kept": int(counts[1]),
             "train_tokens": global_tokens, "optimizer_step": bool(global_tokens)}
    model.train()
    opt.zero_grad(set_to_none=True)
    if not global_tokens:
        # Adam momentum / weight decay can change weights even with zero grads.
        return stats
    totals = torch.zeros(2, device=device, dtype=torch.float64)
    if recs:
        # Exact global token mean, independent of rank or packing boundaries.
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
            ntok = s["train_tokens"]
            (loss * (world * ntok / global_tokens)).backward()
            totals[0] += loss.detach().double() * ntok
            totals[1] += s["logprob_mismatch"] * ntok
    if world > 1:
        for prm in model.parameters():
            if prm.requires_grad:
                if prm.grad is None:
                    prm.grad = torch.zeros_like(prm)
                torch.distributed.all_reduce(prm.grad)
                prm.grad /= world
        torch.distributed.all_reduce(totals)
    stats.update(loss=float(totals[0] / global_tokens),
                 logprob_mismatch=float(totals[1] / global_tokens))
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


def set_attn(model, name: str) -> None:
    """Rollouts decode token by token with a dense mask (sdpa is the fast path); training uses the
    BlockMask (flex_attention). Switching is a config flag, no weights move."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.set_attn_implementation(name)


def sample_seed(rng, cfg):
    lo, hi = cfg.seed_range
    intervals = [(lo, hi)] if hi > lo else []
    for a, b in cfg.excluded_seed_ranges:
        next_intervals = []
        for l, h in intervals:
            if b <= l or a >= h:
                next_intervals.append((l, h))
            else:
                next_intervals.extend((x, y) for x, y in [(l, min(a, h)), (max(b, l), h)] if y > x)
        intervals = next_intervals
    total = sum(b - a for a, b in intervals)
    if not total:
        raise ValueError("No training seeds remain after excluding held-out seeds")
    offset = rng.randrange(total)
    for a, b in intervals:
        if offset < b - a:
            return a + offset
        offset -= b - a


def collect_groups(model, tok, cfg, rng, gen, rollout_fn, target):
    """Replace equal-reward groups, with a finite diagnostic stop if no signal exists."""
    sampled, kept = [], []
    for _ in range(cfg.max_sampling_rounds):
        for _ in range(target - len(kept)):
            g = rollout_fn(model, tok, sample_seed(rng, cfg), rng.choice(cfg.ratios), cfg, gen)
            if len(g) != cfg.group:
                raise ValueError("Rollout group size differs from configured G")
            sampled.append(g)
            if informative([r["reward"] for r in g]):
                kept.append(g)
        if len(kept) == target:
            break
    return sampled, kept


def global_summary(groups, world, device):
    row = summarize(groups)
    # Reduce weighted sufficient statistics only, never gather large trajectories.
    keys = list(summarize([[{"reward": 0., "cost": 0., "peak_resident": 0,
                            "tokens": [], "meta": {}}]]))
    n = sum(len(g) for g in groups)
    v = torch.tensor([n] + [row.get(k, 0.) * n for k in keys], device=device, dtype=torch.float64)
    if world > 1:
        torch.distributed.all_reduce(v)
    return {k: float(v[i + 1] / v[0].clamp(min=1)) for i, k in enumerate(keys)}


def train_grpo(model, tok, cfg: GRPOConfig, log=None, out=None, rank=0, world=1, rollout_fn=None,
               rollout_attn: str = "sdpa", save_every: int = 0) -> list[dict]:
    rng = random.Random(cfg.seed + rank)
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(cfg.seed + rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, betas=(0.9, 0.95))
    rollout_fn = rollout_fn or (batched_rollout_group if cfg.rollout_batch > 1 else rollout_group)
    history = []
    for step in range(cfg.steps):
        t0 = time.time()
        model.eval()
        set_attn(model, rollout_attn)
        target = len(range(rank, cfg.prompts, world))
        sampled, groups = collect_groups(model, tok, cfg, rng, gen, rollout_fn, target)
        device = next(model.parameters()).device
        enough = torch.tensor(int(len(groups) == target), device=device)
        if world > 1:
            torch.distributed.all_reduce(enough, op=torch.distributed.ReduceOp.MIN)
        if not enough.item():
            raise RuntimeError("Dynamic sampling exhausted; no optimizer update. Inspect reward/difficulty.")
        set_attn(model, cfg.attn)
        sampled_count = torch.tensor(len(sampled), device=device)
        if world > 1:
            torch.distributed.all_reduce(sampled_count)
        row = {"step": step, **global_summary(sampled, world, device),
               **train_step(model, opt, groups, cfg, world), "groups_sampled": int(sampled_count),
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        if log and rank == 0:
            append_jsonl(log, row)
        if out and save_every and rank == 0 and (step + 1) % save_every == 0:
            model.save_pretrained(f"{out}/step_{step + 1}")
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
    ap.add_argument("--rollout-batch", type=int, default=1)
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
    ap.add_argument("--save-every", type=int, default=0, help="also save OUT/step_K every K steps")
    ap.add_argument("--yarn", type=float, default=0.0)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    if a.rollout_batch < 1:
        ap.error('--rollout-batch must be positive')
    rank, world, local = dist_info()
    torch.manual_seed(a.seed)  # Replicated adapters must start identically.
    tok = load_tokenizer(a.tokenizer or (None if a.model == "tiny" else a.model))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    base = tiny_qwen3(tok.vocab_size) if a.model == "tiny" else a.model
    model = load_model(base, a.attn, device=device, lora_r=a.lora_r, yarn_factor=a.yarn,
                       grad_ckpt=a.grad_ckpt)
    cfg = GRPOConfig(steps=a.steps, prompts=a.prompts, group=a.group, lr=a.lr, w_eff=a.w_eff, attn=a.attn,
                     max_new_tokens=a.max_new_tokens, max_turns=a.max_turns, reversible=not a.irreversible,
                     notes=not a.no_notes, seed=a.seed, t1=T1Config(n_decisions=a.n_decisions),
                     rollout_batch=a.rollout_batch)
    if rank == 0 and a.log:
        append_jsonl(a.log, {"config": {k: str(v) for k, v in asdict(cfg).items()}})
    train_grpo(model, tok, cfg, log=a.log, out=a.out, rank=rank, world=world, save_every=a.save_every)


if __name__ == "__main__":
    main()
