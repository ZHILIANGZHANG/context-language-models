"""Shared training utilities: model/tokenizer loading, LoRA, FSDP2, JSONL I/O, schedules."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(path, row) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def dist_info() -> tuple[int, int, int]:
    """(rank, world_size, local_rank); initialises torch.distributed under torchrun."""
    if "RANK" in os.environ and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend)
    if torch.distributed.is_initialized():
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return torch.distributed.get_rank(), torch.distributed.get_world_size(), local
    return 0, 1, 0


def configure_yarn(cfg, factor: float):
    """Preserve the pretrained RoPE base under the Transformers 5 config API."""
    if factor <= 1:
        raise ValueError("YaRN extrapolation factor must exceed 1")
    current = getattr(cfg, "rope_parameters", None) or {}
    theta = current.get("rope_theta") or getattr(cfg, "rope_theta", None)
    if theta is None:
        raise ValueError("Missing pretrained rope_theta; cannot safely enable YaRN")
    params = {"rope_type": "yarn", "rope_theta": theta, "factor": factor,
              "original_max_position_embeddings": 32768}
    if hasattr(cfg, "rope_parameters"):
        cfg.rope_parameters = params
    else:
        cfg.rope_scaling = {k: v for k, v in params.items() if k != "rope_theta"}
    cfg.max_position_embeddings = int(32768 * factor)
    return cfg


def load_model(path_or_model, attn: str, dtype: str = "bfloat16", device: str | None = None,
               lora_r: int = 0, lora_alpha: int = 0, grad_ckpt: bool = False, yarn_factor: float = 0.0):
    """Load an HF causal LM (or pass one through) with the attention implementation we need.

    ``yarn_factor`` > 0 enables static YaRN (Qwen3: factor 4 → 131K; docs/experiment_plan.md §1.2)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    if isinstance(path_or_model, str) and os.path.exists(os.path.join(path_or_model, "adapter_config.json")):
        # a LoRA checkpoint from train/sft.py or train/grpo.py: load its base, attach the adapter
        from peft import PeftConfig, PeftModel

        pc = PeftConfig.from_pretrained(path_or_model)
        base = load_model(pc.base_model_name_or_path, attn, dtype, None, 0, 0, grad_ckpt, yarn_factor)
        model = PeftModel.from_pretrained(base, path_or_model, is_trainable=True)
        if device:
            model.to(device)
        return model
    if isinstance(path_or_model, str):
        cfg = AutoConfig.from_pretrained(path_or_model)
        if yarn_factor:
            configure_yarn(cfg, yarn_factor)
        model = AutoModelForCausalLM.from_pretrained(path_or_model, config=cfg, torch_dtype=torch_dtype,
                                                     attn_implementation=attn)
    else:
        model = path_or_model
        model.set_attn_implementation(attn)
    if device:
        model.to(device)
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if lora_r:
        from peft import LoraConfig, get_peft_model  # optional dependency

        model = get_peft_model(model, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return model


def shard_fsdp2(model) -> None:
    """FSDP2: shard each decoder layer, then the backbone and the LM head separately (training calls
    ``model.model`` and ``model.lm_head`` directly, so neither may rely on the root's hooks).
    Written for torchrun on GPUs; not exercised by the CPU tests."""
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in base.model.layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(base.model, mp_policy=mp)
    fully_shard(base.lm_head, mp_policy=mp)


def load_tokenizer(path: str | None):
    """HF tokenizer from ``path``; the toy tokenizer when ``path`` is None or "toy"."""
    from ..core.protocol import Tok

    if path in (None, "toy"):
        from ..testing.toy_tokenizer import ToyTokenizer

        return Tok(ToyTokenizer())
    from transformers import AutoTokenizer

    return Tok(AutoTokenizer.from_pretrained(path))


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def tiny_qwen3(vocab: int, seed: int = 0, attn: str = "sdpa", dtype=torch.float32):
    """Random tiny Qwen3 for CPU tests and smoke runs."""
    import transformers

    cfg = transformers.Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=1 << 16, rope_theta=10000.0,
                                   tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(cfg).to(dtype)
    model.set_attn_implementation(attn)
    return model
