import math

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from self_paging.train.common import configure_yarn


def test_yarn_preserves_pretrained_base_and_extrapolates_finitely():
    cfg = Qwen3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, rope_theta=1e6, max_position_embeddings=40960)
    original = Qwen3RotaryEmbedding(cfg).inv_freq.clone()
    configure_yarn(cfg, 4.)
    rotary = Qwen3RotaryEmbedding(cfg)
    assert rotary.rope_type == "yarn"
    assert cfg.rope_parameters["rope_theta"] == 1e6
    assert cfg.max_position_embeddings == 131072
    assert math.isclose(rotary.attention_scaling, 1 + .1 * math.log(4))
    assert not torch.equal(original, rotary.inv_freq)
    cos, sin = rotary(torch.zeros(1, 1, 64), torch.tensor([[0, 32768, 119999]]))
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
