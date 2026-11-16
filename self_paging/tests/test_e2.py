import torch

from self_paging.core.protocol import Tok
from self_paging.eval.e2_residue import make_case, replay
from self_paging.testing.toy_tokenizer import ToyTokenizer
from self_paging.train.common import tiny_qwen3


def test_e2_exact_visible_tokens_positions_and_cache_history():
    tok = Tok(ToyTokenizer())
    case = make_case(100000, tok, 4096)
    assert case['starts'][case['decision']] - case['starts'][case['a']] - len(case['chunks'][case['a']]) == 4096
    model = tiny_qwen3(tok.vocab_size).eval()
    results = [replay(model, case, name) for name in ("OUT-LATE", "OUT-EARLY", "REWRITE")]
    assert results[0][3:] == results[1][3:] == results[2][3:]
    assert case["correct"] != case["stale"]
    # Layer-zero values depend only on token identity. Higher-layer caches
    # can retain different context despite identical final visible inputs.
    def layer_values(kv, layer):
        return torch.cat([kv.kv[p][layer][1] for p in kv.table.in_order(kv.table.resident())], -2)
    torch.testing.assert_close(layer_values(results[0][0], 0), layer_values(results[2][0], 0), rtol=0, atol=0)
    assert not torch.allclose(layer_values(results[0][0], 1), layer_values(results[2][0], 1), atol=1e-7, rtol=1e-7)
    for kv, _, logits, ids, pos in results:
        assert torch.isfinite(logits).all()
        assert len(ids) == kv.resident_tokens() == len(pos)
        assert pos[-1] == case["prompt_tokens"] - 1
        assert any(b - a > 1 for a, b in zip(pos, pos[1:]))
