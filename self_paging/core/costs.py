"""Cost of an episode (docs/algorithm_design.md §5).

    C = Σ_{generated t} (κ + v_t) + Σ_{prefilled t} (κ + v_t)

``v_t`` is the number of keys token t attends to (its resident set at that moment). ``κ`` converts
the per-token weight read into KV-token units and depends on the serving batch size:
κ = weight_bytes / kv_bytes_per_token / batch. For Qwen3-8B (≈16.4 GB bf16, 144 KB/token [K]):
batch 1 → κ ≈ 111K, batch 32 → κ ≈ 3.5K.
"""

from __future__ import annotations

QWEN3_8B_WEIGHT_BYTES = 16.4e9
QWEN3_8B_KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2


def kappa(batch: int, weight_bytes: float = QWEN3_8B_WEIGHT_BYTES,
          kv_bytes: float = QWEN3_8B_KV_BYTES_PER_TOKEN) -> float:
    return weight_bytes / kv_bytes / batch


def visible_counts(rec: dict) -> list[int]:
    """v_t for every token, in O(T + S·P), from an episode record (``TokenLog.record()``).

    Pages are written one after another, so any other visible page is complete; the own page is
    visible up to and including t. Token-level baselines (``hidden_pages`` from ``PagedKV`` with a
    retention mask) subtract the tokens hidden at that version; every head keeps the same number."""
    import bisect

    starts = {p["pid"]: p["start"] for p in rec["pages"]}
    lengths = {p["pid"]: p["length"] for p in rec["pages"]}
    snaps = [set(s) for s in rec["snapshots"]]
    hidden_log = rec.get("hidden_pages") or []
    hidden_versions = [v for v, _ in hidden_log]
    cache: dict[tuple[int, int], int] = {}
    out = []
    for t, (pid, seg) in enumerate(zip(rec["page_of"], rec["seg_of"])):
        key = (seg, pid)
        if key not in cache:
            visible = [q for q in snaps[seg] if q != pid and starts[q] < starts[pid]]
            i = bisect.bisect_right(hidden_versions, seg) - 1
            hidden = dict(hidden_log[i][1]) if i >= 0 else {}
            cache[key] = sum(lengths[q] - hidden.get(q, 0) for q in visible)
        out.append(cache[key] + (t - starts[pid] + 1))
    return out


def episode_cost(rec: dict, batch: int = 32, k: float | None = None) -> dict:
    v = visible_counts(rec)
    k = kappa(batch) if k is None else k
    gen = [x for x, m in zip(v, rec["loss_mask"]) if m]
    pre = [x for x, m in zip(v, rec["loss_mask"]) if not m]
    return {
        "cost": len(v) * k + sum(v),
        "kv_reads_decode": sum(gen),
        "kv_reads_prefill": sum(pre),
        "generated_tokens": len(gen),
        "mean_resident_decode": (sum(gen) / len(gen)) if gen else 0.0,
        "kappa": k,
    }
