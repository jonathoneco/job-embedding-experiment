# Spec 05: RRF Score Fusion (C5)

**Technique:** T5 — Hybrid Dense+Sparse Score Fusion
**Phase:** 2 (Pipeline Extensions)
**Scope:** Small (~80 lines)
**References:** [architecture.md](architecture.md) C5, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/fusion.py` module implementing Reciprocal Rank Fusion (RRF). Combines rankings from multiple methods (e.g., embedding + BM25, or multiple embedding models) into a single fused ranking. RRF is rank-based (not score-based), solving the score incompatibility between embedding cosine similarity ([-1, 1]) and baseline scores ([0, 1]).

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/fusion.py` | New module |
| Modify | `config.yaml` | Add `fusion` section |

## Interface Contract

### Exposes (src/fusion.py)

```python
def fuse_rankings(
    rankings_by_method: dict[str, list[list[dict]]],
    k: int = 60,
    top_n: int = 10,
) -> list[list[dict]]:
    """Fuse multiple ranking lists using Reciprocal Rank Fusion.

    Args:
        rankings_by_method: {method_name: [[{target_id, score}, ...], ...]}
            Each value is a list of per-query rankings (same structure as rank_targets output).
            All methods must have the same number of queries.
        k: RRF constant (default 60, standard value).
        top_n: Number of results to return per query.

    Returns:
        Fused rankings: [[{target_id, score}, ...], ...] per query.
        Scores are RRF scores (sum of 1/(k + rank) across methods).
    """

def fuse_all(
    all_rankings: list[dict],
    fusion_configs: list[dict],
    test_cases: list[dict],
    k: int = 60,
) -> list[dict]:
    """Apply all fusion configs to produce fused ranking results.

    Args:
        all_rankings: Flat list of uniform ranking results (S0.1 format).
        fusion_configs: From config['fusion']['configs'], each specifying
            name, methods list, and granularity.
        test_cases: For test_case_id mapping.
        k: RRF constant.

    Returns:
        List of uniform ranking results (S0.1 format) for each fusion config.
    """
```

### Config schema

```yaml
fusion:
  enabled: false
  k: 60
  configs:
    - name: "fusion-bge-large-bm25"
      methods: ["bge-large", "bm25"]
      granularity: "role"
    - name: "fusion-bge-large-tfidf"
      methods: ["bge-large", "tfidf"]
      granularity: "role"
```

### Consumes

- Uniform ranking results (S0.1 format) from any method
- Both embedding rankings and baseline rankings must be available for the specified granularity

## RRF Algorithm

```
For each query q:
    For each candidate document d appearing in any ranking:
        rrf_score(d) = Σ_method  1 / (k + rank_method(d))

    where rank_method(d) is the 1-based rank of d in that method's ranking.
    If d does not appear in a method's ranking, it contributes 0 to the sum.

    Sort candidates by rrf_score descending, return top_n.
```

**k=60** is the standard constant from the original RRF paper (Cormack et al. 2009). It controls how quickly rank differences diminish. Not tuned on dev set — the standard value is robust across tasks and tuning adds complexity for marginal gain.

## Implementation Steps

### Step 1: Implement `fuse_rankings()`

**What:** Core RRF algorithm operating on per-query ranking lists.

**Acceptance criteria:**
- [ ] For each query, collects all unique target_ids across all methods
- [ ] Computes RRF score for each target: `sum(1/(k + rank) for each method where target appears)`
- [ ] Rank is 1-based (first result has rank 1)
- [ ] Targets not in a method's ranking contribute 0 from that method
- [ ] Returns top_n results per query, sorted by RRF score descending
- [ ] All methods must have same query count — raises `ValueError` if mismatched
- [ ] Handles 2+ methods (not just pairs)

### Step 2: Implement `fuse_all()`

**What:** Orchestration function that groups rankings by method and granularity, then applies fusion per config.

**Acceptance criteria:**
- [ ] Groups `all_rankings` by `(method, granularity)` key
- [ ] For each fusion config, extracts rankings for the specified methods at the specified granularity
- [ ] Calls `fuse_rankings()` with extracted rankings
- [ ] Wraps results in S0.1 format with `method=config["name"]`, `granularity=config["granularity"]`
- [ ] Skips fusion config if any required method is missing for that granularity (logs warning, does not fail)
- [ ] Returns flat list of all fused ranking results

### Step 3: Add `fusion` config section

**What:** Add fusion configuration to `config.yaml`.

**Acceptance criteria:**
- [ ] `fusion` section with `enabled: false`, `k: 60`, `configs` list
- [ ] At least one config: `fusion-bge-large-bm25` combining dense + sparse on `role` granularity
- [ ] Each config has `name`, `methods` (list of method strings), `granularity`

## Integration Notes

**Primary use case:** Fusing `bge-large` (dense embedding) with `bm25` (sparse baseline) on `role` granularity. The existing `run_all_baselines()` already produces BM25 rankings on `role` granularity — these serve as direct fusion inputs.

**Fusion happens after all individual rankings are collected** — both embedding model runs and baseline runs must complete first. The orchestration (C9) collects all results, then calls `fuse_all()`.

**Cross-encoder + RRF are additive, not redundant:** Cross-encoder improves individual ranking quality (spec 04). RRF combines rankings from multiple methods. A reranked embedding + BM25 fusion is a valid (and potentially the strongest) configuration.

## Testing Strategy

- Unit test for `fuse_rankings()`:
  - Two methods with overlapping candidates: verify RRF score computation
  - Two methods with disjoint candidates: verify correct handling
  - Single method: verify output equals input ranking (RRF of 1 list)
  - Mismatched query counts: verify ValueError
- Unit test for `fuse_all()`:
  - Mock rankings, verify grouping and fusion
  - Missing method: verify skip with warning
- Integration: run fusion on real embedding + BM25 rankings, verify output format
