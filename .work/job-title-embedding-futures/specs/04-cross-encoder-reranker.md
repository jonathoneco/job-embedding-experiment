# Spec 04: Cross-Encoder Reranker (C4)

**Technique:** T4 — Cross-Encoder Re-Ranking
**Phase:** 2 (Pipeline Extensions)
**Scope:** Moderate (~120 lines)
**References:** [architecture.md](architecture.md) C4, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/rerank.py` module implementing a two-stage retrieval pipeline. Bi-encoder produces top-K candidates (K=20), then a cross-encoder rescores each (query, candidate) pair and returns top-N (N=10). Cross-encoder processes full text pairs jointly, providing much better ranking for close candidates.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/rerank.py` | New module |
| Modify | `config.yaml` | Add `reranking` section |

## Interface Contract

### Exposes (src/rerank.py)

```python
from sentence_transformers import CrossEncoder

def load_reranker(config: dict) -> CrossEncoder:
    """Load cross-encoder model from config['reranking']['model'].

    Model loaded once, reused across all queries.
    """

def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: list[dict],    # [{target_id, score, text}, ...] — top-K from bi-encoder
                               # NOTE: caller must enrich with "text" field from target dicts
                               # (S0.1 ranked_results only has target_id + score)
    top_n: int = 10,
) -> list[dict]:
    """Re-score candidates with cross-encoder and return top-N.

    Returns: [{"target_id": str, "score": float}, ...] sorted descending.
    Cross-encoder scores are in [0.0, 1.0] range (sigmoid output).
    """

def rerank_batch(
    reranker: CrossEncoder,
    queries: list[str],
    candidates_per_query: list[list[dict]],
    top_n: int = 10,
) -> list[list[dict]]:
    """Rerank candidates for multiple queries.

    Returns: list of reranked results per query (same structure as rank_targets output).
    """
```

### Consumes

- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Bi-encoder ranked results (top-K candidates with `target_id`, `score`)
- Target text: needs `text` field from target dicts to form (query, candidate_text) pairs

### Config schema

```yaml
reranking:
  enabled: false
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  initial_k: 20    # bi-encoder returns this many candidates
  top_n: 10         # reranker outputs this many
```

## Model Selection

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** chosen over `bge-reranker-base` because:
- 6x smaller (22M vs 137M params) — faster inference
- Well-benchmarked on MS MARCO
- Available via sentence-transformers `CrossEncoder` class (consistent API)
- For short text pairs (job titles, ~5 words each), the smaller model is sufficient
- If results are unsatisfactory, can swap to `bge-reranker-base` via config change

## Implementation Steps

### Step 1: Implement `load_reranker()`

**What:** Load `CrossEncoder` model from config.

**Acceptance criteria:**
- [ ] Uses `CrossEncoder(model_name)` from sentence-transformers
- [ ] Model name read from `config["reranking"]["model"]`
- [ ] Returns `CrossEncoder` instance

### Step 2: Implement `rerank()`

**What:** Score each (query, candidate_text) pair with cross-encoder, sort by score, return top-N.

**Acceptance criteria:**
- [ ] Forms pairs: `[(query, candidate["text"]) for candidate in candidates]`
- [ ] Calls `reranker.predict(pairs)` to get scores
- [ ] Applies sigmoid to raw logits if model outputs logits (ms-marco-MiniLM outputs logits)
- [ ] Sorts candidates by cross-encoder score descending
- [ ] Returns top-N as `[{"target_id": str, "score": float}]`
- [ ] Raises `ValueError` if `candidates` is empty

### Step 3: Implement `rerank_batch()`

**What:** Rerank candidates for all queries in a single batch.

**Acceptance criteria:**
- [ ] Iterates over (query, candidates) pairs, calling `rerank()` for each
- [ ] Returns list of reranked results, same length as input queries
- [ ] Handles queries with empty candidate lists (returns empty list for that query)

### Step 4: Add `reranking` config section

**What:** Add reranking configuration to `config.yaml`.

**Acceptance criteria:**
- [ ] `reranking` section present with `enabled: false`, `model`, `initial_k: 20`, `top_n: 10`
- [ ] When `enabled: false`, no reranking code path is triggered (config-gated)

## Integration Notes

**Caller responsibility (handled in C9 orchestration, not this spec):**
- When reranking enabled, `rank_targets()` must be called with `top_k=config["reranking"]["initial_k"]` (20 instead of default 10)
- After `rank_targets()`, call `rerank_batch()` with the results
- Reranked results get method name `{model_label}+rerank` (see S0.2)
- Target `text` field must be available — the caller passes it from `target_sets[granularity]`

**Cross-encoder input format:** `rerank()` receives candidates enriched with `text` field. The caller joins ranked results with target dicts by `target_id` to add the `text` field before passing to the reranker.

## Testing Strategy

- Unit test for `rerank()`: mock CrossEncoder, verify sorting and top-N selection
- Unit test for `rerank_batch()`: verify batch processing
- Unit test: verify empty candidates raises ValueError
- Integration: run reranker on real bi-encoder output, verify output format matches S0.1
