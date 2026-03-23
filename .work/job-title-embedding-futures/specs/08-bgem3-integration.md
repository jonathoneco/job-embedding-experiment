# Spec 08: BGE-M3 Integration (C8)

**Technique:** T8 — BGE-M3 Full Fusion
**Phase:** 4 (Evaluation/Fallback) — only pursued if Phase 3 doesn't hit MRR 0.75
**Scope:** Moderate (~200 lines + new dependency)
**References:** [architecture.md](architecture.md) C8, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

Integration of BGE-M3 model with all 3 modalities (dense + sparse + ColBERT). Requires the `FlagEmbedding` library (separate from sentence-transformers). Each modality produces its own ranking; fused via RRF from spec 05.

**Gate condition:** Only pursue if cumulative MRR < 0.75 after Phase 3 evaluation.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/bgem3.py` | New module |
| Modify | `config.yaml` | Add BGE-M3 model config |
| Modify | `requirements.txt` or equivalent | Add `FlagEmbedding` dependency |

## Interface Contract

### Exposes (src/bgem3.py)

```python
from FlagEmbedding import BGEM3FlagModel

def load_bgem3(config: dict) -> BGEM3FlagModel:
    """Load BGE-M3 model with all 3 modalities enabled."""

def encode_bgem3(
    model: BGEM3FlagModel,
    texts: list[str],
    batch_size: int = 64,
) -> dict[str, Any]:
    """Encode texts with BGE-M3, returning all 3 representations.

    Returns: {
        "dense": np.ndarray,          # (n, 1024) dense embeddings
        "sparse": list[dict],         # [{token_id: weight}, ...] sparse vectors
        "colbert": list[np.ndarray],  # [np.ndarray(n_tokens, dim), ...] token embeddings
    }
    """

def rank_bgem3_dense(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using dense embeddings only. Same interface as rank_targets()."""

def rank_bgem3_sparse(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using sparse (lexical) representations."""

def rank_bgem3_colbert(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using ColBERT late interaction."""

def run_bgem3(
    config: dict,
    target_sets: dict,
    test_cases: list[dict],
) -> list[dict]:
    """Run BGE-M3 with all 3 modalities on specified granularities.

    Returns uniform ranking results (S0.1 format) for each modality:
    - method="bgem3-dense"
    - method="bgem3-sparse"
    - method="bgem3-colbert"
    """
```

### Config schema

```yaml
bgem3:
  enabled: false                          # Gate: only enable if Phase 3 MRR < 0.75
  model: "BAAI/bge-m3"
  granularities: ["role", "category"]     # Which granularities to evaluate
```

### Consumes

- `FlagEmbedding` library (new dependency)
- Target sets from `build_target_sets()`
- Test cases

## Implementation Steps

### Step 1: Implement `load_bgem3()`

**What:** Load BGE-M3 with all modalities enabled.

**Acceptance criteria:**
- [ ] Uses `BGEM3FlagModel(model_name, use_fp16=True)` for GPU efficiency
- [ ] Model name from `config["bgem3"]["model"]`
- [ ] All 3 modalities enabled (dense, sparse, ColBERT)

### Step 2: Implement `encode_bgem3()`

**What:** Encode texts and return all 3 representations.

**Acceptance criteria:**
- [ ] Calls `model.encode(texts, batch_size=batch_size, return_dense=True, return_sparse=True, return_colbert_vecs=True)`
- [ ] Returns dict with `dense`, `sparse`, `colbert` keys
- [ ] Dense embeddings are L2-normalized

### Step 3: Implement ranking functions

**What:** Three separate ranking functions, one per modality.

**Acceptance criteria:**
- [ ] `rank_bgem3_dense()`: dot product ranking (same as `rank_targets()`)
- [ ] `rank_bgem3_sparse()`: sparse dot product (iterate over token overlaps)
- [ ] `rank_bgem3_colbert()`: MaxSim — for each query token, find max similarity to any target token, then average across query tokens
- [ ] All return same format: `[[{target_id, score}, ...], ...]`

### Step 4: Implement `run_bgem3()` orchestration

**What:** Encode targets and queries, run all 3 modalities, return results.

**Acceptance criteria:**
- [ ] Loads model once, reuses for all granularities
- [ ] For each configured granularity: encode targets, encode queries, rank with all 3 modalities
- [ ] Returns uniform ranking results with methods `bgem3-dense`, `bgem3-sparse`, `bgem3-colbert`
- [ ] Results feed into RRF fusion (spec 05) as `fusion-bgem3-all`

### Step 5: Config and dependency

**What:** Add BGE-M3 config section. Add `FlagEmbedding` to dependencies.

**Acceptance criteria:**
- [ ] `bgem3` config section with `enabled: false`
- [ ] `FlagEmbedding` added to project dependencies
- [ ] When `bgem3.enabled: false`, no FlagEmbedding import occurs (lazy import to avoid dependency error if not installed)

## Integration Notes

**BGE-M3 results + RRF:** The 3 modality rankings feed into `fuse_all()` from spec 05 as:
```yaml
fusion:
  configs:
    - name: "fusion-bgem3-all"
      methods: ["bgem3-dense", "bgem3-sparse", "bgem3-colbert"]
      granularity: "role"
```

**Dense-only BGE-M3 is NOT worth it** — must use all 3 modalities to justify the FlagEmbedding dependency. If only dense is needed, bge-large (via sentence-transformers) is simpler.

**FlagEmbedding API is different from sentence-transformers.** This module is self-contained — it does NOT use `encode_targets()`/`encode_queries()` from `embed.py`. It has its own encode and rank functions.

## Testing Strategy

- Unit test: verify encoding returns all 3 representation types
- Unit test: verify each ranking function produces correct output format
- Integration: run full BGE-M3 pipeline on small subset, verify all 3 modality results
- Integration: feed BGE-M3 results into RRF fusion, verify fused output
