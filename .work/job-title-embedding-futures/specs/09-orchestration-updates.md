# Spec 09: Orchestration Updates (C9)

**Technique:** Cross-cutting pipeline integration
**Phase:** All phases
**Scope:** Small (~50 lines config changes + ~30 lines code changes)
**References:** [architecture.md](architecture.md) C9, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

Updates to `config.yaml`, `src/embed.py`, and the main experiment runner to support new methods, granularities, and pipeline stages. All changes are additive — existing 21 configs remain unchanged when new features are disabled.

## Files

| Action | File | Changes |
|--------|------|---------|
| Modify | `config.yaml` | Add new sections (reranking, fusion, bgem3), model entries |
| Modify | `src/embed.py` | Extend granularity loop, pass instruction, support reranking flow |
| Modify | `src/evaluate.py` | Add `role_augmented` to `_is_correct()` (also in spec 03) |

## Implementation Steps — Phased

### Phase 1 Changes (with C1 + C2)

#### Step 1: Extend `encode_queries()` for instruction prefix

Already specified in spec 02. This step is listed here for completeness of the orchestration view.

**Acceptance criteria:** See spec 02, step 2.

#### Step 2: Call `expand_abbreviations()` before encoding

Already specified in spec 01. Listed here for orchestration completeness.

**Acceptance criteria:** See spec 01, step 3.

### Phase 2 Changes (with C3 + C4 + C5)

#### Step 3: Extend granularity loop in `run_embedding_model()`

**What:** Add `role_augmented` to the hardcoded granularity list in `embed.py:run_embedding_model()`.

**Current code (embed.py ~line 131):**
```python
for granularity in ["role", "role_desc", "cluster", "category_desc", "category"]:
```

**New code:**
```python
for granularity in ["role", "role_desc", "cluster", "category_desc", "category", "role_augmented"]:
    targets = target_sets.get(granularity)
    if targets is None:
        continue  # role_augmented may not exist if augmentation hasn't been run
```

**Acceptance criteria:**
- [ ] `role_augmented` added to loop
- [ ] Uses `.get()` with `None` check instead of direct key access
- [ ] Skips missing granularities gracefully (no error if augmented targets don't exist)
- [ ] Existing 5 granularities still process normally

#### Step 4: Integrate reranking pipeline

**What:** After `rank_targets()`, optionally run cross-encoder reranking when `config["reranking"]["enabled"]` is true.

**Acceptance criteria:**
- [ ] Check `config.get("reranking", {}).get("enabled", False)`
- [ ] When enabled, pass `top_k=config["reranking"]["initial_k"]` to `rank_targets()`
- [ ] After ranking, enrich candidates with `text` field from target dicts (needed by reranker)
- [ ] Call `rerank_batch()` from spec 04
- [ ] Create separate result entries with method `{model_label}+rerank`
- [ ] Original (non-reranked) results also kept (both methods in output)

#### Step 5: Orchestrate baseline rankings as fusion inputs

**What:** Ensure baseline rankings from `run_all_baselines()` are available in the flat results list before fusion runs.

**Acceptance criteria:**
- [ ] `run_all_baselines()` results included in `all_rankings` before `fuse_all()` is called
- [ ] Baseline method names (`tfidf`, `fuzzy`, `bm25`) match what fusion configs reference

#### Step 6: Call `fuse_all()` after all rankings collected

**What:** When `config["fusion"]["enabled"]` is true, run RRF fusion on collected rankings.

**Acceptance criteria:**
- [ ] All embedding model rankings + baseline rankings collected into `all_rankings`
- [ ] `fuse_all(all_rankings, config["fusion"]["configs"], test_cases, k=config["fusion"]["k"])` called
- [ ] Fused results appended to `all_rankings`
- [ ] Fusion happens AFTER all individual methods complete (not interleaved)

### Phase 3 Changes (with C6 + C7)

#### Step 7: Support local model paths in `load_model()`

**What:** Handle `revision: null` for local fine-tuned models.

**Current `load_model()`:**
```python
def load_model(model_config: dict) -> SentenceTransformer:
    return SentenceTransformer(
        model_config["id"],
        revision=model_config["revision"],
        trust_remote_code=False,
    )
```

**New:**
```python
def load_model(model_config: dict) -> SentenceTransformer:
    kwargs = {"trust_remote_code": False}
    if model_config.get("revision"):
        kwargs["revision"] = model_config["revision"]
    return SentenceTransformer(model_config["id"], **kwargs)
```

**Acceptance criteria:**
- [ ] `revision=None` or missing `revision` key: `SentenceTransformer()` called without `revision` kwarg
- [ ] Non-null revision: passed as before
- [ ] Local model path (`models/bge-large-finetuned`) loads correctly

### Phase 4 Changes (with C8)

#### Step 8: BGE-M3 pipeline integration

**What:** When `config["bgem3"]["enabled"]` is true, run BGE-M3 and add results + fusion.

**Acceptance criteria:**
- [ ] Check `config.get("bgem3", {}).get("enabled", False)`
- [ ] When enabled, call `run_bgem3()` from spec 08
- [ ] Add 3 modality results to `all_rankings`
- [ ] Add `fusion-bgem3-all` fusion config if not already present
- [ ] Re-run `fuse_all()` to include BGE-M3 modality fusion

## Config Changes Summary

```yaml
# Existing (unchanged)
experiment: { ... }
taxonomy: { ... }
models: [ ... ]                     # Add instruction field to BGE models
embedding: { ... }
test_data: { ... }
generation: { ... }
evaluation: { ... }
report: { ... }

# New sections (all default disabled)
reranking:
  enabled: false
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  initial_k: 20
  top_n: 10

fusion:
  enabled: false
  k: 60
  configs:
    - name: "fusion-bge-large-bm25"
      methods: ["bge-large", "bm25"]
      granularity: "role"

bgem3:
  enabled: false
  model: "BAAI/bge-m3"
  granularities: ["role", "category"]
```

## Backward Compatibility

When all new sections default to `enabled: false` (or are absent):
- Granularity loop skips `role_augmented` (target set doesn't exist)
- No reranking pipeline triggered
- No fusion pipeline triggered
- No BGE-M3 pipeline triggered
- `encode_queries()` gets `prompt=None` for non-BGE models
- Abbreviation expansion is always active (transparent, no config toggle)
- Existing 21 configs produce identical results

## Testing Strategy

- Run full experiment with all new features disabled: verify 21 configs produce same results as before
- Run with reranking enabled: verify additional `+rerank` method results appear
- Run with fusion enabled: verify fusion results appear (requires baselines to have run)
- Run with role_augmented targets: verify additional granularity in results
- Config validation: verify missing optional sections don't cause errors
