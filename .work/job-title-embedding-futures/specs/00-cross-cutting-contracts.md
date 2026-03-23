# Spec 00: Cross-Cutting Contracts

Shared schemas, interfaces, and conventions consumed by all component specs.

**References:** [architecture.md](architecture.md)

---

## S0.1 Uniform Ranking Result

Every ranking method (embedding, baseline, reranker, fusion) produces results in this format:

```python
{
    "test_case_id": str,         # e.g. "TC-0001"
    "method": str,               # e.g. "bge-large", "tfidf", "bge-large+rerank", "fusion-dense-sparse"
    "granularity": str,          # e.g. "role", "role_augmented", "category"
    "ranked_results": [
        {"target_id": str, "score": float},  # descending by score
        ...
    ]
}
```

**Constraints:**
- `ranked_results` sorted descending by `score`
- Scores clipped to valid range for the method (embedding: [-1.0, 1.0]; baselines: [0.0, 1.0]; cross-encoder: [0.0, 1.0]; RRF: [0.0, +inf))
- `method` string uniquely identifies the ranking pipeline (model + optional stages)

## S0.2 Method Naming Convention

| Pipeline | Method String | Example |
|----------|--------------|---------|
| Base embedding | `{model_label}` | `bge-large` |
| Embedding + instruction | `{model_label}` (instruction is transparent) | `bge-large` |
| Embedding + reranking | `{model_label}+rerank` | `bge-large+rerank` |
| Baseline | `{baseline_method}` | `tfidf`, `fuzzy`, `bm25` |
| RRF fusion | `fusion-{component1}-{component2}` | `fusion-bge-large-bm25` |
| Fine-tuned model | `{model_label}` | `bge-large-ft` |
| BGE-M3 modality | `bgem3-{modality}` | `bgem3-dense`, `bgem3-sparse`, `bgem3-colbert` |
| BGE-M3 fused | `fusion-bgem3-all` | `fusion-bgem3-all` |

**Note:** Instruction prefixing does NOT change the method name — it's a transparent query encoding improvement, not a separate ranking pipeline.

## S0.3 Target Dict Schemas

All targets have a common base plus granularity-specific fields:

```python
# Common fields
{
    "id": str,              # "T-{prefix}-{i:04d}"
    "text": str,            # text to encode
    "category": str,        # category name
    "granularity": str,     # granularity level name
}
```

### Per-granularity extensions

| Granularity | ID Prefix | Extra Fields | Eval Match Field |
|-------------|-----------|-------------|-----------------|
| `role` | `T-role-` | `role: str` | `target["role"]` |
| `role_desc` | `T-rdesc-` | `role: str` | `target["role"]` |
| `role_augmented` | `T-raug-` | `role: str` | `target["role"]` |
| `cluster` | `T-clust-` | `roles: list[str]`, `cluster_label: str` | `target["roles"]` |
| `category_desc` | `T-cdesc-` | `roles: list[str]` | `target["roles"]` |
| `category` | `T-cat-` | `roles: list[str]` | `target["roles"]` |

**Key:** `role_augmented` uses singular `role` field (same as `role`/`role_desc`), mapping each augmented alias back to its parent role name. This ensures `_is_correct()` works without modification — it already checks `target["role"]` for granularities matching `"role"` or `"role_desc"`.

### Evaluation compatibility

`_is_correct()` dispatches on granularity string:
- `granularity in ("role", "role_desc")` → checks `target["role"]`
- Otherwise → checks `target["roles"]`

**For `role_augmented`:** Add `"role_augmented"` to the first branch so it uses `target["role"]`. This is a 1-line change in `src/evaluate.py:_is_correct()`.

## S0.4 Test Case Schema

```python
{
    "id": str,                    # "TC-{i:04d}"
    "input_title": str,           # query text
    "correct_roles": [
        {"role": str, "category": str},
        ...
    ],
    "difficulty": str,            # "easy", "medium", "hard"
    "variation_type": str,
    "source": str,
    "notes": str,
}
```

Unchanged. All new techniques consume this schema as-is.

## S0.5 Config Schema Extensions

New top-level sections added to `config.yaml`:

```yaml
models:
  - id: "BAAI/bge-large-en-v1.5"
    revision: "d4aa6901..."
    dim: 1024
    label: "bge-large"
    instruction: "Represent this sentence for searching relevant passages: "  # NEW: query-side instruction

reranking:                          # NEW section
  enabled: false                    # default off; enable per experiment run
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  initial_k: 20                     # bi-encoder returns top-20 for reranker input
  top_n: 10                         # reranker returns top-10

fusion:                             # NEW section
  enabled: false
  k: 60                             # RRF constant
  configs:
    - name: "fusion-bge-large-bm25"
      methods: ["bge-large", "bm25"]
      granularity: "role"
```

**Backward compatibility:** All new sections default to disabled. Existing configs produce identical results.

## S0.6 File Path Conventions

| Path | Purpose | Component |
|------|---------|-----------|
| `src/preprocess.py` | Query preprocessing (abbreviation expansion) | C1 |
| `src/augment.py` | Target augmentation (LLM alias generation) | C3 |
| `src/rerank.py` | Cross-encoder reranking | C4 |
| `src/fusion.py` | RRF score fusion | C5 |
| `src/generate_training_data.py` | Training data generation | C6 |
| `src/fine_tune.py` | Model fine-tuning | C7 |
| `src/bgem3.py` | BGE-M3 integration | C8 |
| `data/taxonomy/augmented_targets.json` | Augmented role targets | C3 |
| `data/training/pairs.jsonl` | Contrastive training pairs | C6 |
| `data/training/corpus.txt` | TSDAE corpus | C6 |
| `models/bge-large-finetuned/` | Fine-tuned model weights | C7 |

## S0.7 Granularity Registry

Complete list of granularity levels after all phases:

| Level | Target Count | Source | Phase |
|-------|-------------|--------|-------|
| `role` | 692 | taxonomy | existing |
| `role_desc` | 692 | taxonomy | existing |
| `cluster` | ~96 | taxonomy | existing |
| `category_desc` | 42 | taxonomy | existing |
| `category` | 42 | taxonomy | existing |
| `role_augmented` | ~3,500-7,000 | LLM-generated | Phase 2 (C3) |

## S0.8 Error Handling Convention

All new modules follow fail-closed behavior:
- Missing config fields → `KeyError` (not silent defaults)
- Invalid model/file paths → raise, don't fall back
- Encoding failures → propagate exception
- Empty results → return empty list (not fabricated data)
