# Spec 04 — Matching Pipeline (C3)

**Dependencies**: C0 (config), C1 (target sets)
**Refs**: Spec 00 (ranking result schema, method labels, config)

---

## Overview

Load embedding models, encode targets and queries, compute similarity rankings. Also implement 3 non-embedding baselines for comparison. All methods produce the same ranking result format (spec 00).

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/embed.py` | Embedding model loading, encoding, similarity ranking |
| `src/baselines.py` | TF-IDF, Jaro-Winkler, BM25 baseline methods |

---

## Subcomponent: Embedding Engine (`src/embed.py`)

### Function: `load_model(model_config: dict) -> SentenceTransformer`

Load a sentence-transformers model by ID and revision hash from config.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_config["id"], revision=model_config["revision"])
```

### Function: `encode_targets(model: SentenceTransformer, targets: list[dict], batch_size: int, cache_dir: str, model_label: str) -> np.ndarray`

Encode target texts into embeddings. Cache to disk.

**Caching strategy**:
- Cache path: `{cache_dir}/{model_label}/targets_{granularity}.npy`
- If cache file exists with matching shape, load from cache
- Otherwise, encode and save to cache
- Embeddings are L2-normalized after encoding (cosine similarity = dot product)

**Acceptance criteria**:
- Returns numpy array of shape `(n_targets, dim)`
- All vectors are L2-normalized (norm ≈ 1.0)
- Cache files are created on first run
- Subsequent runs load from cache without model inference

### Function: `encode_queries(model: SentenceTransformer, queries: list[str], batch_size: int) -> np.ndarray`

Encode query input titles. No caching (queries are small).

Returns numpy array of shape `(n_queries, dim)`, L2-normalized.

### Function: `rank_targets(query_embeddings: np.ndarray, target_embeddings: np.ndarray, targets: list[dict], top_k: int = 10) -> list[list[dict]]`

Compute cosine similarity matrix and extract top-k ranked targets per query.

```python
# Since both are L2-normalized, cosine similarity = dot product
similarity_matrix = query_embeddings @ target_embeddings.T  # (n_queries, n_targets)

# For each query, argsort descending and take top_k
```

**Returns**: List (per query) of lists of `{"target_id": str, "score": float}`, sorted descending by score. Top 10 results per query.

### Function: `run_embedding_model(model_config: dict, target_sets: dict, test_cases: list[dict], config: dict) -> list[dict]`

End-to-end: load model → encode targets (per granularity) → encode queries → rank → produce ranking results.

**Process**:
1. Load model
2. Extract query texts: `[case["input_title"] for case in test_cases]`
3. For each granularity in `["role", "role_desc", "cluster", "category_desc", "category"]`:
   a. Encode targets (with caching)
   b. Encode queries
   c. Rank targets
   d. Produce ranking result dicts (spec 00 schema)

**Returns**: List of ranking result dicts across all granularities.

**Acceptance criteria**:
- Produces `n_queries × 5` ranking results (one per query per granularity)
- Each result has exactly 10 ranked targets
- Scores are in [0, 1] range (cosine similarity of normalized vectors)
- Cache files are reused across granularities that share the same model

---

## Subcomponent: Baselines (`src/baselines.py`)

Baselines run against **`role` and `category` granularities only** (2 of 5). Description-enriched and cluster targets don't meaningfully help text-matching baselines.

### Function: `run_tfidf(targets: list[dict], test_cases: list[dict], granularity: str) -> list[dict]`

Character n-gram TF-IDF similarity.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
target_texts = [t["text"] for t in targets]
query_texts = [c["input_title"] for c in test_cases]

target_matrix = vectorizer.fit_transform(target_texts)
query_matrix = vectorizer.transform(query_texts)

similarity = cosine_similarity(query_matrix, target_matrix)
```

Extract top-10 per query. Return ranking result dicts with `method="tfidf"`.

### Function: `run_fuzzy(targets: list[dict], test_cases: list[dict], granularity: str) -> list[dict]`

Jaro-Winkler similarity via rapidfuzz.

```python
from rapidfuzz import fuzz

for query in queries:
    scores = [fuzz.jaro_winkler_similarity(query, t["text"]) / 100.0 for t in targets]
```

Extract top-10 per query. Return ranking result dicts with `method="fuzzy"`.

Note: Jaro-Winkler scores are in [0, 100] from rapidfuzz — divide by 100 to normalize to [0, 1].

### Function: `run_bm25(targets: list[dict], test_cases: list[dict], granularity: str) -> list[dict]`

BM25 ranking on word-tokenized texts.

```python
from rank_bm25 import BM25Okapi

tokenized_targets = [t["text"].lower().split() for t in targets]
bm25 = BM25Okapi(tokenized_targets)

for query in queries:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
```

Extract top-10 per query. Normalize scores to [0, 1] range by dividing by max score (per query).

Return ranking result dicts with `method="bm25"`.

### Function: `run_all_baselines(target_sets: dict, test_cases: list[dict]) -> list[dict]`

Run all 3 baselines against `role` and `category` target sets.

```python
results = []
for granularity in ["role", "category"]:
    targets = target_sets[granularity]
    results.extend(run_tfidf(targets, test_cases, granularity))
    results.extend(run_fuzzy(targets, test_cases, granularity))
    results.extend(run_bm25(targets, test_cases, granularity))
return results
```

**Returns**: `n_queries × 6` ranking results (3 baselines × 2 granularities).

**Acceptance criteria**:
- Produces exactly 6 ranking result sets per query
- Baselines only run on `role` and `category` (NOT `role_desc`, `cluster`, `category_desc`)
- All scores normalized to [0, 1]
- Each result has 10 ranked targets (or fewer if target set is smaller)

---

## Interface Contract

**Consumes**:
- `data/taxonomy/target_sets.json` (from C1)
- `data/test-cases/test.json` and `data/test-cases/dev.json` (from C2)
- `config.yaml` (model configs, batch sizes, cache dir)

**Exposes**:
- `run_embedding_model(model_config, target_sets, test_cases, config) -> list[dict]`
- `run_all_baselines(target_sets, test_cases) -> list[dict]`
- Both return lists of ranking result dicts (spec 00 schema)

**Output**: Combined ranking results are aggregated by the orchestrator (`scripts/run_experiment.py`) and passed to the evaluation module.

---

## Testing Strategy

No unit tests for C3 modules — they wrap third-party libraries with straightforward usage. Validation happens via:

1. **Integration testing in orchestrator**: The `run_experiment.py` script validates ranking result shapes and score ranges before passing to evaluation.
2. **Dev set sanity check**: Run the full pipeline on the 100-case dev set first. Verify that at least the trivial cases (exact matches) rank correctly.
3. **Score range assertions**: Inline assertions that scores are in [0, 1] and results are sorted descending.
