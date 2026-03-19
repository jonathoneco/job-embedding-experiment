# Stream B — Matching Pipeline

**Items**: W-04 (JEM-3ft)
**Spec**: 04-matching-pipeline.md
**Dependencies**: W-01 (project setup) must be complete
**Parallel with**: Stream A Phase 2 (taxonomy)

---

## W-04: Embedding Engine + Baselines (JEM-3ft)

**Files to create**:
- `src/embed.py` — Model loading, encoding with caching, cosine similarity ranking
- `src/baselines.py` — TF-IDF, Jaro-Winkler, BM25 baseline methods

**embed.py functions**:
- `load_model(model_config: dict) -> SentenceTransformer` — Load by ID + revision hash
- `encode_targets(model, targets, batch_size, cache_dir, model_label) -> np.ndarray` — Encode + cache to `{cache_dir}/{model_label}/targets_{granularity}.npy`
- `encode_queries(model, queries, batch_size) -> np.ndarray` — Encode queries (no cache)
- `rank_targets(query_embeddings, target_embeddings, targets, top_k=10) -> list[list[dict]]` — Cosine similarity via dot product of L2-normalized vectors
- `run_embedding_model(model_config, target_sets, test_cases, config) -> list[dict]` — End-to-end per model: load → encode targets × 5 granularities → encode queries → rank → produce ranking results

**Key implementation details**:
- L2-normalize all embeddings after encoding (cosine sim = dot product)
- Cache path: `{cache_dir}/{model_label}/targets_{granularity}.npy`
- Check cache shape matches before loading
- Top 10 results per query, sorted descending by score
- Ranking result schema: `{"test_case_id", "method", "granularity", "ranked_results": [{"target_id", "score"}]}`

**baselines.py functions**:
- `run_tfidf(targets, test_cases, granularity) -> list[dict]` — Character n-gram (3,5) TF-IDF + cosine similarity
- `run_fuzzy(targets, test_cases, granularity) -> list[dict]` — Jaro-Winkler via rapidfuzz (divide by 100 to normalize)
- `run_bm25(targets, test_cases, granularity) -> list[dict]` — BM25Okapi, normalize scores by dividing by max per query
- `run_all_baselines(target_sets, test_cases) -> list[dict]` — Run all 3 baselines on `role` and `category` granularities only

**Acceptance criteria**:
- Embedding model produces n_queries × 5 ranking results (one per granularity)
- Each ranking has exactly 10 targets, scores in [0, 1], sorted descending
- Cache files created on first run, reused on subsequent
- Baselines produce n_queries × 6 results (3 methods × 2 granularities)
- Baselines only run on `role` and `category` (NOT role_desc, cluster, category_desc)
- All scores normalized to [0, 1]

**No unit tests**: Validated via integration testing in the orchestrator (dev set sanity check) and inline assertions.

**Note**: Code can be written and linted without target_sets.json existing. Functional testing requires Phase 2 Stream A (W-03) to complete first.

**Beads workflow**:
```bash
bd update JEM-3ft --status=in_progress
# ... implement ...
bd close JEM-3ft
```
