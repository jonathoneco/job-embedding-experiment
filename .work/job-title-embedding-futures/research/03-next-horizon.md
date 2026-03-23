# Next Horizon Techniques (5 Quick Wins)

## 1. Instruction Prefixing
**Complexity: Trivial (2-3 lines)**
- BGE v1.5 supports query-side instruction: `"Represent this sentence for searching relevant passages: "`
- sentence-transformers `model.encode()` accepts `prompt` kwarg
- Add to `encode_queries()` in embed.py, parameterize via config.yaml per model
- Target-side: keep un-prefixed (symmetric short text case)
- Expected: +2-5% MRR
- **Invalidates all target embedding caches** if applied to targets too

## 2. Abbreviation Dictionary Expansion
**Complexity: Simple (new module)**
- Current: 14 abbreviations in generate_rules.py (used for test generation, not query normalization)
- Need: Query-time expansion before embedding — new `src/normalize.py`
- Failing abbreviations from error analysis: TAM, Mktg, Autom, CS, Pharm, Reg, BizDev, HR, IT
- Extended dict: ~22-26 entries (14 existing + 8-12 from error analysis)
- Direction is reversed: existing dict maps full→abbrev (for test gen), need abbrev→full (for query expansion)
- Expected: +2-5% MRR, directly fixes 3-4 of top-10 failure cases

## 3. Target-Side Augmentation
**Complexity: Moderate (new granularity level)**
- LLM-generate 5-10 aliases per taxonomy role offline
- Add new granularity `role_aliases` to targets.py (~3.5K-7K targets vs 692 for role)
- Reuse descriptions.py Claude API infrastructure for generation
- Evaluation: _is_correct() maps alias back to canonical role
- Cost: ~17 API calls (same batching as current descriptions)
- Cache: new `targets_role_aliases.npy` per model
- Expected: +5-20% recall improvement

## 4. Cross-Encoder Re-Ranking
**Complexity: Moderate (new pipeline stage)**
- Two-stage: bi-encoder top-20 → CrossEncoder re-scores pairs → return top-10
- `from sentence_transformers import CrossEncoder` — already in library
- Model: `BAAI/bge-reranker-base` (~200MB VRAM, negligible on 4080)
- New function in embed.py: `rerank_with_cross_encoder()`
- Config: `reranking.enabled`, `reranking.model_id`, `reranking.initial_k=20`
- Must modify rank_targets to return top-20 when reranking enabled
- Expected: +5-15% ranking, primarily on category confusion (top failure mode)

## 5. Hybrid Dense+Sparse Score Fusion (RRF)
**Complexity: Simple (new module, no deps)**
- Reciprocal Rank Fusion: `RRF(d) = Σ(1 / (k + rank(d, method_i)))`, k=60
- Rank-based, scale-agnostic — solves [-1,1] vs [0,1] score incompatibility
- New `src/fuse.py` module
- Best combination: bge-large (dense) + TF-IDF (sparse) — complementary strengths
- Requires orchestrator to collect both dense + sparse rankings before fusing
- Optional: method weights (e.g., dense 2.0, sparse 1.5)
- Expected: +3-8% MRR
