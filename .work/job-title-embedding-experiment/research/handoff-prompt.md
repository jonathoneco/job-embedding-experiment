# Research Handoff → Plan

## What This Step Produced

Three research notes covering embedding model selection, evaluation methodology, and test data generation strategy. All indexed in `research/index.md`. Futures captured in `futures.md`.

## Key Findings

### Taxonomy
- **692 roles** across **42 categories** (source: `job-roles.md`)

### Embedding Models (3 tiers, all local on 4080)
- **Small**: `all-MiniLM-L6-v2` — 22.7M params, 384-dim, STS ~82
- **Medium**: `BAAI/bge-base-en-v1.5` — 109M params, 768-dim, STS ~85.4
- **Large**: `BAAI/bge-large-en-v1.5` — 335M params, 1024-dim, STS ~87.1
- All use cosine similarity (normalize + dot product). No input prefixes needed.
- Entire experiment completes in seconds per model. Throughput is not a concern.

### Granularity Levels (5 levels)
1. **Role** (~692 targets) — role name only
2. **Role+Description** (~692) — role name + one-line description
3. **Cluster** (~80-120) — category-scaffold subclusters with labels
4. **Category+Description** (~42) — category name + key terms
5. **Category** (~42) — category name only

The "enriched description" variants (Role+Desc, Cat+Desc) are cheap to add and often decisive. They test whether richer text helps at the same granularity.

### Evaluation
- **Primary metric**: MRR (Mean Reciprocal Rank)
- **Supporting**: Top-1, Top-3, Top-5 accuracy, Category accuracy, Mean similarity gap
- **Baselines**: TF-IDF (char n-grams), fuzzy matching (Jaro-Winkler), BM25
- **Statistical tests**: McNemar's (pairwise), bootstrap CIs (1000 resamples)
- **Production thresholds**: MRR >= 0.75, Top-3 >= 0.85

### Test Data
- **750 cases**: 250 easy, 325 medium, 175 hard (no impossible tier — system always returns best match)
- **Generation**: LLM (structured multi-pass) + rule-based transforms + manual curation
- **Format**: JSON with multi-label support (accept sets)
- **Split**: 100 dev / 650 test
- **Coverage**: ~18 cases per category (750/42, approximately uniform)
- **Validation**: 100% programmatic (schema, dedup, taxonomy membership) + 13% stratified spot-check (~100 cases)

### Pinned Data Generation Allocation

| Method | Easy | Medium | Hard | Total |
|--------|------|--------|------|-------|
| Rule-based transforms | 120 | — | — | 120 |
| LLM systematic (Pass 1) | 105 | 200 | — | 305 |
| LLM adversarial (Pass 2-3) | — | 125 | 125 | 250 |
| Manual curation | — | — | 50 | 50 |
| O*NET synonyms | 25 | — | — | 25 |
| **Total** | **250** | **325** | **175** | **750** |

Overflow strategy: if LLM generates more than needed for a tier, trim by removing the most similar cases (deduplicate by cosine similarity on input titles).

## Decisions Made
1. Using `sentence-transformers` library for all models (consistent API)
2. Cosine similarity as the metric (all models trained for it)
3. 5 granularity levels instead of 3 (adding enriched description variants)
4. Including non-embedding baselines (TF-IDF, fuzzy, BM25) to strengthen validity argument
5. Accept-set ground truth to handle multi-label ambiguity
6. Computing on garden-pop (4080 GPU) via SSH/Tailscale
7. **Cluster construction**: Category scaffold approach — split categories with 10+ roles into 2-4 subclusters, keep small categories intact. Manual labels. No embedding-based clustering (avoids bootstrap problem).
8. **No impossible tier**: System always returns best match, no rejection scenario. 75 cases redistributed across easy/medium/hard.
9. **Validation**: Programmatic + 13% stratified spot-check. No inter-annotator agreement process.

## Resolved Advisory Items
1. ✅ Taxonomy counts corrected to 692 roles / 42 categories
2. ✅ Cluster construction: category scaffold with manual subclusters
3. ✅ No impossible cases — always best match
4. ✅ Lighter validation: programmatic + 13% spot-check
5. ✅ Generation allocations pinned to exact 750

## Open Questions for Planning
1. **Python project structure**: How to organize scripts, data, results? Single script vs modular?
2. **Role descriptions**: Where to source one-line descriptions for Role+Desc? Generate with LLM? Manual?
3. **Result format**: Jupyter notebook? HTML report? Markdown with embedded charts?
4. **Reproducibility**: How to pin model versions, random seeds, data versions?

## Artifacts
- `research/01-embedding-models.md` — model selection rationale
- `research/02-evaluation-methodology.md` — metrics, statistics, presentation
- `research/03-data-generation.md` — data generation strategy and schema
- `research/index.md` — topic index
- `futures.md` — deferred enhancements

## Instructions for Plan Step
Read this handoff prompt as your primary input. Design the architecture: Python project structure, data pipeline, embedding pipeline, evaluation pipeline, reporting. Address the open questions above. The existing `job-roles.md` in the project root is the taxonomy source (692 roles, 42 categories).
