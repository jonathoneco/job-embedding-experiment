# Archive Summary: job-title-embedding-experiment

**Tier:** 3
**Duration:** 2026-03-18 -> 2026-03-19
**Sessions:** ~4
**Beads epic:** JEM-ms0

## What Was Built
End-to-end experiment validating whether direct embedding comparison can match blurry/user-defined job titles against a maintained taxonomy of 692 roles across 42 categories. Tested 3 embedding models (MiniLM, BGE-base, BGE-large) at 5 granularity levels against 3 baselines (TF-IDF, Jaro-Winkler, BM25) on 567 test cases. Includes full statistical analysis (bootstrap CIs, McNemar's, Friedman+Nemenyi) and automated report generation.

## Key Results
- **Best config:** bge-large @ role — MRR 0.732, Top-3 0.795
- **Production thresholds NOT met** (MRR 0.75, Top-3 0.85) but margins are small
- Embeddings significantly outperform all baselines (+18% MRR over best baseline)
- Primary failure modes: abbreviation blindness, cross-category confusion
- Future improvements identified: instruction prefixing, abbreviation expansion, target augmentation, cross-encoder re-ranking

## Key Files
- `src/taxonomy.py` — Taxonomy parser (692 roles, 42 categories)
- `src/clusters.py` — Hardcoded subclusters (90 clusters)
- `src/descriptions.py` — Claude API role description generation
- `src/targets.py` — 5-granularity target set builder
- `src/embed.py` — Embedding model loading, encoding, ranking
- `src/baselines.py` — TF-IDF, Jaro-Winkler, BM25 baselines
- `src/generate_rules.py` — Rule-based test case generation (120 cases)
- `src/generate_llm.py` — LLM-based test case generation (3-pass, ~550 cases)
- `src/validate.py` — Validation, word-level Jaccard dedup, stratified split
- `src/evaluate.py` — MRR, Top-K, category accuracy, similarity gap
- `src/statistics.py` — Bootstrap CI, McNemar's, Friedman+Nemenyi
- `src/report.py` — 11-section markdown report with 4 charts
- `src/utils.py` — Shared load_json/save_json
- `scripts/prep_taxonomy.py` — Taxonomy preparation orchestrator
- `scripts/prep_test_data.py` — Test data preparation orchestrator
- `scripts/run_experiment.py` — Main experiment orchestrator
- `results/report.md` — Generated experiment report
- `data/test-cases/manual.json` — 75 hand-curated test cases

## Findings Summary
- 27 total findings (27 fixed, 0 deferred)
- 3 critical (McNemar p-value, NaN normalization, float assertion)
- 10 important (schema validation, category consistency, Friedman+Nemenyi, etc.)
- 14 suggestions (code quality, maintainability)

## Runtime Fixes (not in original review)
- TF-IDF score floating-point clipping (same class of bug as embed.py assertion)
- Word-level Jaccard dedup replacing char-set Jaccard (was removing 40% of valid cases)
- LLM case filtering for hallucinated roles + category auto-fix from taxonomy
- max_tokens increased 4096→16384 for LLM response truncation
- stop_reason checking added to detect truncated LLM responses
- dev_size increased 100→150 to accommodate stratification strata count

## Futures Promoted
See docs/futures/job-title-embedding-experiment.md — 8 improvement techniques identified.
