# Job Title Embedding Matching Experiment

**Status:** archived | **Tier:** 3 | **Beads:** JEM-ms0

## What
Verify the validity of using direct embedding comparison to match blurry/user-defined job titles against a maintained taxonomy of 692 roles across 42 categories. Compare 3 embedding model tiers across 5 granularity levels (15 configurations) plus 3 baselines, with statistical rigor, to determine the optimal approach for a recommendation system.

## Components
- **C0: Project Setup** — Dependencies, config, directory structure
- **C1: Taxonomy & Targets** — Parse taxonomy, build clusters, generate descriptions, construct 5-level target sets
- **C2: Test Data** — 750-case generation (rule-based + LLM + manual), validation, dev/test split
- **C3: Matching Pipeline** — 3 embedding models + 3 baselines, similarity computation
- **C4: Evaluation & Reporting** — Metrics (MRR, Top-K, category accuracy), bootstrap CIs, McNemar's test, markdown report with charts

## Key Decisions
- **Subclustering**: Category-scaffold — categories with 10+ roles split into 2-4 subclusters with descriptive labels (~96 total). Hardcoded definitions, not algorithmic.
- **LLM descriptions**: One-line functional descriptions (10-15 words) per role, keyword summaries per category. Generated via Claude API, committed to git.
- **Dev set**: 100 cases used for pipeline sanity checking and threshold exploration only — never for metric reporting.
- **Error analysis**: 10 worst failures per best model, classified by failure mode (near-miss, category-confusion, surface-mismatch, rank-displacement, ambiguity).
- **Config schema**: YAML with sections for experiment, taxonomy, models (with HuggingFace revision hashes), embedding, test_data, generation, evaluation, report.
- **Baselines scope**: TF-IDF, Jaro-Winkler, BM25 run against role and category granularities only (2 of 5).
- **Statistical rigor**: Bootstrap CIs (1000 resamples), McNemar's pairwise with Bonferroni correction (21 tests), Friedman + Nemenyi for multi-model ranking.

## Completed
**Archived:** 2026-03-19 | **Findings:** 27 (all fixed) | **Test cases:** 567

Best configuration bge-large@role achieved MRR 0.732 and Top-3 0.795 — significantly better than all baselines but short of production thresholds (MRR 0.75, Top-3 0.85). Future work identified in `docs/futures/job-title-embedding-experiment.md` covering instruction prefixing, abbreviation expansion, target augmentation, and cross-encoder re-ranking.
