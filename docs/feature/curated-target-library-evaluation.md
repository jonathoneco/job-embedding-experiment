# Curated Target Library Evaluation

**Issue:** JEM-e0c | **Tier:** 2 (Feature) | **Status:** In Progress

## Summary

Run existing embedding experiments against a scoped-down "match-to" role library representing what would actually be stored in production. Evaluate at multiple granularity levels (specific roles, variant parents/clusters, departments/categories) to measure performance against a realistic target set.

## Motivation

The full taxonomy (692 roles, 42 categories, 90 clusters) includes overlap and breadth that may not reflect production usage. A curated subset representing the actual stored library provides a more realistic performance benchmark.

## What

A parallel evaluation path that runs all existing models and baselines against a hand-curated subset of the role taxonomy. The curated library is a flat JSON list of role names validated against the full taxonomy. Target sets are derived by filtering roles, clusters, and categories to only include curated members. Test cases with no valid curated match are excluded from metrics and tracked in a coverage report. Results are written to separate output files for side-by-side comparison with full-taxonomy metrics.

## Components

| ID | Component | Scope | Key Files |
|----|-----------|-------|-----------|
| C1 | Curated role list | New data | `data/taxonomy/curated_roles.json` |
| C2 | Curated target builder | New module | `src/curated_targets.py` |
| C3 | Config extension | Modify | `config.yaml` |
| C4 | Experiment runner | Modify | `scripts/run_experiment.py` |
| C5 | Coverage tracker | Part of C2 | `src/curated_targets.py` |
| C6 | Report extension | Modify | `src/report.py` |

## Key Decisions

1. **Curated library format:** Flat JSON list of role name strings, validated against `roles.json`. Chosen over filter predicates or separate taxonomy files for simplicity.
2. **Three granularities:** `curated_role`, `curated_cluster`, `curated_category`. Descriptions skipped to limit scope.
3. **Cluster filtering:** Clusters keep their labels; only their `roles` lists are narrowed to curated members. Empty clusters are dropped.
4. **Test case exclusion:** Cases with no curated match are excluded from metrics, tracked separately in a coverage report.
5. **Separate outputs:** Curated results go to `curated_rankings.json`, `curated_metrics.json`, `curated_coverage.json` -- not merged with full-taxonomy files.
6. **No modifications to core modules:** `src/embed.py`, `src/baselines.py`, and `src/evaluate.py` are target-set agnostic and require no changes.

## Key Files

- `data/taxonomy/curated_roles.json` -- the curated role list (new)
- `src/curated_targets.py` -- build curated targets, filter test cases (new)
- `config.yaml` -- `target_library.curated` config section (modify)
- `scripts/run_experiment.py` -- curated evaluation block (modify)
- `src/report.py` -- comparison section (modify)
- `.work/curated-target-library-evaluation/specs/architecture.md` -- full architecture
