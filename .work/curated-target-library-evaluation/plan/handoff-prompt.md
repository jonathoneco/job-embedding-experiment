# Handoff: plan -> implement

## What This Step Produced

- Architecture document at `.work/curated-target-library-evaluation/specs/architecture.md`
- Feature summary at `docs/feature/curated-target-library-evaluation.md`
- 6 design decisions, 6 components

## Architecture Summary

The curated evaluation adds a parallel evaluation path alongside the existing full-taxonomy pipeline. A hand-maintained JSON file (`data/taxonomy/curated_roles.json`) lists the production-representative role subset. A new module (`src/curated_targets.py`) filters the full taxonomy data to produce curated target sets at three granularity levels: `curated_role`, `curated_cluster`, `curated_category`. The experiment runner gains a config-gated curated block that feeds these targets through the existing embed/rank/evaluate modules unchanged. Test cases with no valid match in the curated set are excluded from metrics and tracked in a coverage report.

## Key Design Decisions

1. **Curated list = flat JSON of role name strings** validated against `roles.json`. Not a filter predicate, not a separate taxonomy file.
2. **Three granularities only** (`curated_role`, `curated_cluster`, `curated_category`). Descriptions skipped to limit scope.
3. **Clusters filtered, not rebuilt.** Each cluster keeps its label; only its `roles` list is narrowed. Empty clusters are dropped.
4. **Test cases with zero curated matches are excluded** from metrics, tracked in `curated_coverage.json`.
5. **Separate output files** (`curated_rankings.json`, `curated_metrics.json`, `curated_coverage.json`) -- not merged with full-taxonomy outputs.
6. **Config-driven** via `target_library.curated` in `config.yaml`.

## Implementation Instructions

### Order of work

1. **Create `data/taxonomy/curated_roles.json`** -- Start with a placeholder. The user will provide or refine the actual list. Include validation that all names exist in `roles.json`.

2. **Create `src/curated_targets.py`** with three functions:
   - `load_curated_roles(path, all_roles)` -- load JSON, validate, return `set[str]`
   - `build_curated_target_sets(curated_roles, roles, clusters)` -- filter and return `dict[str, list[dict]]` with keys `curated_role`, `curated_cluster`, `curated_category`
   - `filter_covered_test_cases(test_cases, curated_roles)` -- return `(covered_cases, coverage_report_dict)`

3. **Add config section** to `config.yaml`:
   ```yaml
   target_library:
     curated:
       enabled: true
       roles_file: "data/taxonomy/curated_roles.json"
       granularities: ["curated_role", "curated_cluster", "curated_category"]
   ```

4. **Extend `scripts/run_experiment.py`** -- After the existing full-taxonomy block, add curated evaluation:
   - Load curated roles and build curated targets
   - Filter test cases for coverage
   - Run embedding models (reuse `run_embedding_model`)
   - Run baselines (reuse `run_all_baselines`)
   - Evaluate and save to `curated_*.json` output files

5. **Extend `src/report.py`** -- Add a curated comparison section to the report.

6. **Write tests** for `src/curated_targets.py` -- validation, filtering edge cases, coverage tracking.

### Critical constraints

- **Do not modify `src/embed.py`, `src/baselines.py`, or `src/evaluate.py`.** These are target-set agnostic by design. The curated pipeline feeds them different targets through the same interface.
- Curated target IDs must use a distinct prefix (`TC-role-`, `TC-clust-`, `TC-cat-`) to avoid collision with full-taxonomy IDs in the evaluation target lookup.
- The embedding cache keys on `(model_label, granularity)`. Curated granularities have distinct names, so caching works automatically.
- Every role name in the curated JSON must validate against `roles.json` -- hard error on unknown names.

### Files to create

- `data/taxonomy/curated_roles.json` (placeholder)
- `src/curated_targets.py`
- `src/curated_targets_test.py`

### Files to modify

- `config.yaml` (add `target_library` section)
- `scripts/run_experiment.py` (add curated evaluation block)
- `src/report.py` (add comparison section)
