# Architecture: Curated Target Library Evaluation

## Problem Statement

The current evaluation pipeline runs against the full taxonomy: 692 roles, 90 clusters, 42 categories. This taxonomy was designed for breadth and completeness, not as a production artifact. It includes overlapping roles (e.g., "HR Senior Service Manager" vs. "HR Service Manager"), Microsoft-specific variants ("Digital Sales Specialist at Microsoft"), and niche specializations that a production system would likely consolidate.

Evaluating against the full set measures retrieval quality in an idealized world, but production will use a smaller, curated library -- the actual set of canonical roles an organization stores and maps incoming job titles to. Performance may differ significantly when the target space shrinks: fewer distractors should improve top-K accuracy, but removing roles means some test cases may lose their correct match entirely, and cluster/category boundaries shift.

We need to understand how the models perform against the realistic target set, not just the comprehensive one.

## Goals

1. **Define a curated role library** as a hand-maintained JSON file containing the production-representative subset of roles.
2. **Build curated target sets** at three granularity levels (role, cluster, category) derived from the curated library.
3. **Run all existing experiments** (3 embedding models + 3 baselines) against curated targets using the same pipeline.
4. **Produce comparable metrics** -- same schema, same test cases (with coverage tracking), same report format -- so full-taxonomy and curated results sit side by side.
5. **Track test case coverage** -- identify which test cases have no valid match in the curated set and report coverage alongside accuracy.

### Non-Goals

- **Defining the "right" curated set.** The initial curated list is a judgment call by the user. The system supports any subset; curation criteria are outside scope.
- **New evaluation metrics.** We reuse MRR, Top-K, category accuracy, per-difficulty breakdowns. No new metric types.
- **Automated curation tools.** No deduplication algorithms, clustering-based selection, or coverage optimization. The list is hand-picked.
- **Modifying the full-taxonomy pipeline.** All existing results remain reproducible. Curated evaluation is additive.
- **Fine-tuning or retraining on curated targets.** Embedding models are frozen; we only change what they match against.

## Design Decisions

### DD-1: Curated library as a hand-maintained JSON file

**Decision:** The curated library is a new file `data/taxonomy/curated_roles.json` -- a flat list of role names (strings) that are a subset of the 692 roles in `roles.json`.

**Rationale:**
- A filter predicate (e.g., tags on roles) would require modifying the taxonomy format and adding metadata that doesn't exist yet.
- A separate markdown taxonomy file would duplicate structure and diverge from the canonical `roles.json`.
- A simple list of role name strings is the lightest-weight approach. It's easy to hand-edit, diff, and validate against the existing roles.

**Format:**
```json
["Benefits Administrator", "HR Business Partner", "HR Generalist", ...]
```

**Validation:** At load time, every name in the curated list must exist in `roles.json`. Unknown names are a hard error, not a warning.

### DD-2: Curated target sets derived by filtering, not rebuilding

**Decision:** Curated target sets are built by filtering the full taxonomy data, not by creating a separate taxonomy source.

**How it works:**
1. Load `curated_roles.json` as a set of role names.
2. Filter `roles.json` to only curated roles -> curated role targets.
3. Filter `clusters.json`: keep each cluster, but restrict its `roles` list to curated members. Drop clusters with zero curated members. Rename the target set to preserve the cluster label as-is (the cluster name is the embedding text, so it stays).
4. Derive categories from the filtered roles (same logic as `get_categories`). Categories with zero curated roles are dropped.

**Rationale:** Clusters and categories are groupings of roles. When roles are removed, the groupings shrink but their labels remain meaningful. A cluster like "HR Strategy & Business Partnership" still makes sense even if 3 of its 8 roles are removed -- the embedding text is the cluster label, and the `roles` list is only used for correctness checking.

### DD-3: No curated role_desc or category_desc granularities

**Decision:** Curated evaluation uses three granularity levels: `curated_role`, `curated_cluster`, `curated_category`. We skip `role_desc` and `category_desc`.

**Rationale:** The primary question is "does performance change with a smaller target set?" The role, cluster, and category granularities cover the three useful levels of specificity. Adding descriptions would double the scope for marginal insight. If needed later, it's a trivial extension since the filtering logic is the same.

### DD-4: Test case coverage tracking with exclusion

**Decision:** Test cases whose `correct_roles` have zero overlap with the curated set are excluded from curated evaluation metrics but tracked separately.

**How it works:**
1. For each test case, check if any `correct_roles[].role` is in the curated set.
2. If yes: include in evaluation.
3. If no: exclude, record in a `coverage_report` alongside metrics.
4. Report includes: total test cases, covered count, excluded count, excluded case IDs.

**Rationale:** Including test cases with no valid answer would artificially deflate metrics -- MRR=0 for cases that *can't* succeed. Excluding them and reporting coverage gives an honest accuracy number plus a clear signal about how much the curated set misses. This is more useful than a single blended number.

### DD-5: Config-driven target set selection

**Decision:** Add a `target_library` key to `config.yaml` that selects which target sets to evaluate against. The experiment runner loads target sets accordingly.

```yaml
target_library:
  curated:
    enabled: true
    roles_file: "data/taxonomy/curated_roles.json"
    granularities: ["curated_role", "curated_cluster", "curated_category"]
```

**Rationale:** Keeps the experiment runner generic. The curated path is opt-in and parallel to the existing full-taxonomy path. Both can run in the same experiment, producing separate metric files.

### DD-6: Separate output files, not merged metrics

**Decision:** Curated evaluation writes to separate output files: `curated_rankings.json`, `curated_metrics.json`, `curated_coverage.json`. Full-taxonomy outputs are unchanged.

**Rationale:** Merging curated and full metrics into the same files would complicate both the output schema and downstream report generation. Separate files are simpler and let the report explicitly compare the two.

## Component Map

| ID | Component | Scope | Files | Dependencies |
|----|-----------|-------|-------|--------------|
| C1 | Curated role list | New data file | `data/taxonomy/curated_roles.json` | None |
| C2 | Curated target builder | New module | `src/curated_targets.py` | `data/taxonomy/roles.json`, `clusters.json`, `curated_roles.json` |
| C3 | Config extension | Modify | `config.yaml` | None |
| C4 | Experiment runner | Modify | `scripts/run_experiment.py` | C2, C3 |
| C5 | Coverage tracker | New in C2 | `src/curated_targets.py` | Test cases, curated role set |
| C6 | Report extension | Modify | `src/report.py` | C4 output files |

## Data Flow

```
curated_roles.json ──┐
roles.json ──────────┤
clusters.json ───────┤
                     ▼
           build_curated_target_sets()
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
   curated_role  curated_cluster  curated_category
     targets       targets          targets
          │          │          │
          └──────────┼──────────┘
                     ▼
        ┌─── embed + rank (same as full pipeline) ───┐
        │   Models: minilm, bge-base, bge-large      │
        │   Baselines: tfidf, fuzzy, bm25             │
        └─────────────────────────────────────────────┘
                     │
                     ▼
          filter_covered_test_cases()
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    covered rankings      coverage_report
          │                     │
          ▼                     ▼
    compute_metrics()    curated_coverage.json
          │
          ▼
    curated_metrics.json
          │
          ▼
    report (comparison table: full vs curated)
```

## Implementation Approach

### Step 1: Create curated role list (C1)

Create `data/taxonomy/curated_roles.json` with an initial curated subset. The user will hand-edit this to finalize the production-representative set. Start with a placeholder containing ~200 roles as a starting point (or the user provides the list).

### Step 2: Build curated target module (C2)

Create `src/curated_targets.py` with:

```python
def load_curated_roles(path: str) -> set[str]
    # Load and validate against roles.json

def build_curated_target_sets(
    curated_roles: set[str],
    roles: list[dict],
    clusters: list[dict],
) -> dict[str, list[dict]]
    # Filter to curated_role, curated_cluster, curated_category

def filter_covered_test_cases(
    test_cases: list[dict],
    curated_roles: set[str],
) -> tuple[list[dict], dict]
    # Returns (covered_cases, coverage_report)
```

Target IDs use prefix `TC-` to distinguish from full-taxonomy targets: `TC-role-NNNN`, `TC-clust-NNNN`, `TC-cat-NNNN`.

### Step 3: Extend config (C3)

Add `target_library.curated` section to `config.yaml` (DD-5).

### Step 4: Extend experiment runner (C4)

After the existing full-taxonomy evaluation, add a curated block:

1. Load curated roles and build curated target sets.
2. Filter test cases for coverage.
3. Run embedding models against curated targets (reuses `run_embedding_model` with curated `target_sets`).
4. Run baselines against curated targets.
5. Evaluate and save to separate output files.

The existing `run_embedding_model` and `run_all_baselines` already accept `target_sets` as a parameter -- they iterate over whatever granularities are present. No changes needed to the embed/baseline modules, only the orchestrator.

### Step 5: Extend report (C6)

Add a "Curated Library Comparison" section to the report showing:
- Full-taxonomy vs. curated metrics side by side per model.
- Coverage summary.
- Per-difficulty breakdown for curated.

## Scope Exclusions

- **No changes to `src/embed.py`, `src/baselines.py`, or `src/evaluate.py`.** These modules are target-set agnostic. They receive targets and produce rankings/metrics. The curated pipeline feeds them different targets through the same interface.
- **No changes to test case generation.** Test cases are fixed. Coverage tracking handles the mismatch.
- **No curated-specific embedding cache.** The embedding cache keys on `(model_label, granularity)`. Since curated granularities have distinct names (`curated_role` vs `role`), caching works automatically with no changes.
- **No statistical significance tests for curated vs. full.** This is a descriptive comparison, not a hypothesis test. Different test case populations make paired tests invalid.
- **No UI or interactive curation tool.** The curated list is a flat JSON file edited by hand.
