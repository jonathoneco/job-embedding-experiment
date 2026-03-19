# Spec 05 — Evaluation & Reporting (C4)

**Dependencies**: C2 (test cases), C3 (ranking results)
**Refs**: Spec 00 (metrics result schema, error analysis schema, config, method labels)

---

## Overview

Compute evaluation metrics from ranking results, run statistical tests, generate a markdown report with embedded charts. All metrics are computed on the 650-case test set only. Dev set results are reported as a sanity check.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/evaluate.py` | Metric computation (MRR, Top-K, category accuracy) |
| `src/statistics.py` | Bootstrap CIs, McNemar's test, Friedman + Nemenyi |
| `src/report.py` | Markdown report with embedded charts |
| `scripts/run_experiment.py` | Main orchestrator: models → baselines → evaluate → report |
| `tests/test_evaluate.py` | Unit tests for metric computation |

---

## Subcomponent: Metrics Engine (`src/evaluate.py`)

### Function: `compute_metrics(rankings: list[dict], test_cases: list[dict], target_sets: dict, top_k_values: list[int]) -> dict`

Compute all metrics for one method × granularity combination.

**Metric definitions**:

**MRR (Mean Reciprocal Rank)**:
```python
for each query:
    reciprocal_rank = 0
    for rank, result in enumerate(ranked_results, 1):
        target = targets_by_id[result["target_id"]]
        if is_correct(target, test_case):
            reciprocal_rank = 1.0 / rank
            break
    mrr_sum += reciprocal_rank
mrr = mrr_sum / n_queries
```

**is_correct logic**: A target is correct if its `role` matches any entry in the test case's `correct_roles`. For cluster/category targets, correct if the target's `roles` list contains any role from `correct_roles`, OR if the target's `category` matches any category in `correct_roles`.

**Top-K accuracy**: Fraction of queries where at least one correct target appears in the top K results.

**Category accuracy**: Fraction of queries where the rank-1 target's category matches any category in `correct_roles`. Applicable at all granularity levels (every target has a `category` field).

**Mean similarity gap**: Average of (rank-1 score minus rank-2 score) across all queries. Measures confidence/decisiveness of the ranking.

### Function: `compute_by_difficulty(rankings: list[dict], test_cases: list[dict], target_sets: dict, top_k_values: list[int]) -> dict`

Group test cases by difficulty and compute metrics per group. Returns `{"easy": {...}, "medium": {...}, "hard": {...}}`.

### Function: `evaluate_all(all_rankings: list[dict], test_cases: list[dict], target_sets: dict, config: dict) -> list[dict]`

Compute metrics for all method × granularity combinations.

**Process**:
1. Group rankings by (method, granularity)
2. For each group, compute full metrics + by-difficulty breakdown
3. Return list of metrics result dicts (spec 00 schema)

**Acceptance criteria**:
- MRR values in [0, 1]
- Top-K accuracy values in [0, 1] and monotonically non-decreasing (top1 <= top3 <= top5)
- Category accuracy in [0, 1]
- Similarity gap can be negative (but typically small positive)
- Produces metrics for: 3 models × 5 granularities + 3 baselines × 2 granularities = 21 configurations

---

## Subcomponent: Statistical Analysis (`src/statistics.py`)

### Function: `bootstrap_ci(rankings: list[dict], test_cases: list[dict], target_sets: dict, metric_fn: callable, n_resamples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]`

Compute bootstrap confidence interval for a metric.

```python
rng = np.random.default_rng(seed)
scores = []
for _ in range(n_resamples):
    indices = rng.choice(len(test_cases), size=len(test_cases), replace=True)
    resampled_cases = [test_cases[i] for i in indices]
    resampled_rankings = [rankings[i] for i in indices]
    score = metric_fn(resampled_rankings, resampled_cases)
    scores.append(score)
return (np.percentile(scores, 100 * alpha / 2),
        np.percentile(scores, 100 * (1 - alpha / 2)))
```

Compute CIs for MRR, top1, top3, top5 for every configuration.

### Function: `mcnemar_test(rankings_a: list[dict], rankings_b: list[dict], test_cases: list[dict], target_sets: dict) -> dict`

Pairwise comparison: do two methods produce statistically different top-1 accuracy?

```python
# For each test case, classify as (A correct, B correct):
# (1,1), (1,0), (0,1), (0,0)
# McNemar's test compares (1,0) vs (0,1) counts

from scipy.stats import binom_test  # or use exact McNemar's

contingency = [[n_11, n_10], [n_01, n_00]]
# Use exact binomial test on discordant pairs
p_value = ...
```

Note: Use `scipy.stats` for the test. If scipy is not in dependencies, implement the exact binomial test manually (it's trivial: compare n_10 vs n_01 with a two-sided binomial test).

**Returns**: `{"method_a": str, "method_b": str, "p_value": float, "significant": bool}` where `significant` = p_value < 0.05.

### Function: `run_statistical_tests(all_rankings: list[dict], test_cases: list[dict], target_sets: dict, config: dict) -> dict`

Run bootstrap CIs and pairwise McNemar's tests.

**Pairwise comparisons**: All pairs of the 3 embedding models at each granularity (3 choose 2 = 3 pairs × 5 granularities = 15 tests). Plus best embedding model vs each baseline at `role` and `category` granularities (3 baselines × 2 = 6 tests). Total: 21 pairwise tests.

**Bonferroni correction**: Adjust significance threshold to 0.05 / 21 ≈ 0.0024 for the full family of tests.

**Returns**: `{"bootstrap_cis": list[dict], "mcnemar_tests": list[dict], "bonferroni_threshold": float}`

Output saved to `results/metrics/significance.json`.

**Acceptance criteria**:
- Bootstrap CIs computed for all 21 configurations × 4 metrics (MRR, top1, top3, top5)
- McNemar's tests for all 21 pairs
- Bonferroni correction applied
- Friedman + Nemenyi: include only if 3+ models are compared at the same granularity (which they are at all 5 levels). Use `scipy.stats.friedmanchisquare` and a post-hoc Nemenyi test. If scipy is unavailable, skip Friedman and note it in the report.

---

## Subcomponent: Report Generator (`src/report.py`)

### Function: `generate_report(metrics: list[dict], significance: dict, test_cases: list[dict], all_rankings: list[dict], target_sets: dict, config: dict) -> str`

Generate the full markdown report.

### Report Structure

```markdown
# Job Title Embedding Matching — Experiment Report

## Executive Summary
- Best model + granularity combination and its MRR
- Whether production thresholds are met (MRR >= 0.75, Top-3 >= 0.85)
- Recommendation: which configuration to deploy

## Methodology
- Taxonomy: 692 roles, 42 categories
- Test data: 750 cases (250 easy, 325 medium, 175 hard)
- Models: 3 embedding models + 3 baselines
- Granularity levels: 5 (with descriptions of each)

## Results

### Overall Performance
- Table: model × granularity → MRR (with 95% CI)
- Heatmap figure: `heatmap_mrr_by_model_granularity.png`

### Top-K Accuracy
- Bar chart: grouped by model, bars for top-1/3/5 at best granularity
- Figure: `bar_topk_by_model.png`

### Baseline Comparison
- Table: baselines vs best embedding model at role and category granularity
- Bar chart: `bar_baseline_comparison.png`

### Difficulty Breakdown
- Table: MRR by difficulty tier for best model at best granularity
- Grouped bar chart: `bar_mrr_by_difficulty.png`

### Category Accuracy
- Table: category accuracy for best configuration
- Note whether category-level matching is sufficient for coarse routing

### Statistical Significance
- Table: McNemar's p-values for key comparisons
- Note which differences are significant after Bonferroni correction

## Error Analysis
- Table: 10 worst failures for best model at best granularity
- Each row: input_title, expected role, rank-1 prediction, score, correct_rank,
  failure_mode
- Pattern summary: most common failure modes

## Production Threshold Check
- Table: MRR and Top-3 for all configurations vs thresholds
- Pass/fail indicator per configuration

## Appendix
- Full metrics tables for all 21 configurations
- Dev set results (sanity check)
```

### Chart Generation

Use matplotlib + seaborn. Style: `seaborn-v0_8-whitegrid` (or `whitegrid`). DPI: 150. Size: 10×6 inches for most charts, 8×8 for heatmap.

**Heatmap** (`heatmap_mrr_by_model_granularity.png`):
- X-axis: granularity levels
- Y-axis: methods (3 embedding + 3 baselines where applicable)
- Color: MRR value
- Annotate cells with MRR values (2 decimal places)
- Use `seaborn.heatmap` with `annot=True, fmt=".2f", cmap="YlOrRd"`

**Bar charts**: Use `seaborn.barplot` or `matplotlib.pyplot.bar` with grouped bars. Error bars from bootstrap CIs where available.

**Error analysis table**: Extract 10 test cases with lowest reciprocal rank (i.e., correct answer ranked furthest from #1) for the best model at the best granularity. Format as spec 00 error analysis schema. Classify failure mode based on whether rank-1 is in the same category or not.

**Figures saved to**: `results/figures/`

### Function: `write_report(report_md: str, output_path: str)`

Write the markdown string to the output file.

**Acceptance criteria**:
- Report is valid markdown with no broken image links
- All figures referenced in the report exist in `results/figures/`
- Heatmap covers all 21 method × granularity cells (with N/A for baseline × non-applicable granularity)
- Error analysis shows exactly 10 cases
- Executive summary includes a clear recommendation
- Production threshold check is pass/fail per configuration

---

## Orchestrator Integration (`scripts/run_experiment.py`)

The evaluation module is called from the main orchestrator after matching:

```python
# After C3 produces all_rankings:
metrics = evaluate_all(all_rankings, test_cases, target_sets, config)
save_json(metrics, "results/metrics/all_metrics.json")

significance = run_statistical_tests(all_rankings, test_cases, target_sets, config)
save_json(significance, "results/metrics/significance.json")

report = generate_report(metrics, significance, test_cases, all_rankings, target_sets, config)
write_report(report, config["report"]["output"])
```

---

## Testing Strategy (`tests/test_evaluate.py`)

1. **Test MRR computation**:
   - 3 queries where correct answer is at rank 1, 2, 3 → MRR = (1 + 0.5 + 0.333) / 3 ≈ 0.611
   - Query with no correct answer in top 10 → reciprocal rank = 0

2. **Test Top-K accuracy**:
   - 4 queries, correct at ranks 1, 3, 6, 1 → top1 = 0.5, top3 = 0.75, top5 = 0.75

3. **Test category accuracy**:
   - Correct when rank-1 category matches, incorrect otherwise
   - Test with accept sets (multiple correct categories)

4. **Test is_correct with accept sets**:
   - Test case with `correct_roles: [A, B]` — target matching A is correct, target matching C is not

5. **Test similarity gap**:
   - Known scores [0.9, 0.7, ...] → gap = 0.2

6. **Test bootstrap CI**:
   - Feed deterministic data, verify CI bounds are within expected range
   - Verify seed produces reproducible results

7. **No tests for report.py**: Report generation is visual. Validated by manual inspection during implementation.
