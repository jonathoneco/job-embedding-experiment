# Stream D — Evaluation & Reporting

**Items**: W-07 (JEM-04t), W-08 (JEM-zqj) — sequential
**Spec**: 05-evaluation-reporting.md
**Dependencies**: W-04 (matching pipeline) + W-06 (test data) must both be complete

---

## W-07: Metrics Engine + Statistical Analysis + Tests (JEM-04t)

**Depends on**: W-04 (JEM-3ft), W-06 (JEM-g2w)

**Files to create**:
- `src/evaluate.py` — Metric computation (MRR, Top-K, category accuracy, similarity gap)
- `src/statistics.py` — Bootstrap CIs, McNemar's test, Friedman + Nemenyi
- `tests/test_evaluate.py` — Unit tests for metric computation

**evaluate.py functions**:
- `compute_metrics(rankings, test_cases, target_sets, top_k_values) -> dict`
  - MRR: Mean reciprocal rank (1/rank of first correct target, 0 if none in top 10)
  - Top-K accuracy: fraction with correct target in top K
  - Category accuracy: fraction where rank-1 target's category matches
  - Mean similarity gap: average of (rank-1 score - rank-2 score)
  - `is_correct` logic: match on `role` for role/role_desc targets, match on `roles` list or `category` for cluster/category targets
- `compute_by_difficulty(rankings, test_cases, target_sets, top_k_values) -> dict` — Metrics per difficulty group
- `evaluate_all(all_rankings, test_cases, target_sets, config) -> list[dict]` — All method × granularity combos (21 configurations: 3 models × 5 + 3 baselines × 2)

**statistics.py functions**:
- `bootstrap_ci(rankings, test_cases, target_sets, metric_fn, n_resamples, seed, alpha=0.05) -> tuple[float, float]`
  - Resample with replacement, compute metric per resample, return percentile CI
  - CIs for MRR, top1, top3, top5 for all 21 configs
- `mcnemar_test(rankings_a, rankings_b, test_cases, target_sets) -> dict`
  - Pairwise top-1 comparison, exact binomial test on discordant pairs
  - Returns `{"method_a", "method_b", "p_value", "significant"}`
- `run_statistical_tests(all_rankings, test_cases, target_sets, config) -> dict`
  - 21 pairwise McNemar's tests (3C2 × 5 + best vs 3 baselines × 2)
  - Bonferroni correction: α = 0.05 / 21 ≈ 0.0024
  - Friedman + Nemenyi if scipy available
  - Output saved to `results/metrics/significance.json`

**Tests (test_evaluate.py)**:
1. MRR computation: 3 queries with correct at rank 1, 2, 3 → MRR ≈ 0.611
2. Top-K accuracy: 4 queries, correct at ranks 1, 3, 6, 1 → top1=0.5, top3=0.75, top5=0.75
3. Category accuracy: correct when rank-1 category matches
4. is_correct with accept sets: correct_roles [A, B], target A matches, target C doesn't
5. Similarity gap: known scores [0.9, 0.7, ...] → gap = 0.2
6. Bootstrap CI: deterministic data, verify bounds in expected range, seed reproducibility

**Acceptance criteria**:
- MRR values in [0, 1]
- Top-K accuracy monotonically non-decreasing (top1 ≤ top3 ≤ top5)
- All tests pass
- Bootstrap CIs computed for all 21 × 4 metrics
- McNemar's for all 21 pairs with Bonferroni

---

## W-08: Report Generator + Main Orchestrator (JEM-zqj)

**Depends on**: W-07

**Files to create**:
- `src/report.py` — Markdown report with embedded charts
- `scripts/run_experiment.py` — Main orchestrator: models → baselines → evaluate → report

**report.py functions**:
- `generate_report(metrics, significance, test_cases, all_rankings, target_sets, config) -> str`
  - Full markdown report per spec 05 structure:
    - Executive Summary (best config, threshold check, recommendation)
    - Methodology (taxonomy, test data, models, granularity descriptions)
    - Overall Performance (MRR heatmap)
    - Top-K Accuracy (grouped bar chart)
    - Baseline Comparison (table + bar chart)
    - Difficulty Breakdown (grouped bar chart)
    - Category Accuracy
    - Statistical Significance (McNemar's p-values table)
    - Error Analysis (10 worst failures for best model)
    - Production Threshold Check (pass/fail per config)
    - Appendix (full metrics, dev set results)
- `write_report(report_md, output_path)` — Write to file

**Chart generation**:
- Style: `seaborn-v0_8-whitegrid`, DPI 150, 10×6 or 8×8
- `heatmap_mrr_by_model_granularity.png` — seaborn.heatmap, annot=True, fmt=".2f", cmap="YlOrRd"
- `bar_topk_by_model.png` — Grouped bars for top-1/3/5
- `bar_baseline_comparison.png` — Best embedding vs baselines
- `bar_mrr_by_difficulty.png` — MRR by difficulty tier
- Figures saved to `results/figures/`

**Error analysis**: 10 test cases with lowest reciprocal rank for best model at best granularity. Classify failure mode: near-miss, category-confusion, surface-mismatch, rank-displacement, ambiguity.

**run_experiment.py orchestrator**:
```python
# Load data
config = yaml.safe_load(open("config.yaml"))
target_sets = load_json("data/taxonomy/target_sets.json")
test_cases = load_json("data/test-cases/test.json")
dev_cases = load_json("data/test-cases/dev.json")

# Run models
all_rankings = []
for model_config in config["models"]:
    rankings = run_embedding_model(model_config, target_sets, test_cases, config)
    all_rankings.extend(rankings)

# Run baselines
baseline_rankings = run_all_baselines(target_sets, test_cases)
all_rankings.extend(baseline_rankings)

# Evaluate
metrics = evaluate_all(all_rankings, test_cases, target_sets, config)
save_json(metrics, "results/metrics/all_metrics.json")

significance = run_statistical_tests(all_rankings, test_cases, target_sets, config)
save_json(significance, "results/metrics/significance.json")

# Report
report = generate_report(metrics, significance, test_cases, all_rankings, target_sets, config)
write_report(report, config["report"]["output"])
```

**Acceptance criteria**:
- Report is valid markdown with no broken image links
- All figures exist in `results/figures/`
- Heatmap covers all 21 method × granularity cells
- Error analysis shows exactly 10 cases
- Executive summary includes recommendation
- Production threshold check has pass/fail per config

**Note**: The orchestrator's embedding model calls require GPU (garden-pop). The orchestrator can be written and tested with mock data locally; actual execution runs on garden-pop.

**Beads workflow**:
```bash
bd update JEM-04t --status=in_progress  # W-07
# ... implement ...
bd close JEM-04t
bd update JEM-zqj --status=in_progress  # W-08
# ... implement ...
bd close JEM-zqj
```
