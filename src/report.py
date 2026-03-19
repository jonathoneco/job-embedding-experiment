"""Report generator for job title embedding experiment results."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from src.evaluate import _is_correct


def _set_style():
    """Set matplotlib style, handling different seaborn versions."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # Fall back to default


def _plot_heatmap(metrics: list[dict], config: dict) -> str:
    """Generate MRR heatmap by model x granularity.

    Returns:
        Path to saved figure.
    """
    _set_style()
    figures_dir = config["report"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Collect methods and granularities
    methods = sorted({m["method"] for m in metrics})
    granularities = sorted({m["granularity"] for m in metrics})

    # Build MRR matrix
    mrr_matrix = np.zeros((len(methods), len(granularities)))
    for m in metrics:
        i = methods.index(m["method"])
        j = granularities.index(m["granularity"])
        mrr_matrix[i, j] = m["metrics"]["mrr"]

    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(
            mrr_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=granularities, yticklabels=methods, ax=ax,
        )
    else:
        im = ax.imshow(mrr_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(granularities)))
        ax.set_xticklabels(granularities)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        for i in range(len(methods)):
            for j in range(len(granularities)):
                ax.text(j, i, f"{mrr_matrix[i, j]:.2f}",
                        ha="center", va="center")
        fig.colorbar(im)

    ax.set_title("MRR by Model × Granularity")
    ax.set_xlabel("Granularity")
    ax.set_ylabel("Method")
    plt.tight_layout()

    path = os.path.join(figures_dir, "heatmap_mrr_by_model_granularity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_topk_bars(metrics: list[dict], config: dict) -> str:
    """Generate grouped bar chart for top-1/3/5 accuracy by model.

    Returns:
        Path to saved figure.
    """
    _set_style()
    figures_dir = config["report"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Use best granularity for each method (highest MRR)
    best_by_method: dict[str, dict] = {}
    for m in metrics:
        method = m["method"]
        if method not in best_by_method or m["metrics"]["mrr"] > best_by_method[method]["metrics"]["mrr"]:
            best_by_method[method] = m

    methods = sorted(best_by_method.keys())
    top1 = [best_by_method[m]["metrics"]["top1"] for m in methods]
    top3 = [best_by_method[m]["metrics"]["top3"] for m in methods]
    top5 = [best_by_method[m]["metrics"]["top5"] for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, top1, width, label="Top-1")
    ax.bar(x, top3, width, label="Top-3")
    ax.bar(x + width, top5, width, label="Top-5")

    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K Accuracy by Method (Best Granularity)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    path = os.path.join(figures_dir, "topk_accuracy_by_method.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_baseline_comparison(metrics: list[dict], config: dict) -> str:
    """Generate bar chart comparing best embedding model vs baselines.

    Returns:
        Path to saved figure.
    """
    _set_style()
    figures_dir = config["report"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    embedding_methods = {mc["label"] for mc in config["models"]}
    baseline_methods = {"tfidf", "fuzzy", "bm25"}

    # Find best embedding config
    emb_metrics = [m for m in metrics if m["method"] in embedding_methods]
    if not emb_metrics:
        raise ValueError("No embedding model metrics found")
    best_emb = max(emb_metrics, key=lambda m: m["metrics"]["mrr"])

    # Find best baseline configs
    base_metrics = [m for m in metrics if m["method"] in baseline_methods]
    best_baselines: dict[str, dict] = {}
    for m in base_metrics:
        method = m["method"]
        if method not in best_baselines or m["metrics"]["mrr"] > best_baselines[method]["metrics"]["mrr"]:
            best_baselines[method] = m

    all_configs = [best_emb] + [best_baselines[m] for m in sorted(best_baselines)]
    labels = [f"{c['method']}@{c['granularity']}" for c in all_configs]
    mrr_values = [c["metrics"]["mrr"] for c in all_configs]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71"] + ["#3498db"] * len(best_baselines)
    ax.bar(range(len(labels)), mrr_values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MRR")
    ax.set_title("Best Embedding vs Baselines (MRR)")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    path = os.path.join(figures_dir, "baseline_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_difficulty_bars(metrics: list[dict], config: dict) -> str:
    """Generate MRR by difficulty tier chart.

    Returns:
        Path to saved figure.
    """
    _set_style()
    figures_dir = config["report"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Use best granularity for each method
    best_by_method: dict[str, dict] = {}
    for m in metrics:
        method = m["method"]
        if method not in best_by_method or m["metrics"]["mrr"] > best_by_method[method]["metrics"]["mrr"]:
            best_by_method[method] = m

    methods = sorted(best_by_method.keys())
    difficulties = ["easy", "medium", "hard"]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for di, diff in enumerate(difficulties):
        values = [
            best_by_method[m]["by_difficulty"].get(diff, {}).get("mrr", 0.0)
            for m in methods
        ]
        ax.bar(x + di * width - width, values, width, label=diff.capitalize())

    ax.set_xlabel("Method")
    ax.set_ylabel("MRR")
    ax.set_title("MRR by Difficulty Tier (Best Granularity)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    path = os.path.join(figures_dir, "mrr_by_difficulty.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _error_analysis(
    metrics: list[dict],
    all_rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    config: dict,
) -> list[dict]:
    """Find the worst failures for the best model configuration.

    Returns:
        List of error analysis case dicts.
    """
    n_errors = config["report"]["error_analysis_count"]

    # Find best config (highest MRR)
    best = max(metrics, key=lambda m: m["metrics"]["mrr"])
    best_method = best["method"]
    best_gran = best["granularity"]

    # Build target lookup
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    case_lookup = {c["id"]: c for c in test_cases}

    # Filter rankings to best config
    best_rankings = [
        r for r in all_rankings
        if r["method"] == best_method and r["granularity"] == best_gran
    ]

    # Compute reciprocal rank for each
    case_rrs: list[tuple[str, float, dict]] = []
    for ranking in best_rankings:
        case = case_lookup[ranking["test_case_id"]]
        correct_role_names = {cr["role"] for cr in case["correct_roles"]}

        rr = 0.0
        correct_rank = None
        for idx, result in enumerate(ranking["ranked_results"][:10], 1):
            target = target_lookup[result["target_id"]]
            if _is_correct(target, correct_role_names, best_gran):
                rr = 1.0 / idx
                correct_rank = idx
                break

        case_rrs.append((ranking["test_case_id"], rr, ranking))

    # Sort by RR ascending (worst first)
    case_rrs.sort(key=lambda x: x[1])

    errors: list[dict] = []
    for case_id, rr, ranking in case_rrs[:n_errors]:
        case = case_lookup[case_id]
        correct_role_names = {cr["role"] for cr in case["correct_roles"]}
        expected_categories = {cr["category"] for cr in case["correct_roles"]}
        ranked_results = ranking["ranked_results"]

        rank1_target = target_lookup[ranked_results[0]["target_id"]]
        rank1_category = rank1_target["category"]

        # Find correct rank
        correct_rank = None
        for idx, result in enumerate(ranked_results[:10], 1):
            target = target_lookup[result["target_id"]]
            if _is_correct(target, correct_role_names, best_gran):
                correct_rank = idx
                break

        # Classify failure mode
        if rr == 0.0:
            if ranked_results[0]["score"] < 0.5:
                failure_mode = "surface-mismatch"
            elif len(expected_categories) > 1:
                failure_mode = "ambiguity"
            elif rank1_category not in expected_categories:
                failure_mode = "category-confusion"
            else:
                failure_mode = "rank-displacement"
        elif correct_rank and correct_rank > 5:
            failure_mode = "rank-displacement"
        elif rank1_category in expected_categories:
            failure_mode = "near-miss"
        else:
            failure_mode = "category-confusion"

        errors.append({
            "test_case_id": case_id,
            "input_title": case["input_title"],
            "expected_roles": [cr["role"] for cr in case["correct_roles"]],
            "expected_categories": list(expected_categories),
            "rank1_target": ranked_results[0]["target_id"],
            "rank1_score": ranked_results[0]["score"],
            "correct_rank": correct_rank,
            "reciprocal_rank": rr,
            "failure_mode": failure_mode,
            "difficulty": case["difficulty"],
        })

    return errors


def generate_report(
    metrics: list[dict],
    significance: dict,
    test_cases: list[dict],
    all_rankings: list[dict],
    target_sets: dict,
    config: dict,
) -> str:
    """Generate a full markdown report.

    Args:
        metrics: List of metrics result dicts from evaluate_all.
        significance: Dict from run_statistical_tests.
        test_cases: List of test case dicts.
        all_rankings: List of all ranking dicts.
        target_sets: Dict mapping granularity to target lists.
        config: Experiment configuration dict.

    Returns:
        Markdown string of the full report.

    Raises:
        ValueError: If metrics list is empty.
    """
    if not metrics:
        raise ValueError("No metrics provided for report generation")

    figures_dir = config["report"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # Generate figures
    _plot_heatmap(metrics, config)
    _plot_topk_bars(metrics, config)
    _plot_baseline_comparison(metrics, config)
    _plot_difficulty_bars(metrics, config)

    # Find best config
    best = max(metrics, key=lambda m: m["metrics"]["mrr"])
    thresholds = config["evaluation"]["production_thresholds"]
    mrr_threshold = thresholds["mrr"]
    top3_threshold = thresholds["top3"]

    passes_mrr = best["metrics"]["mrr"] >= mrr_threshold
    passes_top3 = best["metrics"]["top3"] >= top3_threshold

    embedding_methods = {mc["label"] for mc in config["models"]}
    baseline_methods = {"tfidf", "fuzzy", "bm25"}

    # Error analysis
    errors = _error_analysis(
        metrics, all_rankings, test_cases, target_sets, config
    )

    # ── Build report ──────────────────────────────────────────────
    lines: list[str] = []

    # 1. Executive Summary
    lines.append("# Job Title Embedding Experiment Report\n")
    lines.append("## 1. Executive Summary\n")
    lines.append(f"**Best configuration**: {best['method']} @ {best['granularity']}\n")
    lines.append(f"- MRR: {best['metrics']['mrr']:.3f}")
    lines.append(f"- Top-1: {best['metrics']['top1']:.3f}")
    lines.append(f"- Top-3: {best['metrics']['top3']:.3f}")
    lines.append(f"- Top-5: {best['metrics']['top5']:.3f}")
    lines.append(f"- Category Accuracy: {best['metrics']['category_accuracy']:.3f}\n")

    rec = "PASS" if (passes_mrr and passes_top3) else "FAIL"
    lines.append(f"**Production threshold check**: {rec}")
    lines.append(f"- MRR >= {mrr_threshold}: {'PASS' if passes_mrr else 'FAIL'} ({best['metrics']['mrr']:.3f})")
    lines.append(f"- Top-3 >= {top3_threshold}: {'PASS' if passes_top3 else 'FAIL'} ({best['metrics']['top3']:.3f})\n")

    if passes_mrr and passes_top3:
        lines.append("**Recommendation**: The embedding approach meets production thresholds and is recommended for deployment.\n")
    else:
        lines.append("**Recommendation**: The embedding approach does not yet meet all production thresholds. Further tuning or alternative approaches may be needed.\n")

    # 2. Methodology
    lines.append("## 2. Methodology\n")
    n_roles = len(target_sets.get("role", []))
    n_categories = len(target_sets.get("category", []))
    n_test_cases = len(test_cases)
    n_models = len(config["models"])
    lines.append(f"- **Taxonomy**: {n_roles} roles across {n_categories} categories")
    lines.append(f"- **Test data**: {n_test_cases} test cases (easy/medium/hard)")
    lines.append(f"- **Embedding models**: {n_models} ({', '.join(mc['label'] for mc in config['models'])})")
    lines.append(f"- **Granularity levels**: 5 (role, role_desc, cluster, category_desc, category)")
    lines.append(f"- **Baselines**: 3 (TF-IDF, Fuzzy, BM25)")
    lines.append(f"- **Evaluation**: MRR, Top-K accuracy (K=1,3,5), category accuracy\n")

    # 3. Overall Performance
    lines.append("## 3. Overall Performance\n")
    lines.append(f"![MRR Heatmap]({figures_dir}/heatmap_mrr_by_model_granularity.png)\n")

    # Full metrics table
    lines.append("| Method | Granularity | MRR | Top-1 | Top-3 | Top-5 | Cat. Acc. |")
    lines.append("|--------|-------------|-----|-------|-------|-------|-----------|")
    for m in sorted(metrics, key=lambda x: (-x["metrics"]["mrr"])):
        lines.append(
            f"| {m['method']} | {m['granularity']} | "
            f"{m['metrics']['mrr']:.3f} | {m['metrics']['top1']:.3f} | "
            f"{m['metrics']['top3']:.3f} | {m['metrics']['top5']:.3f} | "
            f"{m['metrics']['category_accuracy']:.3f} |"
        )
    lines.append("")

    # 4. Top-K Accuracy
    lines.append("## 4. Top-K Accuracy\n")
    lines.append(f"![Top-K Accuracy]({figures_dir}/topk_accuracy_by_method.png)\n")

    # 5. Baseline Comparison
    lines.append("## 5. Baseline Comparison\n")
    lines.append(f"![Baseline Comparison]({figures_dir}/baseline_comparison.png)\n")

    emb_metrics = [m for m in metrics if m["method"] in embedding_methods]
    base_metrics = [m for m in metrics if m["method"] in baseline_methods]
    if emb_metrics and base_metrics:
        best_emb = max(emb_metrics, key=lambda m: m["metrics"]["mrr"])
        best_base = max(base_metrics, key=lambda m: m["metrics"]["mrr"])
        improvement = best_emb["metrics"]["mrr"] - best_base["metrics"]["mrr"]
        lines.append(
            f"Best embedding ({best_emb['method']}@{best_emb['granularity']}) "
            f"MRR: {best_emb['metrics']['mrr']:.3f} vs "
            f"best baseline ({best_base['method']}@{best_base['granularity']}) "
            f"MRR: {best_base['metrics']['mrr']:.3f} "
            f"(delta: {improvement:+.3f})\n"
        )

    # 6. Difficulty Breakdown
    lines.append("## 6. Difficulty Breakdown\n")
    lines.append(f"![MRR by Difficulty]({figures_dir}/mrr_by_difficulty.png)\n")

    lines.append("| Method | Granularity | Easy MRR | Medium MRR | Hard MRR |")
    lines.append("|--------|-------------|----------|------------|----------|")
    for m in sorted(metrics, key=lambda x: (-x["metrics"]["mrr"])):
        bd = m["by_difficulty"]
        lines.append(
            f"| {m['method']} | {m['granularity']} | "
            f"{bd.get('easy', {}).get('mrr', 0.0):.3f} | "
            f"{bd.get('medium', {}).get('mrr', 0.0):.3f} | "
            f"{bd.get('hard', {}).get('mrr', 0.0):.3f} |"
        )
    lines.append("")

    # 7. Category Accuracy
    lines.append("## 7. Category Accuracy\n")
    lines.append("| Method | Granularity | Category Accuracy |")
    lines.append("|--------|-------------|-------------------|")
    for m in sorted(metrics, key=lambda x: (-x["metrics"]["category_accuracy"])):
        lines.append(
            f"| {m['method']} | {m['granularity']} | "
            f"{m['metrics']['category_accuracy']:.3f} |"
        )
    lines.append("")

    # 8. Statistical Significance
    lines.append("## 8. Statistical Significance\n")
    pairwise = significance.get("pairwise_tests", [])
    bonf = significance.get("bonferroni_alpha", 0.05)
    lines.append(f"Bonferroni-corrected alpha: {bonf:.6f}\n")

    if pairwise:
        lines.append("| Method A | Method B | Granularity | n_discordant | p-value | Significant |")
        lines.append("|----------|----------|-------------|-------------|---------|-------------|")
        for t in pairwise:
            sig = "Yes" if t["significant"] else "No"
            lines.append(
                f"| {t['method_a']} | {t['method_b']} | "
                f"{t.get('granularity', 'N/A')} | {t['n_discordant']} | "
                f"{t['p_value']:.4f} | {sig} |"
            )
        lines.append("")

    # Bootstrap CIs
    bootstrap_cis = significance.get("bootstrap_cis", {})
    if bootstrap_cis:
        lines.append("### Bootstrap 95% Confidence Intervals\n")
        lines.append("| Configuration | MRR CI | Top-1 CI | Top-3 CI | Top-5 CI |")
        lines.append("|---------------|--------|----------|----------|----------|")
        for key in sorted(bootstrap_cis.keys()):
            ci = bootstrap_cis[key]
            mrr_ci = ci.get("mrr", [0, 0])
            top1_ci = ci.get("top1", [0, 0])
            top3_ci = ci.get("top3", [0, 0])
            top5_ci = ci.get("top5", [0, 0])
            lines.append(
                f"| {key} | [{mrr_ci[0]:.3f}, {mrr_ci[1]:.3f}] | "
                f"[{top1_ci[0]:.3f}, {top1_ci[1]:.3f}] | "
                f"[{top3_ci[0]:.3f}, {top3_ci[1]:.3f}] | "
                f"[{top5_ci[0]:.3f}, {top5_ci[1]:.3f}] |"
            )
        lines.append("")

    # 9. Error Analysis
    lines.append("## 9. Error Analysis\n")
    lines.append(f"Top {len(errors)} worst failures for {best['method']}@{best['granularity']}:\n")
    if errors:
        lines.append("| Input Title | Expected | Rank-1 | Score | Correct Rank | Failure Mode |")
        lines.append("|-------------|----------|--------|-------|--------------|--------------|")
        for err in errors:
            correct_rank = str(err["correct_rank"]) if err["correct_rank"] else ">10"
            lines.append(
                f"| {err['input_title']} | {', '.join(err['expected_roles'])} | "
                f"{err['rank1_target']} | {err['rank1_score']:.3f} | "
                f"{correct_rank} | {err['failure_mode']} |"
            )
        lines.append("")

        # Failure mode summary
        mode_counts: dict[str, int] = {}
        for err in errors:
            mode = err["failure_mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        lines.append("**Failure mode distribution**:\n")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {mode}: {count}")
        lines.append("")

    # 10. Production Threshold Check
    lines.append("## 10. Production Threshold Check\n")
    lines.append(f"Thresholds: MRR >= {mrr_threshold}, Top-3 >= {top3_threshold}\n")
    lines.append("| Method | Granularity | MRR | MRR Pass | Top-3 | Top-3 Pass | Overall |")
    lines.append("|--------|-------------|-----|----------|-------|------------|---------|")
    for m in sorted(metrics, key=lambda x: (-x["metrics"]["mrr"])):
        mrr = m["metrics"]["mrr"]
        top3 = m["metrics"]["top3"]
        mrr_pass = "PASS" if mrr >= mrr_threshold else "FAIL"
        top3_pass = "PASS" if top3 >= top3_threshold else "FAIL"
        overall = "PASS" if (mrr >= mrr_threshold and top3 >= top3_threshold) else "FAIL"
        lines.append(
            f"| {m['method']} | {m['granularity']} | {mrr:.3f} | "
            f"{mrr_pass} | {top3:.3f} | {top3_pass} | {overall} |"
        )
    lines.append("")

    # 11. Appendix
    lines.append("## 11. Appendix\n")
    lines.append("### Full Metrics Summary\n")
    lines.append("See `results/metrics/all_metrics.json` for complete metrics data.\n")
    lines.append("### Dev Set Results\n")
    lines.append("Dev set metrics are available in `results/metrics/dev_metrics.json` for sanity checking. Dev set results are not included in the main report to avoid data leakage.\n")

    return "\n".join(lines)


def write_report(report_md: str, output_path: str) -> None:
    """Write markdown report to file.

    Args:
        report_md: The markdown report string.
        output_path: Path to write the report file.

    Raises:
        ValueError: If report_md is empty.
    """
    if not report_md:
        raise ValueError("Report content is empty")

    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(report_md)
