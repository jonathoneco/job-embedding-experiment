"""Statistical analysis for embedding experiment results."""

import numpy as np
from scipy.stats import binomtest, friedmanchisquare

from src.evaluate import _is_correct


def bootstrap_ci(
    rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    metric_fn,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        rankings: List of ranking dicts for a single method x granularity.
        test_cases: List of test case dicts.
        target_sets: Dict mapping granularity to list of target dicts.
        metric_fn: Callable(filtered_rankings, filtered_cases, target_sets) -> float.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple of (lower, upper) percentile bounds.

    Raises:
        ValueError: If rankings or test_cases are empty.
    """
    if not rankings:
        raise ValueError("No rankings provided for bootstrap")
    if not test_cases:
        raise ValueError("No test cases provided for bootstrap")

    rng = np.random.RandomState(seed)
    case_ids = [c["id"] for c in test_cases]
    case_lookup = {c["id"]: c for c in test_cases}
    ranking_lookup = {r["test_case_id"]: r for r in rankings}

    stats: list[float] = []
    for _ in range(n_resamples):
        sample_ids = rng.choice(case_ids, size=len(case_ids), replace=True)
        sample_cases = [case_lookup[cid] for cid in sample_ids]
        sample_rankings = [ranking_lookup[cid] for cid in sample_ids]
        stat = metric_fn(sample_rankings, sample_cases, target_sets)
        stats.append(stat)

    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lower, upper)


def mcnemar_test(
    rankings_a: list[dict],
    rankings_b: list[dict],
    test_cases: list[dict],
    target_sets: dict,
) -> dict:
    """Pairwise McNemar's exact test on top-1 accuracy.

    Compares two methods by counting discordant pairs where one method
    is correct at rank 1 and the other is not.

    Args:
        rankings_a: Rankings for method A.
        rankings_b: Rankings for method B.
        test_cases: List of test case dicts.
        target_sets: Dict mapping granularity to list of target dicts.

    Returns:
        Dict with method_a, method_b, n_discordant, p_value, significant.

    Raises:
        ValueError: If rankings are empty or have mismatched granularities.
    """
    if not rankings_a or not rankings_b:
        raise ValueError("Both ranking sets must be non-empty")

    granularity_a = rankings_a[0]["granularity"]
    granularity_b = rankings_b[0]["granularity"]
    if granularity_a != granularity_b:
        raise ValueError(
            f"Granularity mismatch: '{granularity_a}' vs '{granularity_b}'"
        )

    granularity = granularity_a
    method_a = rankings_a[0]["method"]
    method_b = rankings_b[0]["method"]

    # Build target lookup
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    # Build ranking lookups by test_case_id
    lookup_a = {r["test_case_id"]: r for r in rankings_a}
    lookup_b = {r["test_case_id"]: r for r in rankings_b}

    case_lookup = {c["id"]: c for c in test_cases}

    b = 0  # A correct, B not
    c = 0  # B correct, A not

    for case in test_cases:
        case_id = case["id"]
        if case_id not in lookup_a or case_id not in lookup_b:
            raise ValueError(
                f"Test case '{case_id}' missing from one of the ranking sets"
            )

        correct_role_names = {cr["role"] for cr in case["correct_roles"]}

        # Check if A is correct at rank 1
        rank1_a = target_lookup[lookup_a[case_id]["ranked_results"][0]["target_id"]]
        a_correct = _is_correct(rank1_a, correct_role_names, granularity)

        # Check if B is correct at rank 1
        rank1_b = target_lookup[lookup_b[case_id]["ranked_results"][0]["target_id"]]
        b_correct = _is_correct(rank1_b, correct_role_names, granularity)

        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1

    n_discordant = b + c

    if n_discordant == 0:
        return {
            "method_a": method_a,
            "method_b": method_b,
            "n_discordant": 0,
            "p_value": 1.0,
            "significant": False,
        }

    result = binomtest(b, n_discordant, 0.5, alternative='two-sided')
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_discordant": n_discordant,
        "p_value": float(result.pvalue),
        "significant": False,  # Set after Bonferroni correction
    }


def _compute_mrr(
    rankings: list[dict],
    test_cases: list[dict],
    target_lookup: dict[str, dict],
    granularity: str,
) -> float:
    """Compute MRR for a set of rankings at a given granularity."""
    case_lookup = {c["id"]: c for c in test_cases}
    rr_sum = 0.0
    total = 0
    for ranking in rankings:
        case = case_lookup[ranking["test_case_id"]]
        correct = {cr["role"] for cr in case["correct_roles"]}
        rr = 0.0
        for idx, r in enumerate(ranking["ranked_results"][:10], 1):
            if _is_correct(target_lookup[r["target_id"]], correct, granularity):
                rr = 1.0 / idx
                break
        rr_sum += rr
        total += 1
    return rr_sum / total if total > 0 else 0.0


def friedman_nemenyi_test(
    all_rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    config: dict,
) -> dict:
    """Run Friedman test with Nemenyi post-hoc across all methods per granularity.

    For each granularity, collects per-case ranks across all methods (embedding
    + baseline), runs Friedman chi-square test, and if significant, computes
    Nemenyi critical difference.

    Args:
        all_rankings: List of all ranking dicts.
        test_cases: List of test case dicts.
        target_sets: Dict mapping granularity to list of target dicts.
        config: Experiment configuration dict.

    Returns:
        Dict mapping granularity to results with statistic, p_value,
        significant, and cd_groups (if significant).
    """
    # Build target lookup
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    # Group by (method, granularity)
    groups: dict[tuple[str, str], list[dict]] = {}
    for ranking in all_rankings:
        key = (ranking["method"], ranking["granularity"])
        if key not in groups:
            groups[key] = []
        groups[key].append(ranking)

    all_granularities = sorted({key[1] for key in groups})
    case_lookup = {c["id"]: c for c in test_cases}
    results: dict[str, dict] = {}

    for granularity in all_granularities:
        methods = sorted([key[0] for key in groups if key[1] == granularity])
        if len(methods) < 3:
            # Friedman requires at least 3 groups
            continue

        # Compute per-case MRR for each method
        case_ids = [c["id"] for c in test_cases]
        method_scores: dict[str, dict[str, float]] = {}
        for method in methods:
            rankings_for = groups.get((method, granularity), [])
            ranking_by_case = {r["test_case_id"]: r for r in rankings_for}
            scores: dict[str, float] = {}
            for cid in case_ids:
                if cid not in ranking_by_case:
                    scores[cid] = 0.0
                    continue
                ranking = ranking_by_case[cid]
                case = case_lookup[cid]
                correct = {cr["role"] for cr in case["correct_roles"]}
                rr = 0.0
                for idx, r in enumerate(ranking["ranked_results"][:10], 1):
                    if _is_correct(
                        target_lookup[r["target_id"]], correct, granularity
                    ):
                        rr = 1.0 / idx
                        break
                scores[cid] = rr
            method_scores[method] = scores

        # Build rank matrix: for each case, rank methods by MRR (higher = better rank = 1)
        rank_arrays: dict[str, list[float]] = {m: [] for m in methods}
        for cid in case_ids:
            scores_for_case = [(method_scores[m].get(cid, 0.0), m) for m in methods]
            # Sort descending by score; assign ranks (1 = best)
            sorted_scores = sorted(scores_for_case, key=lambda x: -x[0])
            # Handle ties with average ranks
            i = 0
            while i < len(sorted_scores):
                j = i + 1
                while j < len(sorted_scores) and sorted_scores[j][0] == sorted_scores[i][0]:
                    j += 1
                avg_rank = sum(range(i + 1, j + 1)) / (j - i)
                for idx in range(i, j):
                    rank_arrays[sorted_scores[idx][1]].append(avg_rank)
                i = j

        # Run Friedman test
        rank_data = [np.array(rank_arrays[m]) for m in methods]
        stat, p_value = friedmanchisquare(*rank_data)

        result: dict = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": float(p_value) < 0.05,
            "methods": methods,
            "mean_ranks": {m: float(np.mean(rank_arrays[m])) for m in methods},
        }

        if float(p_value) < 0.05:
            # Nemenyi critical difference
            k = len(methods)
            n = len(case_ids)
            # Nemenyi q_alpha values for alpha=0.05
            # From studentized range distribution q_alpha / sqrt(2)
            # Standard table values for k=3..10
            q_alpha_table = {
                3: 2.343,
                4: 2.569,
                5: 2.728,
                6: 2.850,
                7: 2.949,
                8: 3.031,
                9: 3.102,
                10: 3.164,
            }
            q_alpha = q_alpha_table.get(k, 3.164)  # fallback for large k
            cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))
            result["critical_difference"] = float(cd)

            # Determine CD groups: methods within CD of each other
            mean_ranks = result["mean_ranks"]
            sorted_methods = sorted(methods, key=lambda m: mean_ranks[m])
            cd_groups: list[list[str]] = []
            for m in sorted_methods:
                placed = False
                for group in cd_groups:
                    if all(
                        abs(mean_ranks[m] - mean_ranks[g]) < cd
                        for g in group
                    ):
                        group.append(m)
                        placed = True
                        break
                if not placed:
                    cd_groups.append([m])
            result["cd_groups"] = cd_groups

        results[granularity] = result

    return results


def run_statistical_tests(
    all_rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    config: dict,
) -> dict:
    """Run all statistical tests: McNemar's pairwise + bootstrap CIs.

    Args:
        all_rankings: List of all ranking dicts.
        test_cases: List of test case dicts.
        target_sets: Dict mapping granularity to list of target dicts.
        config: Experiment configuration dict.

    Returns:
        Dict with pairwise_tests, bootstrap_cis, bonferroni_alpha.

    Raises:
        ValueError: If no rankings are provided.
    """
    if not all_rankings:
        raise ValueError("No rankings provided")

    n_resamples = config["evaluation"]["bootstrap_resamples"]
    seed = config["experiment"]["seed"]
    top_k_values = config["evaluation"]["top_k"]

    # Group by (method, granularity)
    groups: dict[tuple[str, str], list[dict]] = {}
    for ranking in all_rankings:
        key = (ranking["method"], ranking["granularity"])
        if key not in groups:
            groups[key] = []
        groups[key].append(ranking)

    # Identify embedding models and baselines
    embedding_methods = {mc["label"] for mc in config["models"]}
    baseline_methods = {"tfidf", "fuzzy", "bm25"}

    # Get all granularities present
    all_granularities = {key[1] for key in groups}

    # Build target lookup for metric functions
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    # ── Pairwise McNemar tests ────────────────────────────────────
    pairwise_tests: list[dict] = []

    for granularity in sorted(all_granularities):
        # Get methods present at this granularity
        methods_at_gran = sorted(
            [key[0] for key in groups if key[1] == granularity]
        )

        # Embedding model pairs
        emb_at_gran = [m for m in methods_at_gran if m in embedding_methods]
        for i in range(len(emb_at_gran)):
            for j in range(i + 1, len(emb_at_gran)):
                test_result = mcnemar_test(
                    groups[(emb_at_gran[i], granularity)],
                    groups[(emb_at_gran[j], granularity)],
                    test_cases,
                    target_sets,
                )
                test_result["granularity"] = granularity
                pairwise_tests.append(test_result)

        # Best embedding vs baselines
        base_at_gran = [m for m in methods_at_gran if m in baseline_methods]
        if emb_at_gran and base_at_gran:
            # Select embedding model with highest MRR at this granularity
            best_emb = max(
                emb_at_gran,
                key=lambda m: _compute_mrr(
                    groups[(m, granularity)], test_cases,
                    target_lookup, granularity,
                ),
            )
            for baseline in base_at_gran:
                test_result = mcnemar_test(
                    groups[(best_emb, granularity)],
                    groups[(baseline, granularity)],
                    test_cases,
                    target_sets,
                )
                test_result["granularity"] = granularity
                pairwise_tests.append(test_result)

    # Apply Bonferroni correction
    n_tests = len(pairwise_tests)
    bonferroni_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    for test_result in pairwise_tests:
        test_result["significant"] = test_result["p_value"] < bonferroni_alpha

    # ── Bootstrap CIs ────────────────────────────────────────────

    def _make_mrr_fn(granularity: str):
        def mrr_fn(rankings, cases, ts):
            tl = {}
            for _g, tgts in ts.items():
                for t in tgts:
                    tl[t["id"]] = t
            cl = {c["id"]: c for c in cases}
            total = 0
            rr_sum = 0.0
            for ranking in rankings:
                case = cl[ranking["test_case_id"]]
                correct = {cr["role"] for cr in case["correct_roles"]}
                rr = 0.0
                for idx, r in enumerate(ranking["ranked_results"][:10], 1):
                    if _is_correct(tl[r["target_id"]], correct, granularity):
                        rr = 1.0 / idx
                        break
                rr_sum += rr
                total += 1
            return rr_sum / total if total > 0 else 0.0
        return mrr_fn

    def _make_topk_fn(k: int, granularity: str):
        def topk_fn(rankings, cases, ts):
            tl = {}
            for _g, tgts in ts.items():
                for t in tgts:
                    tl[t["id"]] = t
            cl = {c["id"]: c for c in cases}
            hits = 0
            total = 0
            for ranking in rankings:
                case = cl[ranking["test_case_id"]]
                correct = {cr["role"] for cr in case["correct_roles"]}
                found = False
                for r in ranking["ranked_results"][:k]:
                    if _is_correct(tl[r["target_id"]], correct, granularity):
                        found = True
                        break
                if found:
                    hits += 1
                total += 1
            return hits / total if total > 0 else 0.0
        return topk_fn

    bootstrap_cis: dict[str, dict] = {}
    for (method, granularity), group_rankings in sorted(groups.items()):
        key = f"{method}@{granularity}"
        cis: dict[str, list[float]] = {}

        # MRR CI
        mrr_lo, mrr_hi = bootstrap_ci(
            group_rankings, test_cases, target_sets,
            _make_mrr_fn(granularity), n_resamples, seed,
        )
        cis["mrr"] = [mrr_lo, mrr_hi]

        # Top-K CIs
        for k in top_k_values:
            lo, hi = bootstrap_ci(
                group_rankings, test_cases, target_sets,
                _make_topk_fn(k, granularity), n_resamples, seed,
            )
            cis[f"top{k}"] = [lo, hi]

        bootstrap_cis[key] = cis

    # ── Friedman + Nemenyi tests ────────────────────────────────
    friedman_results = friedman_nemenyi_test(
        all_rankings, test_cases, target_sets, config,
    )

    return {
        "pairwise_tests": pairwise_tests,
        "bootstrap_cis": bootstrap_cis,
        "bonferroni_alpha": bonferroni_alpha,
        "friedman_nemenyi": friedman_results,
    }
