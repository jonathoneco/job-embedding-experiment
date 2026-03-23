"""Reciprocal Rank Fusion (RRF) for combining multiple ranking methods."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def fuse_rankings(
    rankings_by_method: dict[str, list[list[dict]]],
    k: int = 60,
    top_n: int = 10,
) -> list[list[dict]]:
    """Fuse multiple ranking lists using RRF.

    Args:
        rankings_by_method: {method_name: [[{target_id, score}, ...], ...]}
            All methods must have the same number of queries.
        k: RRF constant (default 60).
        top_n: Number of results per query.

    Returns:
        Fused rankings per query: [[{target_id, score}, ...], ...]
    """
    methods = list(rankings_by_method.keys())
    query_counts = {m: len(r) for m, r in rankings_by_method.items()}
    unique_counts = set(query_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            f"Mismatched query counts across methods: {query_counts}"
        )

    n_queries = unique_counts.pop()
    results = []

    for qi in range(n_queries):
        # Accumulate RRF scores per target
        rrf_scores: dict[str, float] = defaultdict(float)
        for method in methods:
            for rank_idx, result in enumerate(rankings_by_method[method][qi], start=1):
                rrf_scores[result["target_id"]] += 1.0 / (k + rank_idx)

        # Sort by RRF score descending
        sorted_targets = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True,
        )
        results.append([
            {"target_id": tid, "score": score}
            for tid, score in sorted_targets[:top_n]
        ])

    return results


def fuse_all(
    all_rankings: list[dict],
    fusion_configs: list[dict],
    test_cases: list[dict],
    k: int = 60,
) -> list[dict]:
    """Apply all fusion configs to produce fused results in S0.1 format.

    Args:
        all_rankings: Flat list of S0.1 ranking dicts.
        fusion_configs: [{name, methods, granularity}, ...]
        test_cases: For test_case_id ordering.
        k: RRF constant.

    Returns:
        List of S0.1 ranking dicts for fused results.
    """
    # Group rankings by (method, granularity) -> {test_case_id: ranked_results}
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    for r in all_rankings:
        key = (r["method"], r["granularity"])
        grouped[key][r["test_case_id"]] = r["ranked_results"]

    case_ids = [c["id"] for c in test_cases]
    fused_results = []

    for fc in fusion_configs:
        name = fc["name"]
        methods = fc["methods"]
        granularity = fc["granularity"]

        # Check all required methods are available
        missing = [m for m in methods if (m, granularity) not in grouped]
        if missing:
            logger.warning(
                "Skipping fusion '%s': missing methods %s for granularity '%s'",
                name, missing, granularity,
            )
            continue

        # Build rankings_by_method in test_case_id order
        rankings_by_method = {}
        for method in methods:
            method_rankings = grouped[(method, granularity)]
            missing = [cid for cid in case_ids if cid not in method_rankings]
            if missing:
                raise ValueError(
                    f"Fusion '{name}': method '{method}' missing test cases: {missing}"
                )
            rankings_by_method[method] = [
                method_rankings[cid] for cid in case_ids
            ]

        fused = fuse_rankings(rankings_by_method, k=k)

        for qi, cid in enumerate(case_ids):
            fused_results.append({
                "test_case_id": cid,
                "method": name,
                "granularity": granularity,
                "ranked_results": fused[qi],
            })

    return fused_results
