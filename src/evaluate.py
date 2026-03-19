"""Metrics engine for evaluating job title embedding matching."""


def _is_correct(target: dict, correct_role_names: set[str], granularity: str) -> bool:
    """Check if a target matches any of the correct role names.

    For role/role_desc granularity: match on singular 'role' field.
    For cluster/category_desc/category: match if ANY role in 'roles' array overlaps.
    """
    if granularity in ("role", "role_desc"):
        return target["role"] in correct_role_names
    else:  # cluster, category_desc, category
        return bool(set(target["roles"]) & correct_role_names)


def compute_metrics(
    rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    top_k_values: list[int],
) -> dict:
    """Compute metrics for a single method x granularity combination.

    Args:
        rankings: List of ranking dicts with test_case_id, method, granularity,
            ranked_results.
        test_cases: List of test case dicts with id, correct_roles, difficulty.
        target_sets: Dict mapping granularity to list of target dicts.
        top_k_values: List of K values for top-K accuracy (e.g., [1, 3, 5]).

    Returns:
        Dict matching spec 00 Metrics Result schema.

    Raises:
        ValueError: If rankings or test_cases are empty, or if required data
            is missing.
    """
    if not rankings:
        raise ValueError("No rankings provided")
    if not test_cases:
        raise ValueError("No test cases provided")

    # All rankings must share method and granularity
    method = rankings[0]["method"]
    granularity = rankings[0]["granularity"]

    # Build target lookup across all granularities
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    # Build test case lookup
    case_lookup = {c["id"]: c for c in test_cases}

    reciprocal_ranks: list[float] = []
    topk_hits: dict[int, int] = {k: 0 for k in top_k_values}
    category_correct = 0
    similarity_gaps: list[float] = []
    total = 0

    for ranking in rankings:
        case_id = ranking["test_case_id"]
        if case_id not in case_lookup:
            raise ValueError(f"Test case '{case_id}' not found in test_cases")

        case = case_lookup[case_id]
        correct_role_names = {cr["role"] for cr in case["correct_roles"]}
        ranked_results = ranking["ranked_results"]

        if not ranked_results:
            raise ValueError(
                f"Empty ranked_results for test case '{case_id}'"
            )

        total += 1

        # MRR: find rank of first correct target (1-indexed)
        rr = 0.0
        for rank_idx, result in enumerate(ranked_results[:10], start=1):
            target = target_lookup[result["target_id"]]
            if _is_correct(target, correct_role_names, granularity):
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

        # Top-K accuracy
        for k in top_k_values:
            found = False
            for result in ranked_results[:k]:
                target = target_lookup[result["target_id"]]
                if _is_correct(target, correct_role_names, granularity):
                    found = True
                    break
            if found:
                topk_hits[k] += 1

        # Category accuracy: rank-1 target's category matches expected
        rank1_target = target_lookup[ranked_results[0]["target_id"]]
        expected_categories = {cr["category"] for cr in case["correct_roles"]}
        if rank1_target["category"] in expected_categories:
            category_correct += 1

        # Mean similarity gap: rank-1 score - rank-2 score
        if len(ranked_results) >= 2:
            gap = ranked_results[0]["score"] - ranked_results[1]["score"]
            similarity_gaps.append(gap)

    mrr = sum(reciprocal_ranks) / total
    topk_acc = {k: topk_hits[k] / total for k in top_k_values}
    cat_acc = category_correct / total
    mean_gap = sum(similarity_gaps) / len(similarity_gaps) if similarity_gaps else 0.0

    by_difficulty = compute_by_difficulty(
        rankings, test_cases, target_sets, top_k_values
    )

    return {
        "method": method,
        "granularity": granularity,
        "split": "test",
        "metrics": {
            "mrr": mrr,
            "top1": topk_acc.get(1, 0.0),
            "top3": topk_acc.get(3, 0.0),
            "top5": topk_acc.get(5, 0.0),
            "category_accuracy": cat_acc,
            "mean_similarity_gap": mean_gap,
        },
        "by_difficulty": by_difficulty,
        "bootstrap_ci": {},
    }


def compute_by_difficulty(
    rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    top_k_values: list[int],
) -> dict:
    """Compute metrics grouped by difficulty level.

    Args:
        rankings: List of ranking dicts.
        test_cases: List of test case dicts with difficulty field.
        target_sets: Dict mapping granularity to list of target dicts.
        top_k_values: List of K values for top-K accuracy.

    Returns:
        Dict keyed by difficulty level (easy, medium, hard), each containing
        mrr, top1, top3, top5, category_accuracy.
    """
    # Build case lookup
    case_lookup = {c["id"]: c for c in test_cases}

    # Build target lookup
    target_lookup: dict[str, dict] = {}
    for _gran, targets in target_sets.items():
        for t in targets:
            target_lookup[t["id"]] = t

    # Group rankings by difficulty
    difficulty_groups: dict[str, tuple[list[dict], list[dict]]] = {}
    for ranking in rankings:
        case = case_lookup[ranking["test_case_id"]]
        difficulty = case["difficulty"]
        if difficulty not in difficulty_groups:
            difficulty_groups[difficulty] = ([], [])
        difficulty_groups[difficulty][0].append(ranking)
        difficulty_groups[difficulty][1].append(case)

    granularity = rankings[0]["granularity"]
    result: dict[str, dict] = {}

    for difficulty in ["easy", "medium", "hard"]:
        if difficulty not in difficulty_groups:
            result[difficulty] = {
                "mrr": 0.0,
                "top1": 0.0,
                "top3": 0.0,
                "top5": 0.0,
                "category_accuracy": 0.0,
            }
            continue

        diff_rankings, diff_cases = difficulty_groups[difficulty]
        total = len(diff_rankings)

        reciprocal_ranks: list[float] = []
        topk_hits: dict[int, int] = {k: 0 for k in top_k_values}
        category_correct = 0

        for ranking in diff_rankings:
            case = case_lookup[ranking["test_case_id"]]
            correct_role_names = {cr["role"] for cr in case["correct_roles"]}
            ranked_results = ranking["ranked_results"]

            # MRR
            rr = 0.0
            for rank_idx, r in enumerate(ranked_results[:10], start=1):
                target = target_lookup[r["target_id"]]
                if _is_correct(target, correct_role_names, granularity):
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)

            # Top-K
            for k in top_k_values:
                found = False
                for r in ranked_results[:k]:
                    target = target_lookup[r["target_id"]]
                    if _is_correct(target, correct_role_names, granularity):
                        found = True
                        break
                if found:
                    topk_hits[k] += 1

            # Category accuracy
            rank1_target = target_lookup[ranked_results[0]["target_id"]]
            expected_categories = {cr["category"] for cr in case["correct_roles"]}
            if rank1_target["category"] in expected_categories:
                category_correct += 1

        result[difficulty] = {
            "mrr": sum(reciprocal_ranks) / total,
            "top1": topk_hits.get(1, 0) / total,
            "top3": topk_hits.get(3, 0) / total,
            "top5": topk_hits.get(5, 0) / total,
            "category_accuracy": category_correct / total,
        }

    return result


def evaluate_all(
    all_rankings: list[dict],
    test_cases: list[dict],
    target_sets: dict,
    config: dict,
) -> list[dict]:
    """Evaluate all method x granularity combinations.

    Groups rankings by (method, granularity), computes metrics for each group.
    Expected: 3 embedding models x 5 granularities + 3 baselines x 2
    granularities = 21 configurations.

    Args:
        all_rankings: List of all ranking dicts across methods and granularities.
        test_cases: List of test case dicts.
        target_sets: Dict mapping granularity to list of target dicts.
        config: Experiment configuration dict.

    Returns:
        List of metrics result dicts, one per (method, granularity) combination.

    Raises:
        ValueError: If no rankings are provided.
    """
    if not all_rankings:
        raise ValueError("No rankings provided")

    top_k_values = config["evaluation"]["top_k"]

    # Group by (method, granularity)
    groups: dict[tuple[str, str], list[dict]] = {}
    for ranking in all_rankings:
        key = (ranking["method"], ranking["granularity"])
        if key not in groups:
            groups[key] = []
        groups[key].append(ranking)

    results: list[dict] = []
    for (method, granularity), group_rankings in sorted(groups.items()):
        metrics = compute_metrics(
            group_rankings, test_cases, target_sets, top_k_values
        )
        results.append(metrics)

    return results
