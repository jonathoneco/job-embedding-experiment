"""Build curated target sets by filtering the full taxonomy to a hand-maintained role list."""

import json


def load_curated_roles(curated_path: str, roles: list[dict]) -> set[str]:
    """Load and validate the curated role list against the full taxonomy.

    Args:
        curated_path: Path to curated_roles.json (flat array of role names).
        roles: Full taxonomy roles list of {"role": str, "category": str} dicts.

    Returns:
        Set of validated curated role name strings.

    Raises:
        ValueError: If any curated role names are not found in the taxonomy.
    """
    with open(curated_path) as f:
        curated_names: list[str] = json.load(f)

    valid_names = {entry["role"] for entry in roles}
    unknown = set(curated_names) - valid_names

    if unknown:
        raise ValueError(f"Unknown role names in curated list: {sorted(unknown)}")

    return set(curated_names)


def build_curated_target_sets(
    curated_roles: set[str],
    roles: list[dict],
    clusters: list[dict],
) -> dict[str, list[dict]]:
    """Build 3 target sets filtered to curated roles only.

    Target sets:
    - curated_role: One target per curated role.
    - curated_cluster: Clusters filtered to curated members (empty clusters dropped).
    - curated_category: Categories derived from curated roles (empty categories dropped).

    Args:
        curated_roles: Set of curated role name strings.
        roles: Full taxonomy roles list of {"role": str, "category": str} dicts.
        clusters: Full taxonomy clusters list of {"cluster_label": str,
                  "category": str, "roles": [str, ...]} dicts.

    Returns:
        Dict mapping granularity name to list of target dicts.
    """
    # ── curated_role targets ───────────────────────────────────────
    role_targets: list[dict] = []
    for i, entry in enumerate(
        [e for e in roles if e["role"] in curated_roles], start=1
    ):
        role_targets.append({
            "id": f"TC-role-{i:04d}",
            "text": entry["role"],
            "role": entry["role"],
            "category": entry["category"],
            "granularity": "curated_role",
        })

    # ── curated_cluster targets ────────────────────────────────────
    cluster_targets: list[dict] = []
    idx = 1
    for cluster in clusters:
        filtered_roles = [r for r in cluster["roles"] if r in curated_roles]
        if not filtered_roles:
            continue
        cluster_targets.append({
            "id": f"TC-clust-{idx:04d}",
            "text": cluster["cluster_label"],
            "roles": filtered_roles,
            "category": cluster["category"],
            "cluster_label": cluster["cluster_label"],
            "granularity": "curated_cluster",
        })
        idx += 1

    # ── curated_category targets ───────────────────────────────────
    categories: dict[str, list[str]] = {}
    for entry in roles:
        if entry["role"] in curated_roles:
            cat = entry["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry["role"])

    category_targets: list[dict] = []
    for i, (cat_name, role_names) in enumerate(categories.items(), start=1):
        category_targets.append({
            "id": f"TC-cat-{i:04d}",
            "text": cat_name,
            "roles": role_names,
            "category": cat_name,
            "granularity": "curated_category",
        })

    return {
        "curated_role": role_targets,
        "curated_cluster": cluster_targets,
        "curated_category": category_targets,
    }


def filter_covered_test_cases(
    test_cases: list[dict],
    curated_roles: set[str],
) -> tuple[list[dict], dict]:
    """Filter test cases to those covered by the curated role set.

    A test case is covered if at least one of its correct_roles has a role
    name present in the curated set.

    Args:
        test_cases: List of test case dicts, each with an "id" field and
                    a "correct_roles" list of {"role": str, ...} dicts.
        curated_roles: Set of curated role name strings.

    Returns:
        Tuple of (covered_cases, coverage_report) where coverage_report
        contains total, covered, excluded counts plus excluded_ids list
        and coverage_pct.
    """
    covered: list[dict] = []
    excluded: list[dict] = []

    for tc in test_cases:
        if any(cr["role"] in curated_roles for cr in tc["correct_roles"]):
            covered.append(tc)
        else:
            excluded.append(tc)

    coverage_report = {
        "total": len(test_cases),
        "covered": len(covered),
        "excluded": len(excluded),
        "excluded_ids": [tc["id"] for tc in excluded],
        "coverage_pct": round(len(covered) / len(test_cases) * 100, 1)
        if test_cases
        else 0.0,
    }

    return covered, coverage_report
