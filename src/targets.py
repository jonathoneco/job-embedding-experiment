"""Build target sets at multiple granularity levels for embedding experiments."""

from src.taxonomy import get_categories


def build_target_sets(
    roles: list[dict],
    clusters: list[dict],
    descriptions: dict,
) -> dict[str, list[dict]]:
    """Build 5 target sets at different granularity levels.

    Target sets:
    - role: One target per role (692)
    - role_desc: Role name + description (692)
    - cluster: One target per cluster (~96)
    - category_desc: Category name + keywords (42)
    - category: One target per category (42)

    Args:
        roles: List of {"role": str, "category": str} dicts.
        clusters: List of {"cluster_label": str, "category": str,
                  "roles": [str, ...]} dicts.
        descriptions: {"roles": {name: desc}, "categories": {name: keywords}}.

    Returns:
        Dict mapping granularity name to list of target dicts.

    Raises:
        KeyError: If a role or category is missing from descriptions.
    """
    categories = get_categories(roles)

    # ── role targets (692) ────────────────────────────────────────
    role_targets: list[dict] = []
    for i, entry in enumerate(roles, start=1):
        role_targets.append({
            "id": f"T-role-{i:04d}",
            "text": entry["role"],
            "role": entry["role"],
            "category": entry["category"],
            "granularity": "role",
        })

    # ── role_desc targets (692) ───────────────────────────────────
    role_desc_targets: list[dict] = []
    for i, entry in enumerate(roles, start=1):
        desc = descriptions["roles"][entry["role"]]
        role_desc_targets.append({
            "id": f"T-rdesc-{i:04d}",
            "text": f"{entry['role']}: {desc}",
            "role": entry["role"],
            "category": entry["category"],
            "granularity": "role_desc",
        })

    # ── cluster targets (~96) ────────────────────────────────────
    cluster_targets: list[dict] = []
    for i, cluster in enumerate(clusters, start=1):
        cluster_targets.append({
            "id": f"T-clust-{i:04d}",
            "text": cluster["cluster_label"],
            "roles": list(cluster["roles"]),
            "category": cluster["category"],
            "cluster_label": cluster["cluster_label"],
            "granularity": "cluster",
        })

    # ── category_desc targets (42) ───────────────────────────────
    category_desc_targets: list[dict] = []
    for i, (cat_name, role_names) in enumerate(categories.items(), start=1):
        keywords = descriptions["categories"][cat_name]
        category_desc_targets.append({
            "id": f"T-cdesc-{i:04d}",
            "text": f"{cat_name}: {keywords}",
            "roles": list(role_names),
            "category": cat_name,
            "granularity": "category_desc",
        })

    # ── category targets (42) ────────────────────────────────────
    category_targets: list[dict] = []
    for i, (cat_name, role_names) in enumerate(categories.items(), start=1):
        category_targets.append({
            "id": f"T-cat-{i:04d}",
            "text": cat_name,
            "roles": list(role_names),
            "category": cat_name,
            "granularity": "category",
        })

    return {
        "role": role_targets,
        "role_desc": role_desc_targets,
        "cluster": cluster_targets,
        "category_desc": category_desc_targets,
        "category": category_targets,
    }
