"""Parse job-roles.md taxonomy into structured role data."""

import re


def parse_taxonomy(source_path: str) -> list[dict]:
    """Parse job-roles.md markdown into a list of role dicts.

    Args:
        source_path: Path to the job-roles.md file.

    Returns:
        List of {"role": str, "category": str} dicts.

    Raises:
        FileNotFoundError: If source_path does not exist.
        ValueError: If parsing produces unexpected results.
    """
    with open(source_path) as f:
        content = f.read()

    roles: list[dict] = []
    current_category: str | None = None

    for line in content.splitlines():
        line = line.strip()

        # Skip blank lines and the top-level header
        if not line or line == "# Job Roles":
            continue

        # Category header: ## <Category Name>
        category_match = re.match(r"^## (.+)$", line)
        if category_match:
            current_category = category_match.group(1).strip()
            continue

        # Role line: - <Role Name>
        role_match = re.match(r"^- (.+)$", line)
        if role_match:
            if current_category is None:
                raise ValueError(f"Role found before any category: {line}")
            role_name = role_match.group(1).strip()
            roles.append({"role": role_name, "category": current_category})
            continue

        # Any other non-blank line is unexpected
        raise ValueError(f"Unexpected line in taxonomy: {line!r}")

    if not roles:
        raise ValueError("No roles parsed from taxonomy file")

    return roles


def get_categories(roles: list[dict]) -> dict[str, list[str]]:
    """Return a mapping of category name to list of role names.

    Args:
        roles: List of {"role": str, "category": str} dicts.

    Returns:
        Dict mapping category names to lists of role names.
    """
    categories: dict[str, list[str]] = {}
    for entry in roles:
        cat = entry["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(entry["role"])
    return categories
