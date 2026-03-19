"""Rule-based test case generator for deterministic job title variations."""

import random
import sys
from collections.abc import Callable

from src.taxonomy import get_categories


# ── Transform helpers ────────────────────────────────────────────────

_ABBREVIATIONS = {
    "Manager": "Mgr",
    "Engineer": "Eng",
    "Specialist": "Spec",
    "Administrator": "Admin",
    "Coordinator": "Coord",
    "Director": "Dir",
    "Analyst": "Anlst",
    "Representative": "Rep",
    "Developer": "Dev",
    "Assistant": "Asst",
    "Operations": "Ops",
    "Associate": "Assoc",
    "Consultant": "Conslt",
    "Accountant": "Acct",
}

_MINOR_REWORDS = {
    "Representative": "Agent",
    "Specialist": "Expert",
    "Coordinator": "Organizer",
    "Administrator": "Supervisor",
    "Manager": "Lead",
    "Engineer": "Developer",
    "Developer": "Programmer",
    "Consultant": "Advisor",
    "Planner": "Strategist",
    "Officer": "Chief",
}


def _apply_level_prefix(role_name: str, rng: random.Random) -> str:
    """Add a level prefix to a role name."""
    prefixes = ["Senior ", "Lead ", "Junior ", "Principal ", "Associate "]
    prefix = rng.choice(prefixes)
    return prefix + role_name


def _apply_level_suffix(role_name: str, rng: random.Random) -> str:
    """Add a level suffix to a role name."""
    suffixes = [" III", " II", " IV", " Lead", " Senior"]
    suffix = rng.choice(suffixes)
    return role_name + suffix


def _apply_word_reorder(role_name: str, rng: random.Random) -> str:
    """Reorder words in a multi-word title."""
    words = role_name.split()
    if len(words) < 2:
        return role_name + ", General"
    # Pick a split point and swap
    split = rng.randint(1, len(words) - 1)
    return ", ".join([" ".join(words[split:]), " ".join(words[:split])])


def _apply_abbreviation(role_name: str, rng: random.Random) -> str:
    """Abbreviate common words in the role name."""
    result = role_name
    applied = False
    for full, abbr in _ABBREVIATIONS.items():
        if full in result:
            result = result.replace(full, abbr, 1)
            applied = True
            break
    if not applied:
        # Fallback: abbreviate first long word
        words = result.split()
        for i, word in enumerate(words):
            if len(word) > 4:
                words[i] = word[:3] + "."
                result = " ".join(words)
                break
    return result


def _apply_minor_rewording(role_name: str, rng: random.Random) -> str:
    """Apply minor rewording substitutions."""
    result = role_name
    applied = False
    for full, reword in _MINOR_REWORDS.items():
        if full in result:
            result = result.replace(full, reword, 1)
            applied = True
            break
    if not applied:
        # Fallback: drop parenthetical qualifiers or add "the"
        if "(" in result:
            idx = result.index("(")
            result = result[:idx].strip()
        else:
            result = "The " + result
    return result


def _apply_case_variation(role_name: str, rng: random.Random) -> str:
    """Apply case variation to the role name.

    Always produces output that differs from the original string
    (exact match), even if the lowercased form is the same.
    """
    candidates = [role_name.lower(), role_name.upper(), role_name.swapcase()]
    # Filter out any that are exactly the same string as the original
    candidates = [c for c in candidates if c != role_name]
    if not candidates:
        return role_name.lower()
    return rng.choice(candidates)


# ── Transform registry ───────────────────────────────────────────────

_TRANSFORMS = [
    ("level-prefix", _apply_level_prefix, 25),
    ("abbreviation", _apply_abbreviation, 25),
    ("minor-rewording", _apply_minor_rewording, 20),
    ("word-reordering", _apply_word_reorder, 15),
    ("case-variation", _apply_case_variation, 15),
    ("level-suffix", _apply_level_suffix, 20),
]


def generate_rule_cases(roles: list[dict], seed: int) -> list[dict]:
    """Generate exactly 120 easy test cases via deterministic text transforms.

    Args:
        roles: List of {"role": str, "category": str} dicts from taxonomy.
        seed: Random seed for deterministic generation.

    Returns:
        List of 120 test case dicts with the test case schema fields.

    Raises:
        ValueError: If unable to generate 120 unique cases.
    """
    rng = random.Random(seed)
    categories = get_categories(roles)
    cat_names = sorted(categories.keys())

    # Build a full pool of all (role_name, category) tuples for fallback
    all_roles: list[tuple[str, str]] = [
        (r["role"], r["category"]) for r in roles
    ]

    # Build a stratified pool: ~3 roles per category
    roles_per_cat = max(1, 120 // len(cat_names))
    extra = 120 - (roles_per_cat * len(cat_names))

    role_pool: list[tuple[str, str]] = []
    for i, cat in enumerate(cat_names):
        cat_roles = categories[cat]
        n = roles_per_cat + (1 if i < extra else 0)
        n = min(n, len(cat_roles))
        selected = rng.sample(cat_roles, n)
        for role_name in selected:
            role_pool.append((role_name, cat))

    # Shuffle the pool
    rng.shuffle(role_pool)

    # Distribute transforms across the pool
    transform_assignments: list[tuple[str, Callable]] = []
    for var_type, fn, count in _TRANSFORMS:
        transform_assignments.extend([(var_type, fn)] * count)

    # Shuffle transform assignments
    rng.shuffle(transform_assignments)

    seen_titles: set[str] = set()
    cases: list[dict] = []
    skipped_count = 0
    skipped_details: list[str] = []

    # Cycle through role pool if needed
    pool_idx = 0

    for var_type, fn in transform_assignments:
        if len(cases) >= 120:
            break

        # Try roles from the pool first, then fall back to all roles
        found = False
        for source_pool in [role_pool, all_roles]:
            if found:
                break
            for attempt_idx in range(len(source_pool)):
                idx = (pool_idx + attempt_idx) % len(source_pool)
                role_name, category = source_pool[idx]

                input_title = fn(role_name, rng)

                # Skip if exact duplicate of original
                if input_title == role_name:
                    continue

                # Skip if case-insensitive duplicate of another generated title
                lower_title = input_title.lower()
                if lower_title in seen_titles:
                    continue

                seen_titles.add(lower_title)
                cases.append({
                    "id": "",
                    "input_title": input_title,
                    "correct_roles": [{"role": role_name, "category": category}],
                    "difficulty": "easy",
                    "variation_type": var_type,
                    "source": "rule-based",
                    "notes": f"{var_type} transform of '{role_name}'",
                })
                found = True
                break

        if not found:
            skipped_count += 1
            skipped_details.append(f"{var_type}:{role_name}")
        pool_idx += 1

    if skipped_count:
        print(
            f"Rule generation: skipped {skipped_count} transforms "
            f"(no unique title produced): {', '.join(skipped_details[:10])}",
            file=sys.stderr,
        )

    if len(cases) != 120:
        raise ValueError(
            f"Generated {len(cases)} cases instead of 120. "
            f"Could not produce enough unique titles."
        )

    return cases
