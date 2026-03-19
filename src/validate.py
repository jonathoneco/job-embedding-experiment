"""Validation, deduplication, and splitting for test cases."""

import sys
import random

from sklearn.model_selection import train_test_split

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_SOURCES = {"rule-based", "llm-systematic", "llm-adversarial", "manual", "onet"}
_VALID_VARIATION_TYPES = {
    "abbreviation", "synonym", "creative", "jargon", "misspelling",
    "cross-category", "combined-role", "level-prefix", "level-suffix",
    "word-reordering", "minor-rewording", "case-variation",
}


def validate_cases(cases: list[dict], roles: list[dict]) -> list[dict]:
    """Validate test cases against schema and taxonomy.

    Args:
        cases: List of test case dicts.
        roles: List of {"role": str, "category": str} dicts from taxonomy.

    Returns:
        The validated cases list, unchanged.

    Raises:
        ValueError: On any validation failure with a descriptive message.
    """
    if not cases:
        raise ValueError("No cases to validate")

    # Build taxonomy lookup: role -> category
    taxonomy_roles: dict[str, str] = {}
    for r in roles:
        taxonomy_roles[r["role"]] = r["category"]

    required_fields = {"id", "input_title", "correct_roles", "difficulty",
                       "variation_type", "source"}

    seen_ids: set[str] = set()
    seen_titles: set[str] = set()

    for i, case in enumerate(cases):
        # Schema check: required fields
        missing = required_fields - set(case.keys())
        if missing:
            raise ValueError(
                f"Case {i}: missing required fields: {missing}"
            )

        # Type checks
        if case["difficulty"] not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Case {i} (id={case['id']}): invalid difficulty "
                f"'{case['difficulty']}', must be one of {_VALID_DIFFICULTIES}"
            )

        if case["source"] not in _VALID_SOURCES:
            raise ValueError(
                f"Case {i} (id={case['id']}): invalid source "
                f"'{case['source']}', must be one of {_VALID_SOURCES}"
            )

        # correct_roles validation
        cr = case["correct_roles"]
        if not isinstance(cr, list) or len(cr) == 0:
            raise ValueError(
                f"Case {i} (id={case['id']}): correct_roles must be "
                f"a non-empty list"
            )

        for j, role_entry in enumerate(cr):
            if not isinstance(role_entry, dict):
                raise ValueError(
                    f"Case {i} (id={case['id']}): correct_roles[{j}] "
                    f"must be a dict"
                )
            if "role" not in role_entry or "category" not in role_entry:
                raise ValueError(
                    f"Case {i} (id={case['id']}): correct_roles[{j}] "
                    f"missing 'role' or 'category'"
                )
            if not isinstance(role_entry["role"], str):
                raise ValueError(
                    f"Case {i} (id={case['id']}): correct_roles[{j}]['role'] "
                    f"must be a string"
                )
            if not isinstance(role_entry["category"], str):
                raise ValueError(
                    f"Case {i} (id={case['id']}): correct_roles[{j}]['category'] "
                    f"must be a string"
                )

            # Taxonomy membership
            role_name = role_entry["role"]
            if role_name not in taxonomy_roles:
                raise ValueError(
                    f"Case {i} (id={case['id']}): role '{role_name}' "
                    f"not found in taxonomy"
                )

            # Category consistency
            expected_category = taxonomy_roles[role_name]
            if role_entry["category"] != expected_category:
                raise ValueError(
                    f"Case {i} (id={case['id']}): role '{role_name}' "
                    f"has category '{role_entry['category']}' but taxonomy "
                    f"says '{expected_category}'"
                )

        # ID uniqueness
        case_id = case["id"]
        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate case id: '{case_id}'"
            )
        seen_ids.add(case_id)

        # Input title uniqueness (case-insensitive)
        title_lower = case["input_title"].lower()
        if title_lower in seen_titles:
            raise ValueError(
                f"Case {i} (id={case['id']}): duplicate input_title "
                f"'{case['input_title']}'"
            )
        seen_titles.add(title_lower)

    return cases


def _word_jaccard(s1: str, s2: str) -> float:
    """Word-level Jaccard similarity of two lowercased strings."""
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 and not set2:
        return 1.0
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


def deduplicate_cases(
    cases: list[dict], threshold: float = 0.75
) -> list[dict]:
    """Remove near-duplicate test cases based on word-level Jaccard similarity.

    Args:
        cases: List of test case dicts.
        threshold: Jaccard similarity threshold above which cases are
            considered duplicates.

    Returns:
        Deduplicated list, keeping higher-difficulty cases on collision.
    """
    # Track which indices to remove
    removed: set[int] = set()

    for i in range(len(cases)):
        if i in removed:
            continue
        for j in range(i + 1, len(cases)):
            if j in removed:
                continue
            sim = _word_jaccard(
                cases[i]["input_title"], cases[j]["input_title"]
            )
            if sim > threshold:
                # Keep the one with higher difficulty
                rank_i = _DIFFICULTY_RANK.get(cases[i]["difficulty"], 0)
                rank_j = _DIFFICULTY_RANK.get(cases[j]["difficulty"], 0)
                if rank_i >= rank_j:
                    print(
                        f"Dedup: removing '{cases[j]['input_title']}' "
                        f"(sim={sim:.3f} with '{cases[i]['input_title']}')",
                        file=sys.stderr,
                    )
                    removed.add(j)
                else:
                    print(
                        f"Dedup: removing '{cases[i]['input_title']}' "
                        f"(sim={sim:.3f} with '{cases[j]['input_title']}')",
                        file=sys.stderr,
                    )
                    removed.add(i)
                    break  # i is removed, stop comparing it

    return [c for idx, c in enumerate(cases) if idx not in removed]


def split_dev_test(
    cases: list[dict], dev_size: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split cases into dev and test sets with stratification.

    Stratifies on combined difficulty + primary category key.

    Args:
        cases: List of test case dicts.
        dev_size: Number of cases in the dev set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (dev_cases, test_cases).

    Raises:
        ValueError: If dev_size >= len(cases).
    """
    if dev_size >= len(cases):
        raise ValueError(
            f"dev_size ({dev_size}) must be less than "
            f"total cases ({len(cases)})"
        )

    # Build stratification keys
    strata_keys = []
    for case in cases:
        primary_cat = case["correct_roles"][0]["category"]
        key = f"{case['difficulty']}|{primary_cat}"
        strata_keys.append(key)

    # Merge small strata (size 1) with a fallback stratum
    from collections import Counter
    strata_counts = Counter(strata_keys)
    merged_keys = []
    for key in strata_keys:
        if strata_counts[key] < 2:
            # Use just difficulty as the stratum
            merged_keys.append(key.split("|")[0])
        else:
            merged_keys.append(key)

    # If merged keys still have singletons, fall back to difficulty only
    merged_counts = Counter(merged_keys)
    final_keys = []
    for key in merged_keys:
        if merged_counts[key] < 2:
            final_keys.append("fallback")
        else:
            final_keys.append(key)

    test_size = len(cases) - dev_size

    dev_cases, test_cases = train_test_split(
        cases,
        test_size=test_size,
        train_size=dev_size,
        random_state=seed,
        stratify=final_keys,
    )

    return dev_cases, test_cases
