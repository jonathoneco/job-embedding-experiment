"""LLM-based test case generator using Anthropic Claude API."""

import json

import anthropic

from src.taxonomy import get_categories


def generate_llm_cases(roles: list[dict], config: dict) -> list[dict]:
    """Multi-pass LLM generation of test cases.

    Args:
        roles: List of {"role": str, "category": str} dicts from taxonomy.
        config: Full config dict (needs generation.api_model, generation.max_tokens).

    Returns:
        List of test case dicts (IDs left empty for orchestrator to assign).

    Raises:
        anthropic.APIError: On API failures (propagated).
        ValueError: On JSON parsing failures.
    """
    client = anthropic.Anthropic()
    model = config["generation"]["api_model"]
    max_tokens = config["generation"]["max_tokens"]

    all_cases: list[dict] = []

    # Pass 1: Systematic (305 cases)
    all_cases.extend(_pass1_systematic(roles, client, model, max_tokens))

    # Pass 2: Adversarial medium (125 cases)
    all_cases.extend(_pass2_adversarial_medium(roles, client, model, max_tokens))

    # Pass 3: Adversarial hard (125 cases)
    all_cases.extend(_pass3_adversarial_hard(roles, client, model, max_tokens))

    return all_cases


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response text.

    Handles ```json, ```, and trailing ``` fences.

    Args:
        text: Raw text possibly wrapped in code fences.

    Returns:
        Text with code fences removed.
    """
    content = text.strip()
    if content.startswith("```"):
        first_newline = content.index("\n")
        content = content[first_newline + 1:]
        if content.endswith("```"):
            content = content[:-3].strip()
    return content


def _check_stop_reason(response, context: str) -> None:
    """Raise if response was truncated by max_tokens."""
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"LLM response truncated (max_tokens hit) during {context}. "
            f"Increase generation.max_tokens in config.yaml."
        )


def _parse_json_response(text: str) -> list[dict]:
    """Parse JSON from an LLM response, stripping code fences if present.

    Args:
        text: Raw text from the LLM response.

    Returns:
        Parsed list of dicts.

    Raises:
        ValueError: If JSON parsing fails.
    """
    content = _strip_code_fences(text)

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM response: {e}\n"
            f"Raw content:\n{text[:500]}"
        ) from e

    if not isinstance(result, list):
        raise ValueError(
            f"Expected JSON array, got {type(result).__name__}. "
            f"Raw content:\n{text[:500]}"
        )

    return result


def _pass1_systematic(
    roles: list[dict],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Generate systematic variations: one API call per category.

    Target: 105 easy + 200 medium = 305 total.
    ~7-8 cases per category, proportional to category size.
    """
    categories = get_categories(roles)
    cat_names = sorted(categories.keys())
    total_roles = sum(len(r) for r in categories.values())

    system_prompt = (
        "You are generating realistic job title variations for testing a "
        "matching system. Create titles that a real person would actually "
        "use on their resume or LinkedIn profile. Each title must map to "
        "exactly one role in the provided taxonomy."
    )

    all_cases: list[dict] = []

    for cat in cat_names:
        cat_roles = categories[cat]
        # Proportional allocation
        n = max(2, round(305 * len(cat_roles) / total_roles))
        easy_count = max(1, round(n * 105 / 305))
        medium_count = n - easy_count

        roles_list = "\n".join(f"- {r}" for r in cat_roles)

        user_prompt = (
            f"Category: {cat}\n\n"
            f"Available roles in this category:\n{roles_list}\n\n"
            f"Generate {n} job title variations for roles in this category:\n"
            f"- {easy_count} easy variations (obvious mappings, simple rewording)\n"
            f"- {medium_count} medium variations (requires some domain knowledge)\n\n"
            f"Return a JSON array where each element has these fields:\n"
            f'- "input_title": the variant job title (string)\n'
            f'- "correct_role": the exact role name from the list above (string)\n'
            f'- "difficulty": "easy" or "medium" (string)\n'
            f'- "variation_type": one of "abbreviation", "synonym", "creative", '
            f'"jargon", "level-prefix" (string)\n'
            f'- "notes": brief explanation of the variation (string)\n\n'
            f"Output ONLY the JSON array, no other text."
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _check_stop_reason(response, f"pass1 category '{cat}'")

        raw_cases = _parse_json_response(response.content[0].text)

        for raw in raw_cases:
            required = {"input_title", "correct_role", "difficulty"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"LLM response missing keys {missing}: {raw}")

            all_cases.append({
                "id": "",
                "input_title": raw["input_title"],
                "correct_roles": [
                    {"role": raw["correct_role"], "category": cat}
                ],
                "difficulty": raw["difficulty"],
                "variation_type": raw.get("variation_type", "synonym"),
                "source": "llm-systematic",
                "notes": raw.get("notes", ""),
            })

    return all_cases


def _pass2_adversarial_medium(
    roles: list[dict],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Generate adversarial medium cases focused on cross-category confusion.

    Target: 125 medium cases across 2-3 API calls.
    """
    categories = get_categories(roles)

    system_prompt = (
        "You are generating challenging but fair job title variations for "
        "testing a matching system. Focus on titles that could plausibly "
        "belong to multiple categories but have a primary correct mapping. "
        "These should be realistic titles from job postings or resumes."
    )

    taxonomy_text = ""
    for cat in sorted(categories.keys()):
        taxonomy_text += f"\n## {cat}\n"
        for role in categories[cat]:
            taxonomy_text += f"- {role}\n"

    all_cases: list[dict] = []
    cases_per_call = [63, 62]  # Split 125 across 2 calls

    for batch_idx, n in enumerate(cases_per_call):
        user_prompt = (
            f"Full taxonomy:\n{taxonomy_text}\n\n"
            f"Generate {n} medium-difficulty job title variations that "
            f"create cross-category confusion. Each title should:\n"
            f"1. Map primarily to one role but could plausibly map to others\n"
            f"2. Use industry jargon, synonyms, or blended terminology\n"
            f"3. Be realistic (someone would actually use this title)\n\n"
            f"Return a JSON array where each element has:\n"
            f'- "input_title": the variant job title (string)\n'
            f'- "correct_role": the primary correct role name (string)\n'
            f'- "correct_category": the category of the correct role (string)\n'
            f'- "plausible_alternatives": list of objects with "role" and '
            f'"category" for other acceptable mappings (array)\n'
            f'- "variation_type": one of "cross-category", "synonym", '
            f'"jargon" (string)\n'
            f'- "notes": brief explanation (string)\n\n'
            f"{'Focus on different categories than the previous batch. ' if batch_idx > 0 else ''}"
            f"Output ONLY the JSON array, no other text."
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _check_stop_reason(response, f"pass2 batch {batch_idx}")

        raw_cases = _parse_json_response(response.content[0].text)

        for raw in raw_cases:
            required = {"input_title", "correct_role", "correct_category"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"LLM response missing keys {missing}: {raw}")

            correct_roles = [
                {"role": raw["correct_role"], "category": raw["correct_category"]}
            ]
            # Merge plausible alternatives into correct_roles
            for alt in raw.get("plausible_alternatives", []):
                correct_roles.append({
                    "role": alt["role"],
                    "category": alt["category"],
                })

            all_cases.append({
                "id": "",
                "input_title": raw["input_title"],
                "correct_roles": correct_roles,
                "difficulty": "medium",
                "variation_type": raw.get("variation_type", "cross-category"),
                "source": "llm-adversarial",
                "notes": raw.get("notes", ""),
            })

    return all_cases


def _pass3_adversarial_hard(
    roles: list[dict],
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
) -> list[dict]:
    """Generate adversarial hard cases: abbreviation chains, creative titles.

    Target: 125 hard cases across 2-3 API calls.
    """
    categories = get_categories(roles)

    system_prompt = (
        "You are generating very challenging job title variations for "
        "testing a matching system. Create titles using heavy abbreviations, "
        "creative/non-standard titles, misspellings, and combined roles. "
        "These should still be mappable to roles in the taxonomy but "
        "require significant interpretation."
    )

    taxonomy_text = ""
    for cat in sorted(categories.keys()):
        taxonomy_text += f"\n## {cat}\n"
        for role in categories[cat]:
            taxonomy_text += f"- {role}\n"

    all_cases: list[dict] = []
    cases_per_call = [42, 42, 41]  # Split 125 across 3 calls

    variation_focus = [
        "abbreviation chains and creative titles",
        "misspellings, non-English equivalents, and combined roles",
        "extreme jargon, slang, and unconventional titles",
    ]

    for batch_idx, n in enumerate(cases_per_call):
        user_prompt = (
            f"Full taxonomy:\n{taxonomy_text}\n\n"
            f"Generate {n} hard-difficulty job title variations focusing on "
            f"{variation_focus[batch_idx]}. Each title should:\n"
            f"1. Be very difficult to match to the correct role\n"
            f"2. Use heavy abbreviations, creative language, or misspellings\n"
            f"3. Still be genuinely mappable to a taxonomy role\n\n"
            f"Return a JSON array where each element has:\n"
            f'- "input_title": the variant job title (string)\n'
            f'- "correct_role": the primary correct role name (string)\n'
            f'- "correct_category": the category of the correct role (string)\n'
            f'- "plausible_alternatives": list of objects with "role" and '
            f'"category" for other acceptable mappings (array, can be empty)\n'
            f'- "variation_type": one of "abbreviation", "creative", '
            f'"misspelling", "combined-role", "jargon" (string)\n'
            f'- "notes": brief explanation (string)\n\n'
            f"Output ONLY the JSON array, no other text."
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        _check_stop_reason(response, f"pass3 batch {batch_idx}")

        raw_cases = _parse_json_response(response.content[0].text)

        for raw in raw_cases:
            required = {"input_title", "correct_role", "correct_category"}
            missing = required - raw.keys()
            if missing:
                raise ValueError(f"LLM response missing keys {missing}: {raw}")

            correct_roles = [
                {"role": raw["correct_role"], "category": raw["correct_category"]}
            ]
            for alt in raw.get("plausible_alternatives", []):
                correct_roles.append({
                    "role": alt["role"],
                    "category": alt["category"],
                })

            all_cases.append({
                "id": "",
                "input_title": raw["input_title"],
                "correct_roles": correct_roles,
                "difficulty": "hard",
                "variation_type": raw.get("variation_type", "creative"),
                "source": "llm-adversarial",
                "notes": raw.get("notes", ""),
            })

    return all_cases
