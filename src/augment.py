"""Target augmentation via LLM-generated role aliases."""

import json
import os

import anthropic

from src.taxonomy import get_categories


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


def generate_augmented_targets(
    roles: list[dict],
    config: dict,
    output_path: str,
) -> list[dict]:
    """Generate augmented targets by creating LLM-generated aliases for each role.

    One API call per category (~42 total). Each role produces 5-10 aliases,
    deduplicated case-insensitively, with the original role name excluded.

    If output_path already exists, loads and returns cached results (idempotent).

    Args:
        roles: List of {"role": str, "category": str} dicts from taxonomy.
        config: Full config dict (needs generation.api_model, generation.max_tokens).
        output_path: Path to write/read the augmented targets JSON file.

    Returns:
        List of augmented target dicts.

    Raises:
        anthropic.APIError: On API failures (propagated).
        ValueError: On JSON parsing failures or truncated responses.
    """
    if os.path.exists(output_path):
        return load_augmented_targets(output_path)

    client = anthropic.Anthropic()
    model = config["generation"]["api_model"]
    max_tokens = config["generation"]["max_tokens"]

    categories = get_categories(roles)

    # Build role -> (id, category) lookup from the canonical role list
    role_lookup: dict[str, tuple[str, str]] = {}
    for i, entry in enumerate(roles, start=1):
        role_id = f"T-role-{i:04d}"
        role_lookup[entry["role"]] = (role_id, entry["category"])

    all_targets: list[dict] = []
    target_counter = 0

    for category, role_names in categories.items():
        role_list = "\n".join(f"- {r}" for r in role_names)

        prompt = (
            f"For each job role below, generate 5-10 alternative job titles that someone might\n"
            f"use to refer to this exact role. Include:\n"
            f"- Common abbreviations (e.g., \"Dev\" for \"Developer\")\n"
            f"- Industry jargon and slang\n"
            f"- Regional variants\n"
            f"- Informal titles\n"
            f"- Compound titles that combine the role with a level or specialty\n"
            f"\n"
            f"Category: {category}\n"
            f"Roles:\n"
            f"{role_list}\n"
            f"\n"
            f"Return JSON array:\n"
            f"[\n"
            f"  {{\n"
            f"    \"role\": \"<exact role name from input>\",\n"
            f"    \"aliases\": [\"alias1\", \"alias2\", ...]\n"
            f"  }},\n"
            f"  ...\n"
            f"]"
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        if response.stop_reason == "max_tokens":
            raise ValueError(
                f"LLM response truncated (max_tokens hit) for category '{category}'. "
                f"Increase generation.max_tokens in config.yaml."
            )

        content = _strip_code_fences(response.content[0].text)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON for category '{category}': {e}\n"
                f"Raw content:\n{response.content[0].text[:500]}"
            ) from e

        if not isinstance(result, list):
            raise ValueError(
                f"Expected JSON array for category '{category}', "
                f"got {type(result).__name__}."
            )

        for entry in result:
            if not isinstance(entry, dict) or "role" not in entry:
                raise ValueError(
                    f"Invalid entry in LLM response for category '{category}': "
                    f"expected dict with 'role' key, got {type(entry).__name__}"
                )
            role_name = entry["role"]
            if not isinstance(role_name, str):
                raise ValueError(
                    f"Invalid 'role' value in LLM response: expected str, "
                    f"got {type(role_name).__name__}"
                )
            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                raise ValueError(
                    f"Invalid 'aliases' for role '{role_name}': expected list of strings"
                )

            # Deduplicate case-insensitively, exclude original role name
            seen: set[str] = set()
            role_lower = role_name.lower()
            unique_aliases: list[str] = []
            for alias in aliases:
                alias_lower = alias.lower()
                if alias_lower == role_lower:
                    continue
                if alias_lower not in seen:
                    seen.add(alias_lower)
                    unique_aliases.append(alias)

            # Look up the source role ID and category
            if role_name in role_lookup:
                source_role_id, cat = role_lookup[role_name]
            else:
                # Fallback: use the category from the current iteration
                source_role_id = "T-role-0000"
                cat = category

            for alias_text in unique_aliases:
                target_counter += 1
                all_targets.append({
                    "id": f"T-raug-{target_counter:05d}",
                    "text": alias_text,
                    "role": role_name,
                    "category": cat,
                    "granularity": "role_augmented",
                    "source_role_id": source_role_id,
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_targets, f, indent=2)

    return all_targets


def load_augmented_targets(output_path: str) -> list[dict]:
    """Load augmented targets from a JSON file.

    Args:
        output_path: Path to the augmented targets JSON file.

    Returns:
        List of augmented target dicts.
    """
    with open(output_path) as f:
        return json.load(f)
