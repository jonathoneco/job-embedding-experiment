"""Generate role descriptions and category keywords via Claude API."""

import json

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


def generate_descriptions(roles: list[dict], config: dict) -> dict:
    """Call Claude API to generate role descriptions and category keywords.

    Makes 84 total API calls: 42 for role descriptions, 42 for category
    keywords.

    Args:
        roles: List of {"role": str, "category": str} dicts.
        config: Parsed config.yaml with generation settings.

    Returns:
        {"roles": {name: description, ...},
         "categories": {name: keywords, ...}}

    Raises:
        anthropic.APIError: If any API call fails.
        ValueError: If JSON parsing fails for role descriptions.
    """
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    model = config["generation"]["api_model"]
    max_tokens = config["generation"]["max_tokens"]

    categories = get_categories(roles)

    role_descriptions: dict[str, str] = {}
    category_keywords: dict[str, str] = {}

    for category_name, role_names in categories.items():
        # ── Role descriptions (1 call per category) ───────────────
        roles_text = "\n".join(role_names)
        role_response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                "Generate concise functional descriptions for job roles. "
                "Each description should be 10-15 words capturing the core "
                "day-to-day responsibility. Be specific and concrete — focus "
                "on what the person actually does, not aspirational language."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f'Generate a one-line functional description for each '
                    f'role in the "{category_name}" category. Return valid '
                    f'JSON mapping role name to description.\n\n'
                    f'Roles:\n{roles_text}\n\n'
                    f'Example output:\n'
                    f'{{\n'
                    f'  "HR Business Partner": "Strategic HR advisor who '
                    f'aligns people strategy with business objectives",\n'
                    f'  "Recruiter": "Sources, screens, and hires candidates '
                    f'to fill open positions"\n'
                    f'}}'
                ),
            }],
        )

        # Parse JSON from response
        response_text = role_response.content[0].text
        # Strip markdown code fences if present
        cleaned = _strip_code_fences(response_text)

        try:
            descriptions = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse role descriptions for category "
                f"'{category_name}': {e}\n"
                f"Response text: {response_text[:500]}"
            ) from e

        if not isinstance(descriptions, dict):
            raise ValueError(
                f"Expected dict for category '{category_name}', "
                f"got {type(descriptions).__name__}"
            )

        missing = [r for r in role_names if r not in descriptions]
        if missing:
            raise ValueError(
                f"LLM omitted descriptions for {len(missing)} roles "
                f"in '{category_name}': {missing[:5]}"
            )

        role_descriptions.update(descriptions)

        # ── Category keywords (1 call per category) ───────────────
        roles_csv = ", ".join(role_names)
        keyword_response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                "Generate a brief keyword summary for job categories. "
                "List 5-8 key functional terms that distinguish this "
                "category from others. These terms should help a matching "
                "system identify roles belonging to this category."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f'Generate a keyword summary for the "{category_name}" '
                    f'category, which contains these roles: {roles_csv}\n\n'
                    f'Return a single line of 5-8 comma-separated key terms.\n'
                    f'Example: "recruitment, compensation, benefits, '
                    f'employee relations, talent management"'
                ),
            }],
        )

        keywords = keyword_response.content[0].text.strip()
        # Strip quotes if the model wraps the response
        if keywords.startswith('"') and keywords.endswith('"'):
            keywords = keywords[1:-1]
        category_keywords[category_name] = keywords

    return {"roles": role_descriptions, "categories": category_keywords}
