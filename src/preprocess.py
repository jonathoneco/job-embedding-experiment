"""Query preprocessing for job title matching."""

import re

from src.generate_rules import _ABBREVIATIONS

# Reverse map: lowercase abbreviation → original title-case full word
ABBREVIATION_MAP: dict[str, str] = {
    abbr.lower(): full for full, abbr in _ABBREVIATIONS.items()
}

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ABBREVIATION_MAP) + r")\b",
    re.IGNORECASE,
)


def expand_abbreviations(query: str) -> str:
    """Expand known abbreviations in a query string.

    Uses word-boundary matching so partial matches (e.g. "Dev" in "DevOps")
    are not expanded.
    """
    if not query:
        return query
    return _PATTERN.sub(lambda m: ABBREVIATION_MAP[m.group(0).lower()], query)
