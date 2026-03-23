# Spec 01: Query Preprocessor (C1)

**Technique:** T2 — Abbreviation Dictionary Expansion
**Phase:** 1 (Quick Wins)
**Scope:** Small (~80 lines)
**References:** [architecture.md](architecture.md) C1, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/preprocess.py` module that expands abbreviations in query strings before encoding. Reverses the existing `_ABBREVIATIONS` dict from `generate_rules.py` (which maps full→abbrev for test generation) into an abbrev→full mapping for query expansion.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/preprocess.py` | New module |
| Modify | `src/embed.py` | Call `expand_abbreviations()` before encoding |

## Interface Contract

### Exposes

```python
# src/preprocess.py

# Module-level constant — loaded once at import time
ABBREVIATION_MAP: dict[str, str]
# Reverse of generate_rules._ABBREVIATIONS:
# {
#     "mgr": "Manager",
#     "eng": "Engineer",
#     "spec": "Specialist",
#     "admin": "Administrator",
#     "coord": "Coordinator",
#     "dir": "Director",
#     "anlst": "Analyst",
#     "rep": "Representative",
#     "dev": "Developer",
#     "asst": "Assistant",
#     "ops": "Operations",
#     "assoc": "Associate",
#     "conslt": "Consultant",
#     "acct": "Accountant",
# }

def expand_abbreviations(query: str) -> str:
    """Expand known abbreviations in a query string.

    - Case-insensitive matching
    - Word-boundary aware (whole words only, not substrings)
    - Returns original query unchanged if no abbreviations found
    - Multiple abbreviations in one query are all expanded
    """
```

### Consumes

- `generate_rules._ABBREVIATIONS` (imported to build reverse map)

## Implementation Steps

### Step 1: Build reverse abbreviation map

**What:** Create `ABBREVIATION_MAP` by reversing `_ABBREVIATIONS` from `generate_rules.py`. Keys are lowercased abbreviations, values are full forms.

**Acceptance criteria:**
- [ ] `ABBREVIATION_MAP` has 14 entries (same count as `_ABBREVIATIONS`)
- [ ] All keys are lowercase
- [ ] Values are original full forms (title case as in source)
- [ ] Map is built at module import time (not per-call)

### Step 2: Implement `expand_abbreviations()`

**What:** Replace abbreviation tokens in query with full forms using regex word boundary matching.

**Acceptance criteria:**
- [ ] `expand_abbreviations("Sr Dev Mgr")` returns `"Sr Developer Manager"`
- [ ] `expand_abbreviations("admin assistant")` returns `"Administrator assistant"` (case-insensitive match)
- [ ] `expand_abbreviations("DevOps Engineer")` returns `"DevOps Engineer"` (no match — "DevOps" is not the same as "Dev" due to word boundary)
- [ ] `expand_abbreviations("unknown title")` returns `"unknown title"` (no-op)
- [ ] `expand_abbreviations("")` returns `""`
- [ ] Uses `re.sub()` with `\b` word boundaries and `re.IGNORECASE`
- [ ] Compiles regex pattern once at module level (not per-call)

### Step 3: Integrate into embed.py

**What:** Call `expand_abbreviations()` on each query before encoding in `run_embedding_model()`.

**Acceptance criteria:**
- [ ] Preprocessing applied to `query_texts` list before `encode_queries()` call
- [ ] Applied unconditionally (no config toggle — abbreviation expansion is always beneficial)
- [ ] Preprocessing happens once, before the granularity loop (not per-granularity)

## Implementation Notes

**Regex pattern:** Compile a single alternation pattern from all abbreviation keys:
```python
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ABBREVIATION_MAP) + r")\b",
    re.IGNORECASE,
)
```

**Replacement function:** Look up matched text (lowercased) in `ABBREVIATION_MAP`.

## Testing Strategy

- Unit tests for `expand_abbreviations()` covering:
  - Single abbreviation expansion
  - Multiple abbreviations in one query
  - Case insensitivity
  - Word boundary respect (no partial matches)
  - No-op for unknown text
  - Empty string
- Integration: verify `run_embedding_model()` produces rankings with expanded queries (compare result count, not scores — expansion changes scores)
