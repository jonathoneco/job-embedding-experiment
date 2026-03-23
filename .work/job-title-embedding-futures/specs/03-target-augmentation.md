# Spec 03: Target Augmentation (C3)

**Technique:** T3 — Target-Side Augmentation
**Phase:** 2 (Pipeline Extensions)
**Scope:** Moderate (~150 lines + data generation)
**References:** [architecture.md](architecture.md) C3, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/augment.py` module that generates alias variants for each taxonomy role via LLM. Creates a new 6th granularity level `role_augmented` with ~3,500-7,000 targets. Each augmented target maps back to its parent role via the `role` field, ensuring evaluation compatibility with `_is_correct()`.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/augment.py` | New module: LLM alias generation + target building |
| Create | `data/taxonomy/augmented_targets.json` | Generated: augmented target data |
| Modify | `src/targets.py` | Extend `build_target_sets()` to include `role_augmented` |
| Modify | `src/evaluate.py` | Add `"role_augmented"` to `_is_correct()` dispatch |

## Interface Contract

### Exposes (src/augment.py)

```python
def generate_augmented_targets(
    roles: list[dict],          # [{"role": str, "category": str}, ...]
    config: dict,               # full config (for api_model, max_tokens)
    output_path: str,           # path to write augmented_targets.json
) -> list[dict]:
    """Generate alias variants for each role and write to disk.

    Returns list of augmented target dicts in the role_augmented schema.
    Idempotent: if output_path exists, loads and returns cached result.
    """

def load_augmented_targets(
    output_path: str,
) -> list[dict]:
    """Load previously generated augmented targets from disk."""
```

### Augmented target schema

```python
{
    "id": "T-raug-{i:04d}",     # sequential across all augmented targets
    "text": str,                  # the alias text (e.g., "IT Admin", "Sysadmin")
    "role": str,                  # parent role name (for _is_correct() matching)
    "category": str,              # parent category name
    "granularity": "role_augmented",
    "source_role_id": str,        # reference to parent T-role-NNNN (for traceability)
}
```

**The `role` field** is the parent role's canonical name. This ensures `_is_correct()` works: when evaluating `role_augmented`, it checks `target["role"] in correct_role_names`.

### Consumes

- Taxonomy roles (same input as `build_target_sets()`)
- Claude API via `generation` config (same pattern as `generate_llm.py`)

## Implementation Steps

### Step 1: Write LLM generation function

**What:** For each role, generate 5-10 alias variants via LLM. Batch roles by category for efficiency (one API call per category).

**Prompt template:**
```
For each job role below, generate 5-10 alternative job titles that someone might
use to refer to this exact role. Include:
- Common abbreviations (e.g., "Dev" for "Developer")
- Industry jargon and slang
- Regional variants
- Informal titles
- Compound titles that combine the role with a level or specialty

Category: {category}
Roles:
{role_list}

Return JSON array:
[
  {
    "role": "<exact role name from input>",
    "aliases": ["alias1", "alias2", ...]
  },
  ...
]
```

**Acceptance criteria:**
- [ ] One API call per category (~42 calls total)
- [ ] Uses same API pattern as `generate_llm.py` (Claude model from config, json parsing, stop reason check)
- [ ] Each role produces 5-10 aliases
- [ ] Total augmented targets: ~3,500-7,000 (692 roles x 5-10 aliases)
- [ ] Aliases are deduplicated per role (no exact duplicates)
- [ ] Original role name is NOT included as an alias (it's already in the `role` granularity)

### Step 2: Build augmented target dicts

**What:** Convert LLM output into target dicts matching the `role_augmented` schema (see S0.3).

**Acceptance criteria:**
- [ ] Each alias becomes one target dict with sequential ID `T-raug-{i:04d}`
- [ ] `role` field set to parent role's canonical name
- [ ] `category` field set to parent category
- [ ] `granularity` set to `"role_augmented"`
- [ ] `source_role_id` references the parent role's target ID

### Step 3: Write caching and loading

**What:** Save to `data/taxonomy/augmented_targets.json`. Load if already exists.

**Acceptance criteria:**
- [ ] `generate_augmented_targets()` checks if `output_path` exists; if so, loads and returns
- [ ] File format: JSON array of target dicts
- [ ] `load_augmented_targets()` reads and returns the array

### Step 4: Extend `build_target_sets()` in targets.py

**What:** Add `role_augmented` to the returned dict if augmented targets file exists.

**Acceptance criteria:**
- [ ] `build_target_sets()` checks for `data/taxonomy/augmented_targets.json`
- [ ] If file exists, loads and adds as `target_sets["role_augmented"]`
- [ ] If file does not exist, `role_augmented` key is absent (not an error — graceful degradation)
- [ ] Existing 5 granularity levels unchanged

### Step 5: Update `_is_correct()` in evaluate.py

**What:** Add `"role_augmented"` to the branch that checks `target["role"]`.

**Acceptance criteria:**
- [ ] `_is_correct()` condition changes from `granularity in ("role", "role_desc")` to `granularity in ("role", "role_desc", "role_augmented")`
- [ ] No other changes to evaluation logic

## Testing Strategy

- Unit test for `generate_augmented_targets()`: mock LLM call, verify output schema
- Unit test for `load_augmented_targets()`: load from fixture file
- Unit test for extended `build_target_sets()`: verify `role_augmented` key present when file exists, absent when not
- Unit test for `_is_correct()`: verify correct behavior for `role_augmented` granularity
- Integration: run evaluation on `role_augmented` targets, verify metrics are computed
