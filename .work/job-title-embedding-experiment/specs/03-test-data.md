# Spec 03 — Test Data (C2)

**Dependencies**: C1 (taxonomy, target sets for validation)
**Refs**: Spec 00 (test case schema, config, dev set policy)

---

## Overview

Generate 750 test cases via three methods (rule-based, LLM, manual), validate and deduplicate, then split into dev (100) and test (650) sets. All outputs are committed JSON files under `data/test-cases/`.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/generate_rules.py` | Rule-based easy case generation |
| `src/generate_llm.py` | LLM systematic + adversarial generation |
| `src/validate.py` | Schema validation, dedup, taxonomy check |
| `data/test-cases/manual.json` | Hand-curated hard cases + O*NET synonyms |
| `scripts/prep_test_data.py` | Orchestrator: generate → validate → split |
| `tests/test_validate.py` | Unit tests for validation logic |

---

## Difficulty Distribution

| Method | Easy | Medium | Hard | Total |
|--------|------|--------|------|-------|
| Rule-based transforms | 120 | — | — | 120 |
| LLM systematic (Pass 1) | 105 | 200 | — | 305 |
| LLM adversarial (Pass 2-3) | — | 125 | 125 | 250 |
| Manual curation | — | — | 50 | 50 |
| O*NET synonyms | 25 | — | — | 25 |
| **Total** | **250** | **325** | **175** | **750** |

---

## Subcomponent: Rule-Based Generator (`src/generate_rules.py`)

### Function: `generate_rule_cases(roles: list[dict], seed: int) -> list[dict]`

Generate 120 easy test cases via deterministic text transforms.

**Transform types** (applied per role, ~3-4 roles per transform to reach 120):

| Transform | Example Input | Example Output | Count |
|-----------|--------------|----------------|-------|
| Level prefix | `"HR Business Partner"` | `"Senior HR Business Partner"` | 25 |
| Level suffix | `"Software Engineer"` | `"Software Engineer III"` | 20 |
| Common abbreviation | `"Human Resources"` | `"HR"` (in title) | 25 |
| Word reordering | `"Sales Operations Analyst"` | `"Operations Analyst, Sales"` | 15 |
| Minor rewording | `"Customer Service Representative"` | `"Customer Service Rep"` | 20 |
| Case variation | `"Data Scientist"` | `"data scientist"` | 15 |

**Role selection**: Stratified by category — select ~3 roles per category (120 / 42 ≈ 3), randomized with seed.

**Acceptance criteria**:
- Exactly 120 cases produced
- All difficulty = `"easy"`, source = `"rule-based"`
- Each case maps to exactly one correct role via the transform (no ambiguity)
- No duplicate input titles

---

## Subcomponent: LLM Generator (`src/generate_llm.py`)

### Function: `generate_llm_cases(roles: list[dict], config: dict) -> list[dict]`

Multi-pass Claude API generation. Returns combined list of all LLM-generated cases.

### Pass 1 — Systematic (305 cases: 105 easy + 200 medium)

**Prompt**:

```
System: You are generating realistic job title variations for testing a matching
system. Create titles that a real person would actually use on their resume or
LinkedIn profile. Each title must map to exactly one role in the provided taxonomy.

User: Generate {n} job title variations for roles in the "{category}" category.

Target distribution:
- {easy_count} easy: common abbreviations, standard level prefixes (Sr., Jr., Lead),
  well-known synonyms
- {medium_count} medium: industry jargon, creative rewording, combined responsibilities,
  department-specific language

For each variation, output JSON:
{
  "input_title": "the realistic title variation",
  "correct_role": "exact role name from taxonomy",
  "difficulty": "easy|medium",
  "variation_type": "abbreviation|synonym|creative|jargon|level-prefix|combined-role",
  "notes": "brief explanation of why this maps to the correct role"
}

Roles in this category:
{newline-separated role names}

Requirements:
- Each input_title must be unique and realistic
- The correct_role must exactly match one of the listed roles
- Easy titles should be obviously recognizable variants
- Medium titles require domain knowledge to map correctly
```

**Batching**: One API call per category. ~7-8 cases per category (305 / 42 ≈ 7.3). Proportional to category size — larger categories get more cases.

**Overflow handling**: If total generated exceeds target counts per difficulty, trim by removing cases with the shortest edit distance to another case's input_title.

### Pass 2 — Adversarial Medium (125 cases)

**Prompt**:

```
System: You are an adversarial tester trying to create challenging but fair job
title variations. These titles should be realistic (someone might actually use them)
but difficult for an automated system to match correctly. Focus on cross-category
confusion and ambiguous terminology.

User: Generate {n} medium-difficulty job title variations that are particularly
challenging. Focus on:
- Titles that could plausibly belong to multiple categories
- Industry-specific jargon that obscures the role
- Titles that emphasize a secondary responsibility
- Informal/startup-style titles ("Head of People", "Revenue Lead")

Each variation must map to exactly one primary role, but you should note if there
are plausible alternatives.

Output JSON array with fields: input_title, correct_role, correct_category,
difficulty ("medium"), variation_type, notes, plausible_alternatives (list of
{"role": str, "category": str})

Available roles (by category):
{category: [roles] for all categories}
```

**Batching**: 2-3 API calls, processing all categories together. The adversarial prompt needs cross-category context.

**Accept-set extraction**: If `plausible_alternatives` are provided, merge them into `correct_roles` as the accept set.

### Pass 3 — Adversarial Hard (125 cases)

**Prompt**:

```
System: You are creating the hardest possible test cases for a job title matching
system. These should be titles that even a human recruiter might need a moment to
place. Every title must still be realistic — something that could appear on a real
LinkedIn profile.

User: Generate {n} hard-difficulty job title variations. Techniques:
- Heavy abbreviation chains ("Sr. HRBP" for Senior HR Business Partner)
- Creative/startup titles ("Growth Wizard", "People Ops Ninja")
- Cross-functional titles that blend two domains ("Marketing Data Analyst")
- Misspellings or non-standard formatting ("Softwre Engneer")
- Regional/industry-specific terms ("Sachbearbeiter" for Administrative Specialist)
- Titles that describe the work, not the role ("Someone Who Makes Spreadsheets Work")

Each must map to one primary correct role. Note plausible alternatives.

Output JSON array with fields: input_title, correct_role, correct_category,
difficulty ("hard"), variation_type, notes, plausible_alternatives

Available roles (by category):
{category: [roles] for all categories}
```

**Batching**: 2-3 API calls.

**Acceptance criteria for all LLM passes**:
- Combined output: 305 + 125 + 125 = 555 cases
- All `correct_role` values exist in the taxonomy
- No duplicate `input_title` values across passes
- Difficulty distribution within ±10% of targets
- Variation types are diverse (no more than 30% of any single type)

---

## Subcomponent: Manual Cases (`data/test-cases/manual.json`)

### 50 Hand-Curated Hard Cases

Created manually during implementation. Must include:
- 10+ cases with heavy abbreviation chains
- 10+ cases with cross-category ambiguity (accept sets with 2+ correct roles)
- 10+ cases with creative/non-standard titles
- 10+ cases with misspellings or non-English terms
- 5+ cases with combined-role titles

### 25 O*NET Synonym Cases

Source alternative titles from O*NET OnLine (onetonline.org) for 25 roles. These are official U.S. Department of Labor alternative titles — high quality, known-correct.

Format: Same test case schema. Difficulty = `"easy"`, source = `"onet"`.

**Acceptance criteria**:
- 75 total manual cases (50 hard + 25 easy)
- All correct_roles exist in taxonomy
- All cases follow spec 00 test case schema
- Each case has a descriptive `notes` field explaining the challenge

---

## Subcomponent: Validator (`src/validate.py`)

### Function: `validate_cases(cases: list[dict], roles: list[dict]) -> list[dict]`

Run all validation checks. Raises `ValueError` on any failure.

**Validation rules**:

1. **Schema check**: Every case matches spec 00 test case schema. Required fields present, types correct.
2. **Taxonomy membership**: Every role in `correct_roles` exists in `roles.json`. Exact string match.
3. **ID uniqueness**: No duplicate `id` fields.
4. **Input uniqueness**: No duplicate `input_title` values (case-insensitive).
5. **Difficulty distribution**: Totals match targets (250/325/175) within ±5 cases after trimming.
6. **Category coverage**: Every category has at least 5 test cases. Alert (not fail) if any category has fewer than 10.

### Function: `deduplicate_cases(cases: list[dict], threshold: float = 0.85) -> list[dict]`

Remove near-duplicate input titles using character-level Jaccard similarity.

- Compute pairwise similarity on `input_title` (lowercased)
- If similarity > threshold, keep the one with higher difficulty rating
- Log removed cases to stderr

### Function: `split_dev_test(cases: list[dict], dev_size: int, seed: int) -> tuple[list[dict], list[dict]]`

Stratified random split preserving:
- Difficulty distribution (proportional in both splits)
- Category distribution (proportional in both splits)

Use `sklearn.model_selection.train_test_split` with `stratify` on a combined difficulty+category key.

**Acceptance criteria**:
- Dev set: exactly 100 cases
- Test set: exactly 650 cases
- Both sets preserve difficulty proportions (within ±5%)
- Both sets have representation from all 42 categories

---

## Orchestrator: `scripts/prep_test_data.py`

Sequential pipeline:

```python
roles = load_json("data/taxonomy/roles.json")

# Generate
rule_cases = generate_rule_cases(roles, config["experiment"]["seed"])
llm_cases = generate_llm_cases(roles, config)
manual_cases = load_json("data/test-cases/manual.json")

# Combine and assign IDs
all_cases = rule_cases + llm_cases + manual_cases
for i, case in enumerate(all_cases):
    case["id"] = f"TC-{i+1:04d}"

# Validate
all_cases = deduplicate_cases(all_cases)
validate_cases(all_cases, roles)

# Save raw
save_json(all_cases, "data/test-cases/raw_cases.json")

# Split
dev, test = split_dev_test(all_cases, config["test_data"]["dev_size"], config["experiment"]["seed"])
save_json(dev, "data/test-cases/dev.json")
save_json(test, "data/test-cases/test.json")
```

---

## Testing Strategy (`tests/test_validate.py`)

1. **Test schema validation**: Feed cases with missing fields, wrong types, invalid difficulty. Verify `ValueError` raised.
2. **Test taxonomy membership**: Feed cases with non-existent role names. Verify rejection.
3. **Test deduplication**: Feed 3 cases where 2 have >0.85 similarity. Verify one removed, higher difficulty kept.
4. **Test split stratification**: Feed 100 cases with known difficulty/category distribution. Verify both splits preserve proportions.
5. **Test ID assignment**: Verify zero-padded sequential IDs.
6. **No tests for generators**: LLM and rule-based generators are validated via the validator. Generator output is reviewed during implementation.
