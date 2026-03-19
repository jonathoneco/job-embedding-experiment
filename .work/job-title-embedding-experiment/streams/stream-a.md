# Stream A — Data Pipeline (Taxonomy + Test Data)

**Phase 2 items**: W-02 (JEM-h0r), W-03 (JEM-3ky) — sequential
**Phase 3 items**: W-05 (JEM-ccr), W-06 (JEM-g2w) — sequential
**Specs**: 02-taxonomy-targets.md (Phase 2), 03-test-data.md (Phase 3)
**Dependencies**: W-01 (project setup) must be complete before starting

---

## Phase 2: Taxonomy & Targets (Spec 02)

### W-02: Taxonomy Parser + Cluster Definitions + Tests (JEM-h0r)

**Files to create**:
- `src/taxonomy.py` — Parse `job-roles.md` → structured role list
- `src/clusters.py` — Hardcoded subcluster definitions for all 42 categories
- `tests/test_taxonomy.py` — Tests for parsing + clustering

**taxonomy.py functions**:
- `parse_taxonomy(source_path: str) -> list[dict]` — Parse markdown, produce 692 roles across 42 categories
- `get_categories(roles: list[dict]) -> dict[str, list[str]]` — Return {category: [role_names]} mapping

**Parsing rules**:
- `## <Category Name>` starts category, `- <Role Name>` defines role
- Strip whitespace, ignore blanks and `# Job Roles` header
- Preserve parenthetical suffixes like `(AEC)` and `(Office)`

**clusters.py**:
- `build_clusters(roles: list[dict]) -> list[dict]` — Hardcoded subclusters
- Splitting rules: <10 roles → intact (label = category name), 10-15 → 2, 16-25 → 2-3, 26+ → 3-4
- Output: `[{"cluster_label": str, "category": str, "roles": [str, ...]}]`
- ~96 total subclusters. Every role in exactly one cluster. No empty clusters.
- See spec 02 for detailed subcluster definitions of the 5 largest categories

**Tests (test_taxonomy.py)**:
1. `test_parse_taxonomy` — Synthetic markdown (3 categories, 5 roles each). Verify count, structure, edge cases.
2. `test_get_categories` — Verify grouping from parsed roles.
3. `test_build_clusters` — Use real taxonomy. Every role in exactly one cluster, count 80-120, small categories intact.

**Acceptance criteria**:
- parse_taxonomy produces exactly 692 roles across 42 categories
- build_clusters: every role in exactly 1 cluster, 80-120 total clusters, descriptive labels
- All tests pass

---

### W-03: Descriptions + Targets + Prep Orchestrator (JEM-3ky)

**Depends on**: W-02

**Files to create**:
- `src/descriptions.py` — Generate descriptions via Claude API
- `src/targets.py` — Construct 5-level target sets
- `scripts/prep_taxonomy.py` — Orchestrator: taxonomy → clusters → descriptions → targets
- `tests/test_taxonomy.py` — Add tests for target building (parts 4-5)

**descriptions.py**:
- `generate_descriptions(roles: list[dict], config: dict) -> dict`
- 42 API calls for role descriptions + 42 for category keywords = 84 total
- Role prompt: "Generate concise functional descriptions... 10-15 words..."
- Category prompt: "Generate a keyword summary... 5-8 key terms..."
- Output: `{"roles": {name: desc}, "categories": {name: keywords}}`
- Uses `config["generation"]["api_model"]` (claude-sonnet-4-20250514)

**targets.py**:
- `build_target_sets(roles, clusters, descriptions) -> dict[str, list[dict]]`
- 5 granularity levels: role (692), role_desc (692), cluster (~96), category_desc (42), category (42)
- Target IDs: `T-role-NNNN`, `T-rdesc-NNNN`, `T-clust-NNNN`, `T-cdesc-NNNN`, `T-cat-NNNN`
- role/role_desc targets: singular `role` + `category` fields
- cluster/category_desc/category targets: `roles` array + `category` field (no singular `role`)

**prep_taxonomy.py orchestrator**:
```python
roles = parse_taxonomy(config["taxonomy"]["source"])
save_json(roles, "data/taxonomy/roles.json")
clusters = build_clusters(roles)
save_json(clusters, "data/taxonomy/clusters.json")
descriptions = generate_descriptions(roles, config)
save_json(descriptions, "data/taxonomy/descriptions.json")
target_sets = build_target_sets(roles, clusters, descriptions)
save_json(target_sets, "data/taxonomy/target_sets.json")
```

**Additional tests**:
4. `test_build_target_sets` — Mock inputs, verify target counts, ID formats, text fields, every role reachable.
5. `test_is_correct_with_accept_sets` — (if applicable at this stage)

**Acceptance criteria**:
- All 692 roles have descriptions (8-20 words each)
- All 42 categories have keyword summaries (5-8 terms)
- Target sets: role=692, role_desc=692, cluster=80-120, category_desc=42, category=42
- Every target has unique ID, non-empty text
- Every role appears in at least one target at each granularity level
- Orchestrator runs end-to-end producing all 4 JSON outputs

**Environment**: Requires `ANTHROPIC_API_KEY` for descriptions.py

---

## Phase 3: Test Data (Spec 03)

### W-05: Rule-Based Generator + Validator + Tests (JEM-ccr)

**Depends on**: W-03

**Files to create**:
- `src/generate_rules.py` — Rule-based easy case generation (120 cases)
- `src/validate.py` — Schema validation, dedup, taxonomy check, split
- `tests/test_validate.py` — Unit tests for validation logic

**generate_rules.py**:
- `generate_rule_cases(roles: list[dict], seed: int) -> list[dict]`
- 6 transform types: level prefix (25), level suffix (20), abbreviation (25), word reorder (15), minor rewording (20), case variation (15)
- Stratified by category (~3 roles per category)
- All difficulty="easy", source="rule-based"

**validate.py**:
- `validate_cases(cases, roles) -> list[dict]` — Schema check, taxonomy membership, ID uniqueness, input uniqueness, difficulty distribution, category coverage
- `deduplicate_cases(cases, threshold=0.85) -> list[dict]` — Character-level Jaccard similarity
- `split_dev_test(cases, dev_size, seed) -> tuple[list, list]` — Stratified split preserving difficulty + category distribution

**Tests (test_validate.py)**:
1. Schema validation with missing/wrong fields
2. Taxonomy membership with non-existent roles
3. Deduplication (2 of 3 similar cases removed, higher difficulty kept)
4. Split stratification verification
5. ID assignment (zero-padded sequential)

**Acceptance criteria**:
- generate_rule_cases produces exactly 120 cases, all easy, no duplicates
- Validator catches invalid schemas, missing taxonomy roles, duplicates
- Split produces dev=100, test=650 with proportional distributions
- All tests pass

---

### W-06: LLM Generator + Manual Cases + Orchestrator (JEM-g2w)

**Depends on**: W-05

**Files to create**:
- `src/generate_llm.py` — LLM systematic + adversarial generation
- `data/test-cases/manual.json` — 50 hard + 25 O*NET cases
- `scripts/prep_test_data.py` — Orchestrator: generate → validate → split

**generate_llm.py**:
- `generate_llm_cases(roles: list[dict], config: dict) -> list[dict]`
- Pass 1 (systematic): 305 cases (105 easy + 200 medium), 1 API call per category
- Pass 2 (adversarial medium): 125 cases, 2-3 API calls with cross-category context
- Pass 3 (adversarial hard): 125 cases, 2-3 API calls
- Full prompts in spec 03
- Accept-set extraction from plausible_alternatives

**manual.json**:
- 50 hand-curated hard cases: 10+ abbreviation chains, 10+ cross-category, 10+ creative, 10+ misspellings, 5+ combined-role
- 25 O*NET synonym cases: alternative titles from onetonline.org, difficulty="easy", source="onet"
- All follow spec 00 test case schema

**prep_test_data.py orchestrator**:
```python
rule_cases = generate_rule_cases(roles, seed)
llm_cases = generate_llm_cases(roles, config)
manual_cases = load_json("data/test-cases/manual.json")
all_cases = rule_cases + llm_cases + manual_cases
# Assign IDs, deduplicate, validate, split
```

**Acceptance criteria**:
- LLM generates 555 cases total (305 + 125 + 125)
- Manual file has 75 cases (50 hard + 25 easy)
- Combined total ~750 after dedup
- All correct_roles exist in taxonomy
- Dev set: 100, test set: 650
- Both sets preserve difficulty and category proportions

**Environment**: Requires `ANTHROPIC_API_KEY` for generate_llm.py

---

## Beads Workflow

```bash
# Phase 2
bd update JEM-h0r --status=in_progress  # W-02
# ... implement W-02 ...
bd close JEM-h0r
bd update JEM-3ky --status=in_progress  # W-03
# ... implement W-03 ...
bd close JEM-3ky

# Phase 3
bd update JEM-ccr --status=in_progress  # W-05
# ... implement W-05 ...
bd close JEM-ccr
bd update JEM-g2w --status=in_progress  # W-06
# ... implement W-06 ...
bd close JEM-g2w
```
