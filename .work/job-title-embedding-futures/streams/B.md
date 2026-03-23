# Stream B: Phase 2 — Target Augmentation

**Phase:** 2
**Work Items:** W-03 (JEM-r7o), W-04 (JEM-g92)
**Dependencies:** Phase 1 complete (Stream A)
**Parallel with:** Stream C

---

## Overview

Create the target augmentation module and integrate it into the evaluation pipeline. Adds `role_augmented` as a 6th granularity level with ~3,500-7,000 LLM-generated alias targets.

## File Ownership

| Action | File | Work Item |
|--------|------|-----------|
| Create | `src/augment.py` | W-03 |
| Create | `data/taxonomy/augmented_targets.json` | W-03 |
| Modify | `src/targets.py` | W-04 |
| Modify | `src/evaluate.py` | W-04 |

**No overlap with Stream C** (which owns rerank.py, fusion.py, config.yaml, embed.py).

## W-03: Target Augmentation Module (JEM-r7o) — spec 03 steps 1-3

**Spec:** `.work/job-title-embedding-futures/specs/03-target-augmentation.md`

**What:** Create `src/augment.py` with LLM-based alias generation for taxonomy roles.

**Steps:**
1. Write LLM generation function — one API call per category (~42 calls), 5-10 aliases per role, using Claude API (same pattern as `generate_llm.py`)
2. Build augmented target dicts — schema: `{id: "T-raug-{i:04d}", text, role, category, granularity: "role_augmented", source_role_id}`
3. Write caching/loading — save to `data/taxonomy/augmented_targets.json`, load if exists

**Acceptance Criteria:**
- One API call per category (~42 total)
- Each role produces 5-10 aliases, deduplicated
- Original role name NOT included as alias
- Total: ~3,500-7,000 augmented targets
- `role` field = parent role's canonical name
- `generate_augmented_targets()` is idempotent (loads from cache if file exists)
- `load_augmented_targets()` reads and returns the array

**Tests:** Mock LLM call and verify output schema; load from fixture file.

## W-04: Augmentation Integration (JEM-g92) — spec 03 steps 4-5

**Spec:** `.work/job-title-embedding-futures/specs/03-target-augmentation.md`
**Depends on:** W-03 (JEM-r7o)

**What:** Wire augmented targets into the evaluation pipeline.

**Steps:**
1. Extend `build_target_sets()` in `src/targets.py` — check for `data/taxonomy/augmented_targets.json`, if exists add as `target_sets["role_augmented"]`
2. Update `_is_correct()` in `src/evaluate.py` — change condition from `granularity in ("role", "role_desc")` to `granularity in ("role", "role_desc", "role_augmented")`

**Acceptance Criteria:**
- `build_target_sets()` returns `role_augmented` key when file exists, omits when absent
- Existing 5 granularities unchanged
- `_is_correct()` works correctly for `role_augmented` (checks `target["role"]`)
- No other evaluation logic changes

**Tests:** `build_target_sets()` with/without augmented file; `_is_correct()` for role_augmented granularity.

## Completion

Close issues with `bd close <id>` when done. Stream C (reranking+fusion) runs in parallel — no coordination needed.
