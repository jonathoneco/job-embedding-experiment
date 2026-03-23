# Stream A: Phase 1 — Quick Wins

**Phase:** 1
**Work Items:** W-01 (JEM-1ub), W-02 (JEM-2me)
**Dependencies:** None (first phase)

---

## Overview

Implement two small improvements to the existing embedding pipeline: abbreviation expansion in query preprocessing and instruction prefix for BGE models. Both are additive, config-compatible, and touch `src/embed.py` (different sections).

## File Ownership

| Action | File | Work Item |
|--------|------|-----------|
| Create | `src/preprocess.py` | W-01 |
| Modify | `src/embed.py` | W-01 (preprocessing call), W-02 (encode_queries prompt kwarg) |
| Modify | `config.yaml` | W-02 (instruction field per model) |

## W-01: Query Preprocessor (JEM-1ub) — spec 01

**Spec:** `.work/job-title-embedding-futures/specs/01-query-preprocessor.md`

**What:** Create `src/preprocess.py` with `expand_abbreviations()`. Reverse the 14-entry `_ABBREVIATIONS` dict from `generate_rules.py` into an abbrev→full map. Use compiled regex with `\b` word boundaries and `re.IGNORECASE`.

**Steps:**
1. Build `ABBREVIATION_MAP` by reversing `_ABBREVIATIONS` (14 entries, lowercase keys)
2. Implement `expand_abbreviations(query: str) -> str` with compiled regex pattern
3. Integrate into `run_embedding_model()` — apply to `query_texts` before `encode_queries()`, unconditionally, once before the granularity loop

**Acceptance Criteria:**
- `ABBREVIATION_MAP` has 14 entries, built at import time
- `expand_abbreviations("Sr Dev Mgr")` → `"Sr Developer Manager"`
- `expand_abbreviations("DevOps Engineer")` → `"DevOps Engineer"` (word boundary)
- `expand_abbreviations("")` → `""`
- Regex compiled once at module level
- Preprocessing applied unconditionally before granularity loop

**Tests:** Unit tests for single/multiple abbreviation, case insensitivity, word boundaries, no-op, empty string.

## W-02: Instruction Prefix (JEM-2me) — spec 02

**Spec:** `.work/job-title-embedding-futures/specs/02-instruction-prefix.md`

**What:** Add `prompt: str | None = None` parameter to `encode_queries()`. BGE models get `instruction: "Represent this sentence for searching relevant passages: "` in config. Thread through `run_embedding_model()`.

**Steps:**
1. Add `instruction` field to BGE model entries in `config.yaml` (not to minilm)
2. Add `prompt` parameter to `encode_queries()`, pass to `model.encode()`
3. In `run_embedding_model()`, read `model_config.get("instruction")` and pass as `prompt=` to `encode_queries()`. NOT to `encode_targets()`.

**Acceptance Criteria:**
- `bge-base` and `bge-large` have `instruction` field in config
- `minilm` has no `instruction` field
- `encode_queries()` accepts and passes `prompt` kwarg to `model.encode()`
- `prompt=None` is a no-op
- L2 normalization still applied
- `encode_targets()` is NOT modified
- Method name unchanged (instruction is transparent)

**Tests:** Config loads correctly, embeddings differ with/without prompt for BGE, targets unaffected, backward compatibility.

## Completion

After both items: run full experiment with existing configs to verify backward compatibility (21 configs produce valid results). Claim issues with `bd update <id> --status=in_progress` before starting, close with `bd close <id>` when done.
